from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess zh-en jsonl for NMT (RNN/Transformer).

Unified pipeline (NO-LEAK):
Mode = fit (TRAIN ONLY):
1) read jsonl
2) clean text
3) char-level length filter/truncate (before tokenization)
4) tokenize by strategy: word / spm_bpe / wordpiece
5) (train only) compute low-freq sets from TRAIN tokens (exclude SPECIAL_TOKENS)
6) apply low-freq policy on TRAIN
7) token-level length filter/truncate
8) build vocab from TRAIN tokens (exclude SPECIAL_TOKENS) + save (with freq)
9) convert to ids (add <bos>/<eos>) + save processed_*.jsonl
10) save preprocess_config.json for reproducibility

Mode = transform (DEV/TEST):
1) read jsonl
2) clean text
3) char-level length filter/truncate
4) tokenize using TRAIN artifacts (spm models / wordpiece models / word tokenizer)
5) apply OOV policy based on TRAIN vocab ONLY (NO frequency statistics on dev/test)
6) token-level length policy
7) convert to ids using TRAIN vocab + save processed_*.jsonl
8) save preprocess_config.json
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from tqdm import tqdm



# -----------------------------
# Special tokens (UNIFIED)
# -----------------------------
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = SPECIAL_TOKENS
############################
def build_vocab_from_spm_model(spm_model_path: str) -> Vocab:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    if not os.path.isfile(spm_model_path):
        raise FileNotFoundError(f"Missing SPM model: {spm_model_path}")
    sp.load(spm_model_path)

    itos = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    # sanity: we configured pad/unk/bos/eos ids to be 0/1/2/3
    if len(itos) < 4 or itos[:4] != SPECIAL_TOKENS:
        raise ValueError(
            f"SPM pieces[0:4] must be {SPECIAL_TOKENS}, but got {itos[:4]}. "
            f"Check your SentencePieceTrainer settings (pad_id/unk_id/bos_id/eos_id and *_piece)."
        )

    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def freq_from_token_seqs(token_seqs: Iterable[List[str]]) -> Counter:
    c = Counter()
    for seq in token_seqs:
        c.update([t for t in seq if t not in SPECIAL_TOKENS])
    return c
#################
def strip_special_tokens(tokens: List[str], keep_unk: bool = True) -> List[str]:
    """Remove BOS/EOS/PAD from token sequence; optionally keep UNK."""
    if keep_unk:
        drop = {PAD, BOS, EOS}
    else:
        drop = set(SPECIAL_TOKENS)
    return [t for t in tokens if t not in drop]


# -----------------------------
# Cleaning utilities
# -----------------------------
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_REPLACEMENT = re.compile(r"\uFFFD+")  # '�'


def _remove_control_chars_keep_whitespace(text: str) -> str:
    """Remove Unicode control chars but keep \t \n \r as whitespace."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            if ch in ("\t", "\n", "\r"):
                out.append(ch)
            else:
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text: str, lang: str, lower_en: bool = True) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _remove_control_chars_keep_whitespace(text)
    text = _RE_REPLACEMENT.sub("", text)
    text = (text.replace("“", '"').replace("”", '"')
                .replace("’", "'")
                .replace("–", "-").replace("—", "-"))
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    if lang == "en" and lower_en:
        text = text.lower()
    return text


# -----------------------------
# Char-level length filter/truncate
# -----------------------------
def char_length_filter(text: str, max_len: int, policy: str) -> Optional[str]:
    """
    max_len<=0 => disabled
    policy: filter|truncate
    truncate: try to cut at word boundary if there is a space (mainly English)
    """
    if max_len <= 0:
        return text

    if policy == "filter":
        return text if len(text) <= max_len else None

    if policy == "truncate":
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_space = truncated.rfind(" ")
        if last_space != -1 and last_space > int(max_len * 0.5):
            truncated = truncated[:last_space].strip()
        else:
            truncated = truncated.strip()
        return truncated if truncated else None

    raise ValueError("char_len_policy must be filter|truncate")


# -----------------------------
# Word-level tokenizers
# -----------------------------
def tokenize_en_word(text: str, use_nltk: bool = True) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if use_nltk:
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]", text)


_HANLP_TOKENIZER = None


def _load_hanlp_tokenizer(model_id: str):
    global _HANLP_TOKENIZER
    if _HANLP_TOKENIZER is not None:
        return _HANLP_TOKENIZER
    try:
        import hanlp  # noqa
    except Exception as e:
        raise RuntimeError(
            "HanLP is not installed. Install via:\n"
            "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple hanlp\n"
        ) from e
    import hanlp  # type: ignore
    _HANLP_TOKENIZER = hanlp.load(model_id)
    return _HANLP_TOKENIZER


def tokenize_zh_word(text: str, mode: str, hanlp_model: str = "") -> List[str]:
    text = text.strip()
    if not text:
        return []
    if mode == "jieba":
        import jieba
        return [t for t in jieba.lcut(text, cut_all=False) if t.strip()]
    if mode == "char":
        return [ch for ch in text if ch.strip()]
    if mode == "hanlp":
        model_id = hanlp_model
        if not model_id:
            try:
                import hanlp  # type: ignore
                model_id = hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL
            except Exception:
                model_id = "CTB6_CONVSEG"
        tokenizer = _load_hanlp_tokenizer(model_id)
        toks = tokenizer(text)
        return [t for t in toks if isinstance(t, str) and t.strip()]
    raise ValueError(f"Unknown zh_tok: {mode}")


# -----------------------------
# Token-level length policy
# -----------------------------
def apply_length_policy(
    zh_tokens: List[str],
    en_tokens: List[str],
    max_len_zh: int,
    max_len_en: int,
    policy: str,  # filter|truncate
) -> Optional[Tuple[List[str], List[str]]]:
    if policy not in {"filter", "truncate"}:
        raise ValueError("len_policy must be filter|truncate")

    if policy == "filter":
        if (max_len_zh > 0 and len(zh_tokens) > max_len_zh) or (max_len_en > 0 and len(en_tokens) > max_len_en):
            return None
        if not zh_tokens or not en_tokens:
            return None
        return zh_tokens, en_tokens

    if max_len_zh > 0:
        zh_tokens = zh_tokens[:max_len_zh]
    if max_len_en > 0:
        en_tokens = en_tokens[:max_len_en]
    if not zh_tokens or not en_tokens:
        return None
    return zh_tokens, en_tokens


# -----------------------------
# SentencePiece BPE
# -----------------------------
def train_sentencepiece(
    texts: List[str],
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    model_type: str = "bpe",
) -> str:
    import sentencepiece as spm
    tmp_txt = model_prefix + ".train.txt"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.train(
        input=tmp_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece=PAD, unk_piece=UNK, bos_piece=BOS, eos_piece=EOS,
        byte_fallback=True,
    )

    os.remove(tmp_txt)
    return model_prefix + ".model"


# NOTE: keep for backward compatibility (not used in the optimized spm path)
def sp_encode(sp_model_path: str, text: str) -> List[str]:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp.encode(text, out_type=str)


def load_spm_processors(zh_model: str, en_model: str):
    import sentencepiece as spm
    sp_zh = spm.SentencePieceProcessor()
    sp_en = spm.SentencePieceProcessor()
    if not os.path.isfile(zh_model):
        raise FileNotFoundError(f"Missing spm_zh_model: {zh_model}")
    if not os.path.isfile(en_model):
        raise FileNotFoundError(f"Missing spm_en_model: {en_model}")
    sp_zh.load(zh_model)
    sp_en.load(en_model)
    return sp_zh, sp_en


# -----------------------------
# Vocab
# -----------------------------
@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]


def build_vocab(
    token_seqs: Iterable[List[str]],
    min_freq: int,
    max_vocab: int
) -> Tuple[Vocab, Counter]:
    counter = Counter()
    for seq in token_seqs:
        counter.update([t for t in seq if t not in SPECIAL_TOKENS])  # exclude specials from stats

    vocab_list = list(SPECIAL_TOKENS)
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    for tok, freq in items:
        if min_freq > 0 and freq < min_freq:
            continue
        vocab_list.append(tok)
        if max_vocab > 0 and len(vocab_list) >= max_vocab:
            break

    stoi = {t: i for i, t in enumerate(vocab_list)}
    return Vocab(stoi=stoi, itos=vocab_list), counter


def save_vocab_with_freq(path: str, vocab: Vocab, freq: Counter) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in vocab.itos:
            f.write(f"{t}\t{freq.get(t, 0)}\n")


def load_vocab_from_txt(path: str) -> Vocab:
    """
    Read vocab_*.txt where each line is: token<TAB>freq
    Only the token column is required.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing vocab file: {path}")

    itos: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tok = line.split("\t", 1)[0]
            itos.append(tok)

    # minimal sanity: ensure first 4 are special tokens
    if len(itos) < 4 or itos[:4] != SPECIAL_TOKENS:
        raise ValueError(f"Vocab file {path} must start with: {SPECIAL_TOKENS}, but got: {itos[:4]}")

    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def tokens_to_ids(tokens: List[str], vocab: Vocab, add_bos_eos: bool = True) -> List[int]:
    ids = [vocab.bos_id] if add_bos_eos else []
    ids.extend([vocab.stoi.get(t, vocab.unk_id) for t in tokens])
    if add_bos_eos:
        ids.append(vocab.eos_id)
    return ids


# -----------------------------
# Low-freq / OOV policy
# -----------------------------
def apply_low_freq_filter(tokens: List[str], low_freq_set: set, policy: str) -> Optional[List[str]]:
    """
    policy:
      - replace_unk: low-freq token -> <unk>
      - remove_token: drop low-freq tokens from sentence
      - drop_sample: if sentence contains any low-freq token -> drop the sample

    Important:
      - Never mutate SPECIAL_TOKENS.
      - low_freq_set is expected to EXCLUDE SPECIAL_TOKENS.
    """
    if not low_freq_set:
        return tokens

    if policy == "replace_unk":
        out = []
        for t in tokens:
            if t in SPECIAL_TOKENS:
                out.append(t)
            elif t in low_freq_set:
                out.append(UNK)
            else:
                out.append(t)
        return out

    if policy == "remove_token":
        out = [t for t in tokens if (t in SPECIAL_TOKENS) or (t not in low_freq_set)]
        out2 = [t for t in out if t not in {PAD, BOS, EOS}]  # keep UNK if any
        return out if out2 else None

    if policy == "drop_sample":
        if any((t not in SPECIAL_TOKENS) and (t in low_freq_set) for t in tokens):
            return None
        return tokens

    raise ValueError("low_freq_policy must be replace_unk|remove_token|drop_sample")


def apply_oov_policy(tokens: List[str], vocab: Vocab, policy: str) -> Optional[List[str]]:
    """
    DEV/TEST side: no frequency stats, so use vocab membership as "allowed set".
    Tokens NOT in vocab are treated as OOV.
    """
    oov_set = {t for t in tokens if (t not in SPECIAL_TOKENS) and (t not in vocab.stoi)}
    return apply_low_freq_filter(tokens, oov_set, policy)


# -----------------------------
# JSONL IO
# -----------------------------
def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Optional: Embedding init
# -----------------------------
def load_pretrained_vectors(vec_path: str):
    from gensim.models import KeyedVectors
    return KeyedVectors.load_word2vec_format(vec_path, binary=False)


def build_embedding_matrix(
    vocab: Vocab,
    dim: int,
    pretrained_path: str,
    random_scale: float = 0.02,
) -> np.ndarray:
    kv = load_pretrained_vectors(pretrained_path) if pretrained_path else None
    emb = np.random.normal(0.0, random_scale, size=(len(vocab.itos), dim)).astype(np.float32)
    emb[vocab.pad_id] = 0.0
    if kv is None:
        return emb

    hit = 0
    for i, tok in enumerate(vocab.itos):
        if tok in SPECIAL_TOKENS:
            continue
        if tok in kv:
            vec = kv[tok]
            if vec.shape[0] == dim:
                emb[i] = vec
                hit += 1
    print(f"[Embedding] Hit {hit}/{len(vocab.itos)} ({hit/len(vocab.itos):.2%})")
    return emb


# -----------------------------
# Tokenization strategies
# -----------------------------
def tokenize_pairs_word(
    pairs_text: List[Tuple[str, str, Optional[int]]],
    zh_tok: str,
    hanlp_model: str,
    use_nltk: bool,
) -> List[Tuple[List[str], List[str], Optional[int]]]:
    out = []
    for zh, en, idx in pairs_text:
        zh_toks = tokenize_zh_word(zh, mode=zh_tok, hanlp_model=hanlp_model)
        en_toks = tokenize_en_word(en, use_nltk=use_nltk)
        zh_toks = strip_special_tokens(zh_toks, keep_unk=True)
        en_toks = strip_special_tokens(en_toks, keep_unk=True)
        out.append((zh_toks, en_toks, idx))
    return out


def tokenize_pairs_spm_fit(
    pairs_text: List[Tuple[str, str, Optional[int]]],
    output_dir: str,
    spm_vocab_size: int,
    spm_char_cov_zh: float,
    spm_char_cov_en: float,
) -> Tuple[List[Tuple[List[str], List[str], Optional[int]]], str, str]:
    """
    TRAIN ONLY: train SPM models and encode with processors loaded once.
    Returns: (tokenized_pairs, zh_model_path, en_model_path)
    """
    import sentencepiece as spm  # local import

    zh_texts = [p[0] for p in pairs_text]
    en_texts = [p[1] for p in pairs_text]

    zh_model = train_sentencepiece(
        texts=zh_texts,
        model_prefix=os.path.join(output_dir, "spm_zh"),
        vocab_size=spm_vocab_size,
        character_coverage=spm_char_cov_zh,
        model_type="bpe",
    )
    en_model = train_sentencepiece(
        texts=en_texts,
        model_prefix=os.path.join(output_dir, "spm_en"),
        vocab_size=spm_vocab_size,
        character_coverage=spm_char_cov_en,
        model_type="bpe",
    )
    print(f"[SPM] zh_model={zh_model}")
    print(f"[SPM] en_model={en_model}")

    sp_zh = spm.SentencePieceProcessor()
    sp_en = spm.SentencePieceProcessor()
    sp_zh.load(zh_model)
    sp_en.load(en_model)

    out = []
    for zh, en, idx in tqdm(pairs_text, desc="SPM Encode"):
        zh_toks = sp_zh.encode(zh, out_type=str)
        en_toks = sp_en.encode(en, out_type=str)
        zh_toks = strip_special_tokens(zh_toks, keep_unk=True)
        en_toks = strip_special_tokens(en_toks, keep_unk=True)
        out.append((zh_toks, en_toks, idx))
    return out, zh_model, en_model


def tokenize_pairs_spm_transform(
    pairs_text: List[Tuple[str, str, Optional[int]]],
    spm_zh_model: str,
    spm_en_model: str,
) -> List[Tuple[List[str], List[str], Optional[int]]]:
    """
    DEV/TEST: load pre-trained SPM models (from TRAIN) and encode.
    """
    sp_zh, sp_en = load_spm_processors(spm_zh_model, spm_en_model)

    out = []
    for zh, en, idx in tqdm(pairs_text, desc="SPM Encode"):
        zh_toks = sp_zh.encode(zh, out_type=str)
        en_toks = sp_en.encode(en, out_type=str)
        zh_toks = strip_special_tokens(zh_toks, keep_unk=True)
        en_toks = strip_special_tokens(en_toks, keep_unk=True)
        out.append((zh_toks, en_toks, idx))
    return out


def tokenize_pairs_wordpiece(
    pairs_text: List[Tuple[str, str, Optional[int]]],
    zh_wp_model: str,
    en_wp_model: str,
) -> List[Tuple[List[str], List[str], Optional[int]]]:
    """
    WordPiece:
    - Use encode(add_special_tokens=False) to GUARANTEE no [CLS]/[SEP] injected
    - Map tokenizer special tokens to unified tokens if they appear (robustness)
    - Remove <bos>/<eos>/<pad> from token sequence before returning
    """
    try:
        from transformers import BertTokenizer
    except Exception as e:
        raise RuntimeError("WordPiece requires transformers. Install: pip install transformers") from e

    zh_tok = BertTokenizer.from_pretrained(zh_wp_model)
    en_tok = BertTokenizer.from_pretrained(en_wp_model)

    special_mapping_zh = {
        zh_tok.unk_token: UNK,
        zh_tok.cls_token: BOS,
        zh_tok.sep_token: EOS,
        zh_tok.pad_token: PAD,
    }
    special_mapping_en = {
        en_tok.unk_token: UNK,
        en_tok.cls_token: BOS,
        en_tok.sep_token: EOS,
        en_tok.pad_token: PAD,
    }

    out = []
    for zh, en, idx in pairs_text:
        zh_ids = zh_tok.encode(zh, add_special_tokens=False)
        en_ids = en_tok.encode(en, add_special_tokens=False)

        zh_toks = zh_tok.convert_ids_to_tokens(zh_ids)
        en_toks = en_tok.convert_ids_to_tokens(en_ids)

        zh_toks = [special_mapping_zh.get(t, t) for t in zh_toks]
        en_toks = [special_mapping_en.get(t, t) for t in en_toks]

        zh_toks = strip_special_tokens(zh_toks, keep_unk=True)
        en_toks = strip_special_tokens(en_toks, keep_unk=True)

        out.append((zh_toks, en_toks, idx))

    return out


# -----------------------------
# Validate + save config
# -----------------------------
def validate_args(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode not in {"fit", "transform"}:
        raise ValueError("--mode must be fit or transform")

    if args.mode == "fit" and args.tokenization == "wordpiece":
        print("[INFO] WordPiece uses HF tokenizers; vocab is built from TRAIN tokens for this assignment.")

    if args.max_char_len_zh > 0 or args.max_char_len_en > 0:
        print("[INFO] Char-level length limits are applied BEFORE tokenization (--max_char_len_*).")
    if args.max_len_zh > 0 or args.max_len_en > 0:
        print("[INFO] Token-level length limits are applied AFTER tokenization (--max_len_*).")

    # defaults for transform artifacts
    if args.mode == "transform":
        if args.tokenization == "spm_bpe":
            if not args.spm_zh_model:
                args.spm_zh_model = os.path.join(args.output_dir, "spm_zh.model")
            if not args.spm_en_model:
                args.spm_en_model = os.path.join(args.output_dir, "spm_en.model")

        if not args.vocab_zh:
            args.vocab_zh = os.path.join(args.output_dir, "vocab_zh.txt")
        if not args.vocab_en:
            args.vocab_en = os.path.join(args.output_dir, "vocab_en.txt")


def save_config(args) -> None:
    config_path = os.path.join(args.output_dir, "preprocess_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[Config] Saved to {config_path}")


# -----------------------------
# Unified pipeline
# -----------------------------
def process_pipeline(args) -> None:
    validate_args(args)
    save_config(args)

    raw = list(read_jsonl(args.input))
    print(f"[Load] {len(raw)} samples from {args.input}")

    # Clean + char length
    pairs_text: List[Tuple[str, str, Optional[int]]] = []
    dropped_char = 0
    for ex in tqdm(raw, desc="Clean+CharLen"):
        en = clean_text(ex.get("en", ""), lang="en", lower_en=args.lower_en)
        zh = clean_text(ex.get("zh", ""), lang="zh", lower_en=False)

        zh2 = char_length_filter(zh, args.max_char_len_zh, args.char_len_policy)
        en2 = char_length_filter(en, args.max_char_len_en, args.char_len_policy)
        if zh2 is None or en2 is None:
            dropped_char += 1
            continue
        pairs_text.append((zh2, en2, ex.get("index", None)))
    print(f"[CharLen] kept {len(pairs_text)} / dropped {dropped_char}")

    # Tokenize
    if args.tokenization == "word":
        tokenized = tokenize_pairs_word(pairs_text, args.zh_tok, args.hanlp_model, args.use_nltk)

    # elif args.tokenization == "spm_bpe":
    #     if args.mode == "fit":
    #         tokenized, _, _ = tokenize_pairs_spm_fit(
    #             pairs_text, args.output_dir, args.spm_vocab_size, args.spm_char_cov_zh, args.spm_char_cov_en
    #         )
    #     else:
    #         tokenized = tokenize_pairs_spm_transform(pairs_text, args.spm_zh_model, args.spm_en_model)
    elif args.tokenization == "spm_bpe":
        if args.mode == "fit":
            tokenized, zh_model, en_model = tokenize_pairs_spm_fit(
                pairs_text, args.output_dir, args.spm_vocab_size, args.spm_char_cov_zh, args.spm_char_cov_en
            )
            # make artifacts explicit for reproducibility
            args.spm_zh_model = zh_model
            args.spm_en_model = en_model
            save_config(args)  # overwrite config with trained artifact paths
        else:
            tokenized = tokenize_pairs_spm_transform(pairs_text, args.spm_zh_model, args.spm_en_model)

    elif args.tokenization == "wordpiece":
        tokenized = tokenize_pairs_wordpiece(pairs_text, args.zh_wp_model, args.en_wp_model)

    else:
        raise ValueError("Unknown tokenization")

    # Branch by mode
    if args.mode == "fit":
        _fit_and_save(tokenized, args)
    else:
        _transform_and_save(tokenized, args)


def _fit_and_save(tokenized: List[Tuple[List[str], List[str], Optional[int]]], args) -> None:
    """
    TRAIN ONLY:
    - compute low-freq sets from TRAIN tokens
    - apply low-freq policy
    - length policy
    - build vocab from TRAIN tokens
    - save vocab + processed
    """
    is_spm = (args.tokenization == "spm_bpe")
    if is_spm:
        after_lowfreq = tokenized
        dropped_low = 0
        print(f"[LowFreq][SPM] skipped (kept {len(after_lowfreq)})")
    else:
        zh_counter = Counter()
        en_counter = Counter()
        for zh_toks, en_toks, _ in tokenized:
            zh_counter.update([t for t in zh_toks if t not in SPECIAL_TOKENS])
            en_counter.update([t for t in en_toks if t not in SPECIAL_TOKENS])

        zh_low = {t for t, c in zh_counter.items() if args.min_freq_zh > 0 and c < args.min_freq_zh}
        en_low = {t for t, c in en_counter.items() if args.min_freq_en > 0 and c < args.min_freq_en}

        # 2) apply low-freq policy
        after_lowfreq: List[Tuple[List[str], List[str], Optional[int]]] = []
        dropped_low = 0
        for zh_toks, en_toks, idx in tqdm(tokenized, desc="LowFreq"):
            zh2 = apply_low_freq_filter(zh_toks, zh_low, args.low_freq_policy)
            en2 = apply_low_freq_filter(en_toks, en_low, args.low_freq_policy)

            if zh2 is not None:
                zh2 = strip_special_tokens(zh2, keep_unk=True)
            if en2 is not None:
                en2 = strip_special_tokens(en2, keep_unk=True)

            if zh2 is None or en2 is None or (not zh2) or (not en2):
                dropped_low += 1
                continue
            after_lowfreq.append((zh2, en2, idx))
        print(f"[LowFreq] kept {len(after_lowfreq)} / dropped {dropped_low}")

    # 3) token length policy
    final_pairs: List[Tuple[List[str], List[str], Optional[int]]] = []
    zh_token_seqs: List[List[str]] = []
    en_token_seqs: List[List[str]] = []
    dropped_len = 0

    for zh_toks, en_toks, idx in tqdm(after_lowfreq, desc="TokLen"):
        kept = apply_length_policy(zh_toks, en_toks, args.max_len_zh, args.max_len_en, args.len_policy)
        if kept is None:
            dropped_len += 1
            continue
        zh3, en3 = kept
        final_pairs.append((zh3, en3, idx))
        zh_token_seqs.append(zh3)
        en_token_seqs.append(en3)
    print(f"[TokLen] kept {len(final_pairs)} / dropped {dropped_len}")

    # 4) build vocab (TRAIN ONLY)
    if is_spm:
        # vocab must be EXACTLY SPM pieces (id-aligned)
        zh_vocab = build_vocab_from_spm_model(args.spm_zh_model)
        en_vocab = build_vocab_from_spm_model(args.spm_en_model)

        # freq 只用于写 vocab_*.txt 的第二列（可选），不要用它裁剪 vocab
        zh_freq = freq_from_token_seqs(zh_token_seqs)
        en_freq = freq_from_token_seqs(en_token_seqs)  
    else:
        zh_vocab, zh_freq = build_vocab(zh_token_seqs, min_freq=args.min_freq_zh, max_vocab=args.max_vocab_zh)
        en_vocab, en_freq = build_vocab(en_token_seqs, min_freq=args.min_freq_en, max_vocab=args.max_vocab_en)
    print(f"[Vocab] zh={len(zh_vocab.itos)} en={len(en_vocab.itos)}")

    vocab_zh_path = os.path.join(args.output_dir, "vocab_zh.txt")
    vocab_en_path = os.path.join(args.output_dir, "vocab_en.txt")
    save_vocab_with_freq(vocab_zh_path, zh_vocab, zh_freq)
    save_vocab_with_freq(vocab_en_path, en_vocab, en_freq)

    # 5) convert to ids + save processed
    out_rows = []
    for zh_toks, en_toks, idx in final_pairs:
        out_rows.append({
            "index": idx,
            "zh_tokens": zh_toks,
            "en_tokens": en_toks,
            "zh_ids": tokens_to_ids(zh_toks, zh_vocab, add_bos_eos=True),
            "en_ids": tokens_to_ids(en_toks, en_vocab, add_bos_eos=True),
        })

    out_path = os.path.join(args.output_dir, args.out_name)
    write_jsonl(out_path, out_rows)
    print(f"[Save] processed -> {out_path}")

    # 6) optional embedding init (still TRAIN vocab)
    if args.init_en_vec:
        emb_en = build_embedding_matrix(en_vocab, args.emb_dim, args.init_en_vec)
        np.save(os.path.join(args.output_dir, "emb_en.npy"), emb_en)
        print("[Save] emb_en.npy")
    if args.init_zh_vec:
        emb_zh = build_embedding_matrix(zh_vocab, args.emb_dim, args.init_zh_vec)
        np.save(os.path.join(args.output_dir, "emb_zh.npy"), emb_zh)
        print("[Save] emb_zh.npy")

    # stats
    zh_lens = [len(r["zh_ids"]) for r in out_rows]
    en_lens = [len(r["en_ids"]) for r in out_rows]
    print(f"[Stats] zh_len mean={np.mean(zh_lens):.2f} p95={np.percentile(zh_lens,95):.0f}")
    print(f"[Stats] en_len mean={np.mean(en_lens):.2f} p95={np.percentile(en_lens,95):.0f}")


def _transform_and_save(tokenized: List[Tuple[List[str], List[str], Optional[int]]], args) -> None:
    """
    DEV/TEST:
    - load vocab from TRAIN
    - apply OOV policy based on vocab membership ONLY
    - apply length policy
    - convert to ids and save
    """
    is_spm = (args.tokenization == "spm_bpe")

    zh_vocab = load_vocab_from_txt(args.vocab_zh)
    en_vocab = load_vocab_from_txt(args.vocab_en)
    print(f"[VocabLoad] zh={len(zh_vocab.itos)} from {args.vocab_zh}")
    print(f"[VocabLoad] en={len(en_vocab.itos)} from {args.vocab_en}")

    if is_spm:
    # SPM: no OOV policy; tokenizer already emits <unk> when needed.
        after_oov = tokenized
        dropped_oov = 0
        print(f"[OOVPolicy][SPM] skipped (kept {len(after_oov)})")
    else:
        after_oov: List[Tuple[List[str], List[str], Optional[int]]] = []
        dropped_oov = 0 
        for zh_toks, en_toks, idx in tqdm(tokenized, desc="OOVPolicy"):
            zh2 = apply_oov_policy(zh_toks, zh_vocab, args.low_freq_policy)
            en2 = apply_oov_policy(en_toks, en_vocab, args.low_freq_policy)

            if zh2 is not None:
                zh2 = strip_special_tokens(zh2, keep_unk=True)
            if en2 is not None:
                en2 = strip_special_tokens(en2, keep_unk=True)

            if zh2 is None or en2 is None or (not zh2) or (not en2):
                dropped_oov += 1
                continue
            after_oov.append((zh2, en2, idx))
        print(f"[OOVPolicy] kept {len(after_oov)} / dropped {dropped_oov}")

    final_pairs: List[Tuple[List[str], List[str], Optional[int]]] = []
    dropped_len = 0
    for zh_toks, en_toks, idx in tqdm(after_oov, desc="TokLen"):
        kept = apply_length_policy(zh_toks, en_toks, args.max_len_zh, args.max_len_en, args.len_policy)
        if kept is None:
            dropped_len += 1
            continue
        zh3, en3 = kept
        final_pairs.append((zh3, en3, idx))
    print(f"[TokLen] kept {len(final_pairs)} / dropped {dropped_len}")

    out_rows = []
    for zh_toks, en_toks, idx in final_pairs:
        out_rows.append({
            "index": idx,
            "zh_tokens": zh_toks,
            "en_tokens": en_toks,
            "zh_ids": tokens_to_ids(zh_toks, zh_vocab, add_bos_eos=True),
            "en_ids": tokens_to_ids(en_toks, en_vocab, add_bos_eos=True),
        })

    out_path = os.path.join(args.output_dir, args.out_name)
    write_jsonl(out_path, out_rows)
    print(f"[Save] processed -> {out_path}")

    zh_lens = [len(r["zh_ids"]) for r in out_rows]
    en_lens = [len(r["en_ids"]) for r in out_rows]
    print(f"[Stats] zh_len mean={np.mean(zh_lens):.2f} p95={np.percentile(zh_lens,95):.0f}")
    print(f"[Stats] en_len mean={np.mean(en_lens):.2f} p95={np.percentile(en_lens,95):.0f}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", required=True, choices=["fit", "transform"],
                    help="fit: TRAIN ONLY (train spm/vocab). transform: DEV/TEST using TRAIN artifacts.")
    ap.add_argument("--input", required=True, help="Path to *.jsonl")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--out_name", default="processed.jsonl",
                    help="Output processed jsonl file name (e.g., processed_train.jsonl).")

    # Cleaning
    ap.add_argument("--lower_en", action="store_true", help="Lowercase English")

    # Char-level length filter (before tokenization)
    ap.add_argument("--max_char_len_zh", type=int, default=0,
                    help="Char-level max length for zh BEFORE tokenization; 0 disables.")
    ap.add_argument("--max_char_len_en", type=int, default=0,
                    help="Char-level max length for en BEFORE tokenization; 0 disables.")
    ap.add_argument("--char_len_policy", default="filter", choices=["filter", "truncate"],
                    help="Char-level policy applied BEFORE tokenization: filter or truncate.")

    # Tokenization strategy
    ap.add_argument("--tokenization", default="word", choices=["word", "spm_bpe", "wordpiece"],
                    help="Tokenization strategy: word | spm_bpe | wordpiece")

    # word tokenizers
    ap.add_argument("--zh_tok", default="jieba", choices=["jieba", "hanlp", "char"],
                    help="Chinese tokenizer in word mode: jieba | hanlp | char")
    ap.add_argument("--hanlp_model", default="",
                    help="Optional HanLP model id/identifier (used when --zh_tok hanlp).")
    ap.add_argument("--use_nltk", action="store_true",
                    help="Use NLTK tokenizer for English in word mode.")

    # wordpiece models
    ap.add_argument("--zh_wp_model", default="bert-base-chinese",
                    help="HF model name/path for Chinese WordPiece tokenizer.")
    ap.add_argument("--en_wp_model", default="bert-base-uncased",
                    help="HF model name/path for English WordPiece tokenizer.")

    # low-freq/OOV policy (unified)
    ap.add_argument("--min_freq_zh", type=int, default=2,
                    help="(fit only) Min freq for zh tokens (AFTER tokenization).")
    ap.add_argument("--min_freq_en", type=int, default=2,
                    help="(fit only) Min freq for en tokens (AFTER tokenization).")
    ap.add_argument("--low_freq_policy", default="replace_unk",
                    choices=["replace_unk", "remove_token", "drop_sample"],
                    help="fit: low-freq policy; transform: OOV policy based on TRAIN vocab.")

    # vocab
    ap.add_argument("--max_vocab_zh", type=int, default=50000, help="0 means no limit (fit only)")
    ap.add_argument("--max_vocab_en", type=int, default=50000, help="0 means no limit (fit only)")

    # token-level length (after tokenization)
    ap.add_argument("--max_len_zh", type=int, default=80,
                    help="Token-level max length for zh AFTER tokenization; 0 disables.")
    ap.add_argument("--max_len_en", type=int, default=80,
                    help="Token-level max length for en AFTER tokenization; 0 disables.")
    ap.add_argument("--len_policy", default="filter", choices=["filter", "truncate"],
                    help="Token-level policy applied AFTER tokenization: filter or truncate.")

    # sentencepiece (fit)
    ap.add_argument("--spm_vocab_size", type=int, default=16000)
    ap.add_argument("--spm_char_cov_zh", type=float, default=0.9995)
    ap.add_argument("--spm_char_cov_en", type=float, default=1.0)

    # artifacts for transform (from TRAIN)
    ap.add_argument("--spm_zh_model", default="",
                    help="(transform) Path to TRAIN spm_zh.model; default: output_dir/spm_zh.model")
    ap.add_argument("--spm_en_model", default="",
                    help="(transform) Path to TRAIN spm_en.model; default: output_dir/spm_en.model")
    ap.add_argument("--vocab_zh", default="",
                    help="(transform) Path to TRAIN vocab_zh.txt; default: output_dir/vocab_zh.txt")
    ap.add_argument("--vocab_en", default="",
                    help="(transform) Path to TRAIN vocab_en.txt; default: output_dir/vocab_en.txt")

    # embedding init (fit only)
    ap.add_argument("--init_en_vec", default="")
    ap.add_argument("--init_zh_vec", default="")
    ap.add_argument("--emb_dim", type=int, default=300)

    args = ap.parse_args()
    process_pipeline(args)


if __name__ == "__main__":
    main()
