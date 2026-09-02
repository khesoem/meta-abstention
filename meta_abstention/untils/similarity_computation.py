"""
Pairwise code similarity: CodeBLEU (symmetrized), CodeBERT cosine, UniXcoder-base cosine.

Install:
    pip install codebleu tree-sitter-python transformers torch

For CPU-only, the default `pip install torch` works, but the CPU-only wheel is
much smaller:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import code_bert_score
from code_bert_score import BERTScorer


# # ---------- CodeBLEU (symmetrized) ----------

# def codebleu_sim(code1: str, code2: str, lang: str = "python") -> float:
#     """Mean of both directions since CodeBLEU is asymmetric (ref vs. pred)."""
#     a = calc_codebleu([code1], [code2], lang=lang)["codebleu"]
#     b = calc_codebleu([code2], [code1], lang=lang)["codebleu"]
#     return (a + b) / 2


@lru_cache(maxsize=None)
def _bertscorer(lang: str = "python"):
    return BERTScorer(lang=lang)  # loads model + tokenizer once

def codebertscore_sim(code1: str, code2: str, lang: str = "python") -> float:
    _, _, f1 = _bertscorer(lang).score(cands=[code1], refs=[code2])
    return f1.item()

# ---------- CodeBERT cosine ----------

@lru_cache(maxsize=1)
def _codebert():
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").eval()
    return tok, model


@torch.no_grad()
def codebert_cosine_sim(code1: str, code2: str) -> float:
    """Cosine similarity of mean-pooled CodeBERT embeddings."""
    tok, model = _codebert()
    enc = tok([code1, code2], return_tensors="pt",
              padding=True, truncation=True, max_length=512)
    hidden = model(**enc).last_hidden_state                # (2, L, H)
    mask = enc["attention_mask"].unsqueeze(-1)             # (2, L, 1)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    v = F.normalize(pooled, dim=-1)
    return (v[0] @ v[1]).item()


# ---------- UniXcoder-base cosine ----------

@lru_cache(maxsize=1)
def _unixcoder():
    tok = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base").eval()
    return tok, model


@torch.no_grad()
def unixcoder_sim(code1: str, code2: str) -> float:
    """Cosine similarity of mean-pooled UniXcoder-base embeddings."""
    tok, model = _unixcoder()
    enc = tok([code1, code2], return_tensors="pt",
              padding=True, truncation=True, max_length=512)
    hidden = model(**enc).last_hidden_state                # (2, L, H)
    mask = enc["attention_mask"].unsqueeze(-1)             # (2, L, 1)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    v = F.normalize(pooled, dim=-1)
    return (v[0] @ v[1]).item()


# ---------- demo ----------

if __name__ == "__main__":
    a = "def add(x, y):\n    return x + y\n"
    b = "def sum_two(first, second):\n    return second + first\n"
    # print(f"CodeBLEU:         {codebleu_sim(a, b):.4f}")
    print(f"CodeBERT cosine:  {codebert_sim(a, b):.4f}")
    print(f"UniXcoder cosine: {unixcoder_sim(a, b):.4f}")