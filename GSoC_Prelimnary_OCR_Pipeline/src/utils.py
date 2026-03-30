import re
import math
import unicodedata
from collections import Counter

def normalize(text: str, keep_newlines: bool = False, lower: bool = False) -> str:
    """Normalize OCR text, removing hyphenated breaks and standardizing spacing."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Fix hyphenated line breaks first
    text = re.sub(r"-\s*\n\s*", "", text)

    # Remove control chars that mess up comparison
    text = re.sub(r"[\u200b\u200c\u200d\u2060\u00ad]", "", text)

    if keep_newlines:
        text = text.replace("\r", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text).strip()
    else:
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()

    if lower:
        text = text.lower()

    return text

def edit_distance(a, b):
    """Compute the Levenshtein distance between two sequences (strings or lists)."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i]
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost
            ))
        prev = curr
    return prev[-1]

def cer(ref: str, hyp: str) -> float:
    """Calculate Character Error Rate."""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)

def wer(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)

def bleu4(ref: str, hyp: str) -> float:
    """Calculate BLEU-4 score."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        if len(hyp_tokens) < n:
            precisions.append(1e-9)
            continue

        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))

        overlap = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items())
        total = max(len(hyp_tokens) - n + 1, 1)

        # smoothing
        precisions.append((overlap + 1) / (total + 1))

    log_prec = sum(math.log(max(p, 1e-12)) for p in precisions) / 4

    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))

    return bp * math.exp(log_prec)
