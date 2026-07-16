"""
Text preprocessing for Bengali sentiment analysis.

BanglaBERT (csebuetnlp/banglabert) was pretrained on text that went through a
specific normalization pipeline (Unicode NFKC normalization + punctuation /
whitespace normalization via the `normalizer` package from the same authors).
Skipping this step measurably hurts downstream performance, so we try to use
it and fall back to a reasonable manual normalization if the package isn't
available (e.g. offline sessions without internet enabled).

Install the official normalizer with:
    pip install git+https://github.com/csebuetnlp/normalizer
"""

import re
import unicodedata

try:
    from normalizer import normalize as _bnlp_normalize  # type: ignore

    _HAS_BNLP_NORMALIZER = True
except ImportError:
    _HAS_BNLP_NORMALIZER = False


_URL_RE = re.compile(r"http\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_PUNC_RE = re.compile(r"([।!?.,])\1{1,}")  # collapse repeated punctuation


def basic_clean(text: str) -> str:
    """Fallback cleaning used regardless of whether the official normalizer
    is available: Unicode normalization, URL stripping, whitespace collapse.
    """
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _MULTI_PUNC_RE.sub(r"\1", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """Full normalization pipeline: basic cleaning, then the official
    BanglaBERT normalizer if installed.
    """
    text = basic_clean(text)
    if _HAS_BNLP_NORMALIZER:
        try:
            text = _bnlp_normalize(text)
        except Exception:
            # Never let a normalization edge case crash training.
            pass
    return text


def normalizer_available() -> bool:
    return _HAS_BNLP_NORMALIZER
