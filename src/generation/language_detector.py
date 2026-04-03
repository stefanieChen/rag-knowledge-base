"""Automatic language detection for user queries.

Uses the langdetect library to identify query language, enabling
automatic prompt template selection and language-aware processing.
"""

from typing import Optional

from src.logging.logger import get_logger

logger = get_logger("generation.language_detector")

_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect, detect_langs, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    logger.info("langdetect not available — install langdetect for auto language detection")


# Map langdetect codes to our internal language names
_LANG_MAP = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "ru": "ru",
}

# Map detected language to recommended prompt template
_TEMPLATE_MAP = {
    "zh": "chinese_v1",
    "en": "default_v1",
}


def detect_language(text: str) -> Optional[str]:
    """Detect the language of a text string.

    Args:
        text: Input text to analyze.

    Returns:
        ISO language code (e.g., 'en', 'zh'), or None if detection fails.
    """
    if not _LANGDETECT_AVAILABLE:
        return None

    if not text or len(text.strip()) < 3:
        return None

    try:
        raw = detect(text)
        normalized = _LANG_MAP.get(raw, raw)
        logger.debug(f"Language detected: '{text[:30]}...' → {raw} → {normalized}")
        return normalized
    except LangDetectException:
        return None
    except Exception as e:
        logger.debug(f"Language detection failed: {e}")
        return None


def detect_language_with_confidence(text: str) -> dict:
    """Detect language with confidence scores.

    Args:
        text: Input text to analyze.

    Returns:
        Dict with 'language' (str or None), 'confidence' (float),
        and 'all_results' (list of {lang, prob} dicts).
    """
    if not _LANGDETECT_AVAILABLE:
        return {"language": None, "confidence": 0.0, "all_results": []}

    if not text or len(text.strip()) < 3:
        return {"language": None, "confidence": 0.0, "all_results": []}

    try:
        results = detect_langs(text)
        all_results = [
            {"lang": _LANG_MAP.get(str(r.lang), str(r.lang)), "prob": round(r.prob, 4)}
            for r in results
        ]
        top = all_results[0] if all_results else {"lang": None, "prob": 0.0}
        return {
            "language": top["lang"],
            "confidence": top["prob"],
            "all_results": all_results,
        }
    except Exception:
        return {"language": None, "confidence": 0.0, "all_results": []}


def suggest_template(text: str, default: str = "default_v1") -> str:
    """Suggest a prompt template based on detected language.

    Args:
        text: User query text.
        default: Default template to return if detection fails.

    Returns:
        Recommended template name string.
    """
    lang = detect_language(text)
    if lang and lang in _TEMPLATE_MAP:
        suggested = _TEMPLATE_MAP[lang]
        logger.debug(f"Template suggestion: lang={lang} → {suggested}")
        return suggested
    return default


def is_available() -> bool:
    """Check whether language detection is available.

    Returns:
        True if langdetect is installed.
    """
    return _LANGDETECT_AVAILABLE
