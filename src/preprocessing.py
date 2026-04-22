import re


def clean_text(text: str) -> str:
    """Apply lightweight inference-time text normalization.

    Keep this function aligned with the preprocessing used during training.
    The current version only strips the text and collapses repeated whitespace,
    because aggressive cleaning can shift the inference distribution.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
