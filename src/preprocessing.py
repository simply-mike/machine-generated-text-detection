import html
import re


def clean_text(text: str) -> str:
    """Apply the same text normalization used in the training notebook."""
    text = html.unescape(str(text))
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
