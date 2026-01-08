import re

def clean_text(text: str) -> str:
    """
    Basic text normalization: 
    - remove extra whitespace
    - remove non-printable characters
    """

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()