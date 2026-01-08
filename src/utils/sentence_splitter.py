from nltk.tokenize import sent_tokenize

def split_into_sentences(text: str):
    """
    split cleaned text into sentences.
    """
    return sent_tokenize(text)