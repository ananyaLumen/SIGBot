from utils.text_cleaner import clean_text
from utils.sentence_splitter import split_into_sentences
from ingestion.chunker import chunk_sentences

def chunk_page_text(pages, max_words: int = 200):
    """
    chunk each page separately and attach page metadata.
    """
    structured_chunks = []
    chunk_id = 0

    for page in pages:
        page_number = page["page_number"]
        raw_text = page["text"]

        cleaned_text = clean_text(raw_text)
        sentences = split_into_sentences(cleaned_text)

        page_chunks = chunk_sentences(
            sentences,
            max_words = max_words
        )

        for chunk_text in page_chunks:
            structured_chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk_text
            })
            chunk_id += 1

    return structured_chunks