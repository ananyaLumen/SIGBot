from ingestion.pdf_extraction import extract_pdf_text
from ingestion.page_chunker import chunk_page_text
import json
from pathlib import Path

PDF_PATH = "data/raw_docs/sample_soc2.pdf"
OUTPUT_PATH = Path("data/chunks/page_chunks.json")

if __name__ == "__main__":
    pages = extract_pdf_text(PDF_PATH)
    chunks = chunk_page_text(pages, max_words=200)

    Path("data/chunks").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH}")