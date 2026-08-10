"""Ingestion pipeline: markdown files -> chunks -> embeddings -> ChromaDB.

Run with:  python ingest.py
Re-running only indexes documents that are not already in the store.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 60        # characters shared between neighbouring chunks
COLLECTION_NAME = "rag_demo"
DB_PATH = Path(__file__).parent / "chroma_db"


def get_client() -> genai.Client:
    """Create the Gemini client using the API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=api_key)


def read_markdown_files(directory: Path) -> dict[str, str]:
    """Return {filename: full_text} for every .md file in the directory."""
    docs = {}
    for path in sorted(directory.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of roughly `size` characters."""
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a list of strings with Gemini and return a list of vectors."""
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    response = client.models.embed_content(model=model, contents=texts)
    return [e.values for e in response.embeddings]


def main() -> None:
    docs = read_markdown_files(DOCUMENTS_DIR)

    # Build (chunk, metadata) pairs, remembering which file each chunk came from.
    chunks: list[str] = []
    metadatas: list[dict] = []
    for filename, text in docs.items():
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(chunk)
            metadatas.append({"source": filename})

    # Reuse an existing ChromaDB store and skip files already indexed.
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing = collection.get()
    # Set of source filenames already indexed in the collection.
    existing_sources = {m.get("source") for m in (existing["metadatas"] or [])}
    # Filter to chunks whose source file has not been indexed yet.
    fresh_chunks = [
        (text, meta)
        for text, meta in zip(chunks, metadatas)
        if meta["source"] not in existing_sources
    ]

    if fresh_chunks:
        fresh_texts = [text for text, _ in fresh_chunks]
        fresh_metas = [meta for _, meta in fresh_chunks]
        embeddings = embed_texts(get_client(), fresh_texts)
        ids = [f"chunk_{idx}" for idx in range(len(fresh_texts))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=fresh_texts,
            metadatas=fresh_metas,
        )
        print(f"Added {len(fresh_texts)} new chunks.")

    total = collection.count()
    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Stored {total} embeddings")


if __name__ == "__main__":
    main()
