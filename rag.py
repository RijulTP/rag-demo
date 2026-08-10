"""RAG pipeline: embed a question, retrieve context, generate an answer.

answer_question(question) is the only public function you need.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

COLLECTION_NAME = "rag_demo"
DB_PATH = Path(__file__).parent / "chroma_db"
TOP_K = 3  # number of chunks to retrieve and pass to the model


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=api_key)


def retrieve_context(client: genai.Client, question: str) -> list[tuple[str, str]]:
    """Embed the question and return the TOP_K most similar chunks as (source, text)."""
    collection = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION_NAME)

    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    response = client.models.embed_content(model=model, contents=question)
    question_embedding = response.embeddings[0].values

    results = collection.query(query_embeddings=[question_embedding], n_results=TOP_K)
    sources = [m["source"] for m in results["metadatas"][0]]
    texts = results["documents"][0]
    return list(zip(sources, texts))


def answer_question(question: str) -> tuple[str, list[tuple[str, str]]]:
    """Answer `question` from the knowledge base. Returns (answer, retrieved_chunks)."""
    client = get_client()

    # 1. Retrieve the most relevant chunks.
    retrieved = retrieve_context(client, question)

    # 2. Build a prompt with the question and the retrieved context.
    context_block = "\n\n".join(
        f"Source: {source}\n{text}" for source, text in retrieved
    )
    prompt = (
        "You are a helpful assistant for a company's knowledge base.\n\n"
        "Context:\n"
        "--------------------\n"
        f"{context_block}\n"
        "--------------------\n\n"
        f"Question: {question}\n\n"
        "Answer the question using ONLY the context above. "
        "Do not use outside knowledge and do not invent facts. "
        "If the answer is not present in the context, say clearly that "
        "the information is not available in the knowledge base."
    )

    # 3. Generate the answer with the retrieved context.
    model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Answer only using the provided context. Do not use outside "
                "knowledge. If the answer is not in the context, say that the "
                "information is not available in the knowledge base. Do not "
                "invent facts."
            )
        ),
    )

    return response.text, retrieved


def answer_without_rag(question: str) -> str:
    """Ask Gemini directly with NO retrieval. Answers from training memory only.

    This is the "before RAG" baseline: no context is provided, so the model
    can guess or make things up instead of answering from the knowledge base.
    """
    client = get_client()
    model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(model=model, contents=question)
    return response.text


if __name__ == "__main__":
    print(answer_question("What is the refund policy?"))
