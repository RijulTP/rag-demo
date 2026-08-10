"""Command-line chatbot that demonstrates the RAG flow.

Run with:  python chat.py            # AFTER RAG (with retrieval)
           python chat.py --no-rag   # BEFORE RAG (Gemini only, no retrieval)
           python chat.py --compare  # both side by side, for demos
Type "exit" to quit.
"""

import argparse

from rag import answer_question, answer_without_rag


def show_chunks(retrieved, limit: int | None = 120) -> None:
    """Print the retrieved chunks so the user can see what was passed to Gemini."""
    print("Retrieved context:")
    for i, (source, text) in enumerate(retrieved, start=1):
        preview = text[:limit].strip()
        if len(text) > limit:
            preview += " ..."
        print(f"[{i}] {source}")
        print(f"    {preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG demo chatbot")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="BEFORE RAG: ask Gemini directly with no retrieved context.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show BEFORE RAG and AFTER RAG answers side by side.",
    )
    args = parser.parse_args()

    mode = "BEFORE RAG (no context)" if args.no_rag else "AFTER RAG (with context)"
    if args.compare:
        mode = "COMPARE (before vs after RAG)"
    elif args.no_rag and args.compare:
        mode = "BEFORE RAG only"

    print("RAG Demo")
    print("--------")
    print(f"Mode: {mode}")
    print('Ask a question about the company knowledge base.')
    print('Type "exit" to quit.\n')

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        if args.compare:
            before = answer_without_rag(question)
            after, retrieved = answer_question(question)
            print(f"\n--- BEFORE RAG (no context) ---\n{before}")
            print("\n--- AFTER RAG ---")
            show_chunks(retrieved)
            print(f"--- ANSWER ---\n{after}\n")
        else:
            if args.no_rag:
                answer = answer_without_rag(question)
                print(f"\nAnswer:\n{answer}\n")
            else:
                answer, retrieved = answer_question(question)
                show_chunks(retrieved)
                print(f"Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
