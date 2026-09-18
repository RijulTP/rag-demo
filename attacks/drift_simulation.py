"""Attack 4: Gradual Drift / Cumulative Poisoning.

Simulates a wiki-style knowledge base where small edits accumulate over time,
gradually shifting the facts. Ingests 5 versions of company.md sequentially,
queries after each, and produces a PCA scatter plot showing embedding drift.

Usage:
    python attacks/drift_simulation.py
    python attacks/drift_simulation.py --query "How many employees does Acme Widgets have?"
    python attacks/drift_simulation.py --versions-dir attacks/drift_versions
"""

import argparse
import shutil
import sys
from pathlib import Path

import chromadb
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from sklearn.decomposition import PCA

# Add parent dir so we can import from rag.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "rag_demo"
DEFAULT_VERSIONS_DIR = Path(__file__).resolve().parent / "drift_versions"
COMPANY_SOURCE = "company.md"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def delete_company_chunks() -> None:
    """Delete only chunks with source == company.md (metadata filter)."""
    collection = get_collection()
    existing = collection.get(where={"source": COMPANY_SOURCE})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Deleted {len(existing['ids'])} old company.md chunks.")


def ingest_version(version_path: Path) -> int:
    """Ingest a single version of company.md into ChromaDB. Returns chunk count."""
    text = version_path.read_text(encoding="utf-8")

    # Chunk it (same logic as ingest.py)
    chunk_size = 400
    chunk_overlap = 60
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap

    # Embed
    client = rag.get_client()
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    response = client.models.embed_content(model=model, contents=chunks)
    embeddings = [e.values for e in response.embeddings]

    # Store
    collection = get_collection()
    ids = [f"drift_company_{i}" for i in range(len(chunks))]
    metadatas = [{"source": COMPANY_SOURCE, "version": version_path.stem}] * len(chunks)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def query_versions(queries: list[str]) -> list[dict]:
    """Query the current state of the collection. Returns list of result dicts."""
    client = rag.get_client()
    results = []
    for q in queries:
        answer, retrieved = rag.answer_question(q)
        results.append({"query": q, "answer": answer, "retrieved": retrieved})
    return results


def collect_all_chunks() -> tuple[list, list, list]:
    """Pull all company.md chunks for PCA. Returns (embeddings, texts, versions)."""
    collection = get_collection()
    records = collection.get(
        include=["embeddings", "documents", "metadatas"],
        where={"source": COMPANY_SOURCE},
    )
    embeddings = records["embeddings"]
    texts = records["documents"]
    versions = [
        (meta or {}).get("version", "unknown") for meta in records["metadatas"]
    ]
    return embeddings, texts, versions


def build_drift_plot(embeddings, versions, output_path: Path) -> None:
    """PCA project to 3D and save an interactive Plotly HTML plot."""
    vectors = np.array(embeddings)
    pca = PCA(n_components=3)
    projected = pca.fit_transform(vectors)
    variance = pca.explained_variance_ratio_.sum()

    figure = go.Figure()

    unique_versions = sorted(set(versions))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, version in enumerate(unique_versions):
        mask = [v == version for v in versions]
        figure.add_trace(go.Scatter3d(
            x=projected[mask, 0],
            y=projected[mask, 1],
            z=projected[mask, 2],
            mode="markers",
            name=version,
            marker=dict(size=6, color=colors[i % len(colors)]),
            text=[f"chunk {j}" for j, m in enumerate(mask) if m],
            hovertemplate="%{text}<br>version: " + version + "<extra></extra>",
        ))

    figure.update_layout(
        title=f"Drift Simulation — company.md embeddings over 5 versions<br>"
              f"(PCA variance explained: {variance:.0%})",
        scene={
            "xaxis_title": "PCA Axis 1",
            "yaxis_title": "PCA Axis 2",
            "zaxis_title": "PCA Axis 3",
        },
        legend_title="Version",
    )

    figure.write_html(str(output_path))
    print(f"\n  Drift plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attack 4: Gradual Drift Simulation")
    parser.add_argument(
        "--versions-dir", type=Path, default=DEFAULT_VERSIONS_DIR,
        help="Directory containing v1.md through v5.md"
    )
    parser.add_argument(
        "--query", default=None,
        help="Additional query to run after each version (can repeat)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "drift_plot.html",
        help="Output path for the PCA plot HTML"
    )
    args = parser.parse_args()

    queries = [
        "When was the company founded?",
        "How many employees does Acme Widgets have?",
    ]
    if args.query:
        queries.append(args.query)

    versions_dir = args.versions_dir
    if not versions_dir.exists():
        print(f"ERROR: versions directory not found: {versions_dir}")
        sys.exit(1)

    version_files = sorted(versions_dir.glob("v*.md"))
    if not version_files:
        print(f"ERROR: no v*.md files found in {versions_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Attack 4: Gradual Drift / Cumulative Poisoning")
    print(f"{'='*60}")
    print(f"Versions directory: {versions_dir}")
    print(f"Version files: {[f.name for f in version_files]}")
    print(f"Queries: {queries}\n")

    all_results = []

    for vfile in version_files:
        print(f"--- Ingesting {vfile.name} ---")
        delete_company_chunks()
        chunk_count = ingest_version(vfile)
        print(f"  Ingested {chunk_count} chunks from {vfile.name}")

        # Query
        results = query_versions(queries)
        all_results.append({"version": vfile.name, "results": results})

        for r in results:
            print(f"\n  Q: {r['query']}")
            answer_preview = r["answer"][:200].strip()
            print(f"  A: {answer_preview}")
            if r["retrieved"]:
                src, txt = r["retrieved"][0]
                print(f"  Top chunk: [{src}] {txt[:100].strip()}...")
        print()

    # Collect all company.md chunks for PCA plot
    print("--- Building drift visualization ---")
    embeddings, texts, versions = collect_all_chunks()
    if embeddings:
        build_drift_plot(embeddings, versions, args.output)
    else:
        print("  No chunks found for visualization.")

    # Summary
    print(f"\n{'='*60}")
    print("Drift Summary")
    print(f"{'='*60}")
    for entry in all_results:
        version = entry["version"]
        for r in entry["results"]:
            answer_preview = r["answer"][:100].strip()
            print(f"  {version}: {r['query'][:50]}")
            print(f"    -> {answer_preview}")

    print(f"\nPlot: {args.output.resolve()}")


if __name__ == "__main__":
    main()
