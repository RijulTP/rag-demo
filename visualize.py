"""Interactive 3D visualization of the embeddings stored in ChromaDB.

Every chunk in the vector store is a 3072-dimensional vector. Humans cannot
see 3072 dimensions, so this script compresses each vector down to 3
dimensions with PCA and draws one point per chunk in an interactive Plotly
3D scatter plot. Points are colored by their source document, which makes
clusters of related chunks visible at a glance.

Run with:  python visualize.py
"""

from pathlib import Path

import chromadb
import plotly.graph_objects as go
from sklearn.decomposition import PCA

COLLECTION_NAME = "rag_demo"
DB_PATH = Path(__file__).parent / "chroma_db"
TARGET_AXES = 3


def open_store() -> chromadb.Collection:
    """Return a handle to the local vector collection."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_collection(COLLECTION_NAME)


def collect_entries(store: chromadb.Collection) -> tuple[list, list, list, list]:
    """Pull every stored record out of the collection.

    Returns (vectors, ids, source_tags, chunk_texts) in lock-step order.
    """
    records = store.get(include=["embeddings", "documents", "metadatas"])
    vectors = records["embeddings"]
    ids = records["ids"]
    source_tags = [
        (meta or {}).get("source", "unknown") for meta in records["metadatas"]
    ]
    chunk_texts = records["documents"] or []
    return vectors, ids, source_tags, chunk_texts


def compress_to_3d(vectors: list) -> tuple[object, object]:
    """Fit a PCA model on the vectors and return (model, projected_points)."""
    model = PCA(n_components=TARGET_AXES)
    projected_points = model.fit_transform(vectors)
    return model, projected_points


def build_hover_labels(ids, source_tags, chunk_texts) -> list[str]:
    """Compose the rich tooltip shown when hovering over a point."""
    return [
        f"id: {rid}<br>source: {src}<br>chunk: {text[:120]!r}"
        for rid, src, text in zip(ids, source_tags, chunk_texts)
    ]


def render_scene(
    projected_points,
    source_tags,
    hover_labels,
    variance_explained: float,
) -> None:
    """Build the interactive 3D figure and open it in the browser."""
    figure = go.Figure()
    figure.update_layout(
        title=(
            f"{COLLECTION_NAME} embeddings projected into 3D "
            f"(explained variance {variance_explained:.0%})"
        ),
        scene={
            "xaxis_title": "PCA Axis 1",
            "yaxis_title": "PCA Axis 2",
            "zaxis_title": "PCA Axis 3",
        },
    )

    # One trace per source document so each cluster can be toggled in the legend.
    for source in sorted(set(source_tags)):
        mask = [tag == source for tag in source_tags]
        figure.add_trace(
            go.Scatter3d(
                x=projected_points[mask, 0],
                y=projected_points[mask, 1],
                z=projected_points[mask, 2],
                mode="markers",
                name=source,
                customdata=[hover_labels[i] for i in range(len(mask)) if mask[i]],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    figure.show()


def main() -> None:
    store = open_store()

    vectors, ids, source_tags, chunk_texts = collect_entries(store)
    if len(vectors) == 0:
        print("No embeddings found. Run `python ingest.py` first.")
        return

    model, projected_points = compress_to_3d(vectors)
    hover_labels = build_hover_labels(ids, source_tags, chunk_texts)
    variance_explained = model.explained_variance_ratio_.sum()

    render_scene(projected_points, source_tags, hover_labels, variance_explained)
    print(
        f"Plotted {len(vectors)} chunks in 3D "
        f"({variance_explained:.0%} of variance preserved)."
    )


if __name__ == "__main__":
    main()