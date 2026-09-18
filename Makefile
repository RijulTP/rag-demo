.PHONY: help install ingest chat chat-no-rag chat-compare chat-defended \
       demo demo-live demo-attack1 demo-attack2 \
       visualize probe drift clean clean-poison

PYTHON := $(shell test -f .venv/bin/python && echo .venv/bin/python || echo python3)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────

install: ## Install dependencies
	$(PYTHON) -m pip install -r requirements.txt

# ── Ingestion ────────────────────────────────────────────────────────────────

ingest: ## Ingest documents/ into ChromaDB
	$(PYTHON) ingest.py

# ── Chat ─────────────────────────────────────────────────────────────────────

chat: ## Start chatbot (RAG mode)
	$(PYTHON) chat.py

chat-no-rag: ## Start chatbot (no RAG — Gemini only)
	$(PYTHON) chat.py --no-rag

chat-compare: ## Start chatbot (side-by-side: before vs after RAG)
	$(PYTHON) chat.py --compare

chat-defended: ## Start chatbot (RAG + defense layer)
	$(PYTHON) chat.py --defended

# ── Demo ─────────────────────────────────────────────────────────────────────

demo: ## Article demo (offline, no API key needed)
	$(PYTHON) demo_runner.py article

demo-live: ## Full live demo (all 4 attacks + defense, needs API key)
	$(PYTHON) demo_runner.py live

demo-attack1: ## Live demo — Attack 1 only (fact injection + defense)
	$(PYTHON) demo_runner.py attack1

demo-attack2: ## Live demo — Attack 2 only (prompt injection + defense)
	$(PYTHON) demo_runner.py attack2

# ── Attacks (standalone) ────────────────────────────────────────────────────

probe: ## Run embedding similarity probe
	$(PYTHON) attacks/embedding_probe.py --query "how do I cancel my subscription"

drift: ## Run drift simulation (5 versions + PCA plot)
	$(PYTHON) attacks/drift_simulation.py

# ── Visualization ────────────────────────────────────────────────────────────

visualize: ## Open interactive 3D scatter plot of embeddings
	$(PYTHON) visualize.py

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove ChromaDB (rebuild with make ingest)
	rm -rf chroma_db

clean-poison: ## Remove only poison chunks from ChromaDB
	@$(PYTHON) -c "import chromadb; c = chromadb.PersistentClient(path='chroma_db'); col = c.get_or_create_collection(name='rag_demo'); [col.delete(ids=col.get(where={'source': s})['ids']) for s in ['poison_pricing.md', 'poison_instructions.md', 'poison_probe.md'] if col.get(where={'source': s})['ids']]; print(f'Clean. {col.count()} chunks remain.')"
