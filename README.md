# RAG Demo with Google Gemini

A tiny, educational Retrieval-Augmented Generation (RAG) demo. It answers
questions about a fictional company using a local knowledge base of Markdown
files, a local vector store (ChromaDB), and the Gemini API for both embeddings
and answer generation.

No frameworks, no agents, no web servers. Just a few readable Python files.

## Quick start (tl;dr)

```bash
git clone <your-repo-url> && cd rag-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Gemini API key into .env
python ingest.py            # build the vector store from documents/
python chat.py              # start the chatbot
```

See the sections below for details on the API key, ingestion, and usage.

## 1. What is RAG?

**Retrieval-Augmented Generation** means giving a large language model (LLM)
relevant documents *before* it answers, so it can answer from those documents
instead of relying only on what it memorized during training.

Two steps happen:

1. **Retrieval** — find the documents most relevant to the user's question.
2. **Generation** — let the LLM write an answer grounded in those documents.

## 2. Why is retrieval needed?

An LLM only "knows" what it saw during training. It cannot answer about your
company's private pricing, policies, or products — and if you ask anyway, it
may confidently **make things up** (hallucinate). Retrieval fixes both problems:

- **Private data**: it pulls in your documents, which the model has never seen.
- **Fewer hallucinations**: the model is told to answer only from the retrieved
  context, so it stops inventing facts.

## 3. What this demo does

It loads Markdown files from `documents/`, splits them into chunks, embeds each
chunk, and stores the vectors in a local ChromaDB database (`ingest.py`).
When you ask a question, it embeds the question, finds the 3 most similar
chunks, and passes them to Gemini along with the question (`rag.py`). The
answer is generated from that context only (`chat.py` wraps it in a CLI).

## 4. The RAG pipeline

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector database
   ↓
User question
   ↓
Question embedding
   ↓
Similarity search
   ↓
Relevant chunks
   ↓
Gemini + retrieved context
   ↓
Answer
```

## 5. Get and configure a Gemini API key

1. Go to <https://aistudio.google.com/app/apikey> and sign in with a Google
   account.
2. Click **Create API key** and copy it.
3. Copy the example environment file and paste your key in:

   ```bash
   cp .env.example .env
   # open .env and set GEMINI_API_KEY=your_key_here
   ```

Model names live in `.env` and are read at runtime:

```text
GEMINI_API_KEY=your_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_GENERATION_MODEL=gemini-2.5-flash
```

Never commit your real `.env`.

## 6. Install dependencies

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## 7. Ingest the documents

```bash
python ingest.py
```

You should see output like:

```text
Loaded 4 documents
Created 12 chunks
Stored 12 embeddings
```

The script is idempotent: running it again only adds chunks for source files
that are not already in the database, so it won't duplicate data.

## 8. Run the chatbot

```bash
python chat.py            # AFTER RAG (retrieval + context)
python chat.py --no-rag   # BEFORE RAG (Gemini only, no context) - for the demo
python chat.py --compare  # BOTH side by side - best for the live demo
```

The chatbot reads the retrieved chunks and the final answer from the same
pipeline you saw in `rag.py`. In RAG mode the retrieved chunks are printed
above the answer so you can see exactly what was passed to Gemini.

Example session:

```text
RAG Demo
--------
Ask a question about the company knowledge base.
Type "exit" to quit.

Question: What is the refund policy?

Retrieved context:
[1] policies.md
    # Policies
    ## Refund policy
    We offer a **30-day money-back guarantee**...
[2] policies.md
    ...returned to the original payment method...
[3] pricing.md
    The Pro plan costs $29 per user per month...

Answer:
We offer a 30-day money-back guarantee... on your paid subscription, request a
full refund by emailing support@acmewidgets.example...
```

## 9. Example questions

Try these:

- What is the refund policy? *(answer is in `policies.md`)*
- How much does the Pro plan cost? *(answer is in `pricing.md`)*
- When was the company founded? *(answer is in `company.md`)*
- What are the key features of WidgetBoard? *(answer is in `products.md`)*
- Who is the CEO? *(NOT in the knowledge base — the model should say it can't answer)*

Notice the last question: the retrieved context won't contain the answer, and
the model is instructed to say the information is not available instead of
guessing. This shows the difference between retrieval and generation.

## 10. How the retrieved context is passed to Gemini

In `rag.py`, `answer_question` does the following:

1. Embeds the question with the Gemini embeddings model.
2. Queries ChromaDB for the top 3 most similar chunks.
3. Builds a text prompt that labels each chunk with its source file and adds
   the question:

   ```text
   Context:
   --------------------
   Source: policies.md
   We offer a 30-day money-back guarantee...
   --------------------
   Question: What is the refund policy?
   ```

4. Sends that prompt to the Gemini generation model with a
   `system_instruction` that says: answer only from the context, don't use
   outside knowledge, and say "not available" if the answer isn't there.

So the model sees only your documents plus the question — that's what keeps the
answer grounded in the knowledge base.

## Project structure

```text
rag-demo/
├── documents/        # the knowledge base (Markdown)
│   ├── company.md
│   ├── products.md
│   ├── pricing.md
│   └── policies.md
├── ingest.py         # build chunks + embeddings, store in ChromaDB
├── rag.py            # answer_question(): retrieve + generate
├── chat.py           # command-line interface
├── requirements.txt
├── .env.example
└── README.md
```

