# YouTube Transcript Analyzer

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a YouTube video using its transcript.

## How it works

1. **Ingestion** — Fetches the transcript of a YouTube video via `youtube-transcript-api`, converting each segment into a LangChain `Document` with timestamp/source metadata.
2. **Chunking** — Merges the transcript and splits it into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding & Indexing** — Embeds chunks with OpenAI's `text-embedding-3-small` model and stores them in a FAISS vector store.
4. **Retrieval** — Uses Maximal Marginal Relevance (MMR) search to fetch diverse, relevant chunks for a given query.
5. **Generation** — Passes the retrieved context and question to `gpt-5-nano` via a RAG chain to produce a cited answer.

## Requirements

- Python 3.9+
- An OpenAI-compatible API key

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here   # optional, for OpenAI-compatible providers
```

## Usage

Edit the `video_id` and `question` variables in `main.py`, then run:

```bash
python main.py
```

The script fetches the transcript, builds the vector index, retrieves relevant chunks, and prints the generated answer with source timestamp links.

## Project structure

```
main.py             # Full RAG pipeline (ingestion, chunking, embedding, retrieval, generation)
requirements.txt    # Python dependencies
.env                # API keys (not committed)
```
