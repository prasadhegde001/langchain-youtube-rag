# YouTube Transcript Analyzer

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a YouTube video using its transcript.

## Architecture

```
                        YouTube Video (video_id)
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  get_youtube_transcript()      │  1. INGESTION
                 │  youtube-transcript-api        │
                 │  → List[Document]              │
                 │    (text + start/duration/     │
                 │     source metadata)           │
                 └───────────────┬────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  chunking_documents()          │  2. CHUNKING
                 │  RecursiveCharacterTextSplitter│
                 │  merge → split (500/100 overlap)│
                 │  → List[Document] chunks       │
                 └───────────────┬────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  embedding_vector_store()      │  3. EMBEDDING + INDEX
                 │  OpenAIEmbeddings              │
                 │  (text-embedding-3-small)      │
                 │  → FAISS vectorstore           │
                 └───────────────┬────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  retrival_documents()          │  4. RETRIEVAL
                 │  vectorstore.as_retriever()    │
                 │  MMR search (k=4, fetch_k=20)  │
                 │  → retriever                   │
                 └───────────────┬────────────────┘
                                 ▼
        User question ──────────┤
                                 ▼
                 ┌───────────────────────────────┐
                 │  build_rag_chain()             │  5. GENERATION
                 │  retriever → format_docs()     │
                 │       ↓ context                │
                 │  RunnableParallel(context, Q)  │
                 │       ↓                        │
                 │  ChatPromptTemplate            │
                 │       ↓                        │
                 │  ChatOpenAI (gpt-5-nano)       │
                 │       ↓                        │
                 │  StrOutputParser()             │
                 └───────────────┬────────────────┘
                                 ▼
                    Answer (with cited timestamps)
```

**Pipeline stages** (each maps to a function in `main.py`):

| Stage | Function | Responsibility |
|---|---|---|
| Ingestion | `get_youtube_transcript(video_id)` | Pull raw transcript, wrap each segment as a `Document` with a source timestamp link |
| Chunking | `chunking_documents(docs, video_id)` | Merge transcript text and split into overlapping chunks for retrieval |
| Embedding & Indexing | `embedding_vector_store(chunks)` | Embed chunks and build an in-memory FAISS index |
| Retrieval | `retrival_documents(vectorstore)` | Configure an MMR retriever for diverse, relevant chunk selection |
| Context formatting | `format_docs(docs)` | Flatten retrieved docs into a single citation-annotated context string |
| Generation | `build_rag_chain(retriever)` | Wire retriever + prompt + LLM + output parser into one runnable RAG chain |

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
