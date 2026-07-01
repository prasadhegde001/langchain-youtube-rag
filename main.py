from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Extract Transcript from Youtube (Document Ingestion)
def get_youtube_transcript(video_id):
    """
    Fetches transcript from a YouTube video and returns LangChain Documents.
    Each transcript segment becomes a Document with metadata.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        documents = []

        for entry in transcript:
            doc = Document(
                page_content = entry.text,
                metadata = {
                    "start": entry.start,
                    "duration": entry.duration,
                    "video_id": video_id,
                    "source": f"https://www.youtube.com/watch?v={video_id}&t={int(entry.start)}s"
                }
            )
            documents.append(doc)
        
        return documents

        
    except TranscriptsDisabled:
        print("No captions available for this video.")


# Chunking - Recursive Character Text Splitter
def chunking_documents(docs, video_id):

    # First, merge all transcript segments into one document
    full_transcript = " ".join([doc.page_content for doc in docs])

    # Then split with optimal settings for transcripts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    chunks = text_splitter.create_documents(
    texts=[full_transcript],
    metadatas=[{"video_id": video_id, "source": f"https://youtube.com/watch?v={video_id}"}]
    )

    return chunks


def embedding_vector_store(chunks):

    # 1. Initialize Embeddings
    embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # cheaper & fast, good for transcripts
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
    )
    
    # 2. Create Vector Store and index chunks
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
        # collection_name="youtube_transcripts"
    )
    print("Indexing complete!")

    return vectorstore


# MMR retriver
def retrival_documents(vectorstore):
    # Maximum Marginal Relevance - avoids returning duplicate/similar chunks
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs= {      
        "k": 4,           # final number of docs to return
        "fetch_k": 20,    # fetch 20, then pick 4 most diverse
        "lambda_mult": 0.7  # 0=max diversity, 1=max relevance
        }
    )

    return mmr_retriever


# Format retrieved documents into a single context string with source references
def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# RAG Chain - retrieval + augmentation + generation
def build_rag_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant that answers questions about a YouTube video using only its transcript.
Use the context below to answer the question. If the answer isn't in the context, say you don't know.
Cite the relevant source timestamp link(s) from the context when possible.

Context:
{context}

Question: {question}

Answer:"""
    )

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain




if __name__ == '__main__':
    load_dotenv()
    video_id = "Gfr50f6ZBvo"

    docs = get_youtube_transcript(video_id)
    # print(docs)

    chunks = chunking_documents(docs, video_id)

    # print(chunks[10].page_content)

    vector_store = embedding_vector_store(chunks)

    retriever = retrival_documents(vector_store)

    rag_chain = build_rag_chain(retriever)

    question = "What is this video about?"
    answer = rag_chain.invoke(question)
    print(answer)


