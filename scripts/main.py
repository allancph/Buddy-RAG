import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Importer LlamaIndex komponenter
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# Importer vores custom hybrid retriever
from hybrid_retrieval import HybridRetriever, get_hybrid_retriever

# Indlæs konfiguration
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2:latest")

# --- Pydantic Modeller ---
class QueryRequest(BaseModel):
    query: str

class SourceNode(BaseModel):
    file_name: str
    page_label: str | None = None
    text_snippet: str
    score: float | None = None

class QueryResponse(BaseModel):
    response: str
    sources: list[SourceNode] = []

# --- Initialisering ---
app = FastAPI(title="Buddy RAG API", description="Persistent RAG Service with Hybrid Search & Reranking")

# Globals
query_engine = None
reranker = None

@app.on_event("startup")
def startup_event():
    global query_engine, reranker
    print("🚀 Starting Buddy RAG API...")
    
    # 1. Setup Models
    print(f"🧠 Loading LLM: {LLM_MODEL_NAME} (Context: 4096)...")
    Settings.llm = Ollama(
        model=LLM_MODEL_NAME, 
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        context_window=4096,
        additional_kwargs={"num_ctx": 4096}
    )
    
    print(f"🧬 Loading Embeddings: {EMBED_MODEL_NAME}...")

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME, 
        base_url=OLLAMA_BASE_URL
    )
    
    # 2. Load Reranker into Memory (This is the heavy lift we do ONCE)
    print("⚖️  Loading Reranker Model (BAAI/bge-reranker-v2-m3)...")
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3", 
        top_n=5, # We take top 5 chunks for the final answer
        device="cpu"
    )
    
    # 3. Setup Retriever
    print("🔍 Initializing Hybrid Retriever...")
    # top_k=20 fetches a broad net before reranking
    retriever = get_hybrid_retriever(top_k=20)
    
    # 4. Create Query Engine
    print("⚙️  Assembling Query Engine...")
    query_engine = VectorStoreIndex.from_vector_store(
        vector_store=QdrantVectorStore(
            client=QdrantClient(url=QDRANT_URL), 
            collection_name=QDRANT_COLLECTION
        )
    ).as_query_engine(
        llm=Settings.llm,
        node_postprocessors=[reranker],
        response_mode="compact"
    )
    print("✅ System Ready!")

@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """
    FastAPI endpoint that uses the pre-loaded engine.
    """
    print(f"📨 Received query: {request.query}")
    
    # Custom prompt to ensure Danish answers
    QA_PROMPT = (
        "Du er en hjælpsom assistent. Svar på spørgsmålet baseret på nedenstående kontekst.\n"
        "Hvis svaret ikke findes i konteksten, så sig det.\n"
        "Svar kort og præcist på Dansk.\n\n"
        "Kontekst:\n---------------------\n{context_str}\n---------------------\n\n"
        "Spørgsmål: {query_str}\nSvar:"
    )
    
    # Update prompt for this specific query
    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(QA_PROMPT)})
    
    response = query_engine.query(request.query)
    
    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append(SourceNode(
            file_name=meta.get("file_name", "N/A"),
            page_label=meta.get("page_label"),
            text_snippet=node.get_content()[:200] + "...",
            score=node.score
        ))

    return QueryResponse(response=str(response), sources=sources)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
