load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import qdrant_client

# Importer de nødvendige LlamaIndex komponenter
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Indlæs konfiguration fra .env filen
load_dotenv()

# --- Pydantic Modeller (API datastruktur) ---
class QueryRequest(BaseModel):
    query: str

class SourceNode(BaseModel):
    file_name: str
    page_label: str | None = None
    text_snippet: str

class QueryResponse(BaseModel):
    response: str
    sources: list[SourceNode] = []

# --- Initialisering af App og Forbindelser ---
app = FastAPI(title="Buddy RAG API", description="API til at forespørge et LlamaIndex RAG system bygget på Qdrant og Ollama.")

# 1. Konfigurer globale LlamaIndex indstillinger
# Dette sikrer, at alle dele af LlamaIndex bruger de samme modeller
Settings.llm = Ollama(model=os.getenv("LLM_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))
Settings.embed_model = OllamaEmbedding(model_name=os.getenv("EMBED_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))

# 2. Forbind til den eksisterende Qdrant vector database
qdrant_client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"))
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=os.getenv("QDRANT_COLLECTION"))

# 3. Opret et 'index' objekt fra vores vector store
# Dette objekt repræsenterer vores vidensdatabase
index = VectorStoreIndex.from_vector_store(vector_store)

# 4. Opret en query engine - den primære grænseflade til at stille spørgsmål
query_engine = index.as_query_engine(streaming=False)


# --- API Endpoints ---
@app.get("/")
def read_root():
    """Et simpelt endpoint til at tjekke om API'et kører."""
    return {"status": "Buddy RAG API is running"}


@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """
    Modtager en forespørgsel, sender den til RAG-systemet,
    og returnerer svaret sammen med kilde-dokumenterne.
    """
    # Udfør selve forespørgslen
    response_object = query_engine.query(request.query)

    # Uddrag kilde-nodes fra svaret for at give reference
    source_nodes = []
    for node in response_object.source_nodes:
        # Uddrag relevante metadata fra hver node
        file_name = node.metadata.get("file_name", "N/A")
        page_label = node.metadata.get("page_label")
        # Begræns længden af tekst-snip for et pænere output
        text_snippet = node.get_content(metadata_mode="all")[:300] + "..."

        source_nodes.append(
            SourceNode(
                file_name=file_name,
                page_label=page_label,
                text_snippet=text_snippet,
            )
        )

    return QueryResponse(response=str(response_object), sources=source_nodes)