import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

def test_query():
    print(f"🔎 Connecting to Qdrant: {QDRANT_URL} / {QDRANT_COLLECTION}")
    
    # 1. Setup Models
    Settings.context_window = 4096
    Settings.num_output = 512
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0, context_window=4096)
    
    # 2. Connect to Vector Store
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    
    # 3. Create Index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 4. Create Query Engine
    query_engine = index.as_query_engine()
    
    # 5. Run Query
    query = "forklar MOB funktionen"
    print(f"❓ Query: {query}")
    
    response = query_engine.query(query)
    
    print(f"💡 Response: {response}")
    print("\n📄 Source Nodes:")
    for node in response.source_nodes:
        print(f" - [{node.score:.2f}] {node.metadata.get('filename', 'N/A')}: {node.text[:100]}...")

if __name__ == "__main__":
    test_query()
