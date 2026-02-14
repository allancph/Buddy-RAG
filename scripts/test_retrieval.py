import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

def test_retrieval():
    print(f"🔎 Connecting to Qdrant: {QDRANT_URL} / {QDRANT_COLLECTION}")
    
    # 1. Setup Embed Model Only (No LLM)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = None
    
    # 2. Connect to Vector Store
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    
    # 3. Create Index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 4. Create Retriever
    retriever = index.as_retriever(similarity_top_k=3)
    
    # 5. Run Retrieval
    query = "Hvor ligger em-trak?" # Eller "Midsomer Norton"
    print(f"❓ Query: {query}")
    
    nodes = retriever.retrieve(query)
    
    print(f"💡 Retrieved {len(nodes)} nodes:")
    for node in nodes:
        print(f" - [{node.score:.2f}] {node.metadata.get('filename', 'N/A')}:\n   {node.text[:200]}...")

if __name__ == "__main__":
    test_retrieval()
