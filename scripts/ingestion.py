import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from robustness import retry_with_backoff

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL")

def sanitize_metadata(documents: list[Document]) -> list[Document]:
    """
    Sanitizes metadata in documents to ensure Qdrant compatibility.
    Converts complex types (dict, list of list) to JSON strings.
    """
    sanitized_docs = []
    for doc in documents:
        new_metadata = {}
        for key, value in doc.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                new_metadata[key] = value
            elif isinstance(value, list):
                # Check if list contains simple types
                if all(isinstance(x, (str, int, float, bool)) for x in value):
                    new_metadata[key] = value
                else:
                    # Convert complex list to JSON string
                    new_metadata[key] = json.dumps(value)
            else:
                # Convert dict or other objects to JSON string
                new_metadata[key] = json.dumps(value)
        
        doc.metadata = new_metadata
        sanitized_docs.append(doc)
    
    return sanitized_docs

def get_pipeline():
    """
    Creates and returns the LlamaIndex IngestionPipeline.
    """
    # 1. Initialize Qdrant Client
    client = QdrantClient(url=QDRANT_URL)
    
    # 2. Initialize Vector Store
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    
    # 3. Initialize Embedding Model (Ollama)
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )

    # 4. Initialize Transformations
    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        embed_model,
    ]

    # 5. Create Ingestion Pipeline
    pipeline = IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
    )

    return pipeline

@retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2)
def ingest_documents(documents: list[Document]):
    """
    Ingests a list of LlamaIndex Document objects into the pipeline.
    Sanitizes metadata first.
    """
    print(f"🧹 Sanitizing metadata for {len(documents)} documents...")
    documents = sanitize_metadata(documents)

    pipeline = get_pipeline()
    
    print(f"🚀 Starting ingestion of {len(documents)} documents...")
    
    # Run the pipeline
    nodes = pipeline.run(documents=documents, show_progress=True)
    
    print(f"✅ Ingestion complete! Processed {len(nodes)} nodes.")
    return nodes

if __name__ == "__main__":
    # Test with a dummy document containing complex metadata
    print("🧪 Running test ingestion with complex metadata...")
    complex_metadata = {
        "file_name": "test_doc_complex.txt",
        "tags": ["test", "complex", "metadata"],
        "author": {"name": "Allan", "role": "Pilot"}, # Dict -> should be JSON string
        "metrics": [1, 2, 3], # List of int -> should be kept
        "nested_list": [[1, 2], [3, 4]] # List of list -> should be JSON string
    }
    
    dummy_doc = Document(
        text="This is a test document with complex metadata to verify sanitization.",
        metadata=complex_metadata
    )
    
    try:
        nodes = ingest_documents([dummy_doc])
        print("🎉 Test passed: Metadata sanitized successfully.")
        print("Sanitized Metadata:", nodes[0].metadata)
    except Exception as e:
        print(f"❌ Test failed: {e}")
