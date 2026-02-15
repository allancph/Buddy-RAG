import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import Settings, VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import List, Optional

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL")

# Initialize Global Settings
Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)
Settings.llm = None 

class HybridRetriever(BaseRetriever):
    """
    Custom Hybrid Retriever combining BM25 and Vector Search.
    """
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)

        all_nodes = []
        node_ids = set()
        
        for node in vector_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)
        
        for node in bm25_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)

        return all_nodes

def get_hybrid_retriever(top_k=5):
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    print("⏳ Building BM25 Retriever (fetching docs from Qdrant)...")
    try:
        all_docs = []
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for point in points:
                text_content = point.payload.get("text", "")
                if not text_content:
                    node_content = point.payload.get("_node_content", "{}")
                    if isinstance(node_content, str):
                        try:
                            import json
                            nc = json.loads(node_content)
                            text_content = nc.get("text", "")
                        except:
                            pass
                
                metadata = point.payload if isinstance(point.payload, dict) else {}

                node = TextNode(
                    text=text_content,
                    id_=point.id,
                    metadata=metadata
                )
                all_docs.append(node)
            
            offset = next_offset
            if offset is None:
                break
        
        print(f"📄 Fetched {len(all_docs)} nodes for BM25.")
        bm25_retriever = BM25Retriever.from_defaults(nodes=all_docs, similarity_top_k=top_k)
        
    except Exception as e:
        print(f"⚠️ Failed to build BM25 Retriever: {e}")
        return vector_retriever

    return HybridRetriever(vector_retriever, bm25_retriever)

def retrieve_and_rerank(query, top_k=10, rerank_top_n=3):
    # 1. Get Hybrid Retriever
    retriever = get_hybrid_retriever(top_k=top_k)
    
    # 2. Retrieve Nodes
    print(f"🔍 Retrieving nodes for query: '{query}'")
    nodes = retriever.retrieve(query)
    print(f"📊 Found {len(nodes)} candidate nodes.")
    
    # 3. Rerank using Local SentenceTransformer (BGE-M3)
    try:
        # LlamaIndex has SentenceTransformerRerank in core.postprocessor.sbert_rerank
        from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
        
        print(f"⚖️ Reranking with Local BGE-M3 (BAAI/bge-reranker-v2-m3)...")
        # We use the official HF model name. It will download to cache.
        # device="cpu" is safer for LXC unless GPU passthrough is confirmed.
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3", 
            top_n=rerank_top_n,
            device="cpu" 
        )
        
        query_bundle = QueryBundle(query_str=query)
        ranked_nodes = reranker.postprocess_nodes(nodes, query_bundle)
        print("✅ Reranking successful.")
        return ranked_nodes
    except ImportError:
        print("⚠️ `sentence-transformers` library likely missing.")
        print("Please run: pip install sentence-transformers")
        return nodes[:rerank_top_n]
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        return nodes[:rerank_top_n]

if __name__ == "__main__":
    test_query = "forklar MOB funktionen"
    results = retrieve_and_rerank(test_query, top_k=10, rerank_top_n=3)
    
    print("\n🏆 Top Results:")
    for i, node in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {node.score}) ---")
        print(f"Text: {node.node.get_text()[:200]}...")
        print(f"Metadata: {node.node.metadata}")
