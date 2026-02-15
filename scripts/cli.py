import os
import sys
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, PromptTemplate
from hybrid_retrieval import retrieve_and_rerank

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL_NAME = "llama3.1:8b"

# Initialize LLM
llm = Ollama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    request_timeout=120.0,
    context_window=4096,
    additional_kwargs={"num_ctx": 4096}
)

# Define a simple RAG Prompt
QA_PROMPT_TMPL = (
    "Du er en hjælpsom assistent for en B737 pilot. Svar på spørgsmålet baseret på nedenstående kontekst.\n"
    "Hvis svaret ikke findes i konteksten, så sig det.\n"
    "Svar kort og præcist på Dansk.\n\n"
    "Kontekst:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Spørgsmål: {query_str}\n"
    "Svar:"
)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

def generate_answer(query: str):
    """
    1. Retrieve relevant context (Hybrid + Rerank)
    2. Format prompt
    3. Generate answer via LLM
    """
    print(f"\n✈️  Behandler spørgsmål: '{query}'")
    
    # 1. Retrieval
    print("🔍 Henter viden...")
    nodes = retrieve_and_rerank(query, top_k=10, rerank_top_n=3)
    
    if not nodes:
        print("⚠️  Ingen relevante data fundet.")
        return "Jeg kunne ikke finde information om dette i databasen."

    # 2. Format Context
    context_str = "\n\n".join([n.node.get_text() for n in nodes])
    print(f"📄 Brugte {len(nodes)} kilder til kontekst.")

    # 3. Generate
    print("🧠 Genererer svar med Llama 3.1...")
    prompt = qa_prompt.format(context_str=context_str, query_str=query)
    
    response = llm.complete(prompt)
    
    return str(response)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "forklar MOB funktionen"
        
    answer = generate_answer(query)
    print("\n" + "="*30)
    print("SVAR:")
    print("="*30)
    print(answer)
    print("="*30 + "\n")