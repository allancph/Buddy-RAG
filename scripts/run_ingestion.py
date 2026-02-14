import os
import requests
import json
import time
import re
from dotenv import load_dotenv
from llama_index.core import Document
from ingestion import ingest_documents

# Load environment variables
load_dotenv()

# Configuration
PAPERLESS_URL = "http://192.168.0.102:8000"
PAPERLESS_TOKEN = "497dcbd54f4bc168678662947c857d4295603dd7" # Bør flyttes til .env, men hardcoded i original script
DOCLING_URL = "http://192.168.0.110:5001"
PAPERLESS_HEADERS = {"Authorization": f"Token {PAPERLESS_TOKEN}"}

def get_documents(limit=10):
    """Fetch recent documents from Paperless."""
    print(f"📥 Fetching last {limit} documents from Paperless...")
    url = f"{PAPERLESS_URL}/api/documents/?page_size={limit}&ordering=-created"
    try:
        res = requests.get(url, headers=PAPERLESS_HEADERS)
        res.raise_for_status()
        return res.json()['results']
    except Exception as e:
        print(f"❌ Failed to fetch documents: {e}")
        return []

def download_document(doc_id, original_filename):
    """Downloads a document from Paperless to a temp file."""
    url = f"{PAPERLESS_URL}/api/documents/{doc_id}/download/"
    try:
        res = requests.get(url, headers=PAPERLESS_HEADERS, stream=True)
        res.raise_for_status()
        
        temp_path = f"/tmp/{original_filename}"
        with open(temp_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_path
    except Exception as e:
        print(f"❌ Failed to download document {doc_id}: {e}")
        return None

def process_docling_async(file_path):
    """Processes a file with Docling Async API."""
    submit_url = f"{DOCLING_URL}/v1/convert/file/async"
    
    # Enable Picture Description & Table Structure
    data = {
        "options": json.dumps({
            "do_picture_description": True,
            "do_table_structure": True,
            "image_export_mode": "placeholder"
        })
    }
    
    files = [('files', (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf'))]
    
    try:
        print(f"🚀 Sending {os.path.basename(file_path)} to Docling...")
        res = requests.post(submit_url, files=files, data=data)
        res.raise_for_status()
        task_id = res.json()['task_id']
    except Exception as e:
        print(f"❌ Docling upload failed: {e}")
        return None

    # Poll for completion
    poll_url = f"{DOCLING_URL}/v1/status/poll/{task_id}"
    result_url = f"{DOCLING_URL}/v1/result/{task_id}"
    
    start = time.time()
    while time.time() - start < 3600: # 60 min timeout
        try:
            status_res = requests.get(poll_url)
            status_res.raise_for_status()
            status = status_res.json().get('task_status')
            
            if status == 'success':
                print("✅ Docling processing complete.")
                return requests.get(result_url).json()
            elif status == 'failure':
                print("❌ Docling task failed.")
                return None
            elif status == 'pending' or status == 'started':
                time.sleep(5) # Wait before polling again
            else:
                 print(f"Unknown status: {status}")
                 time.sleep(5)
                 
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(5)
            
    print("❌ Docling timeout.")
    return None

def clean_markdown(text):
    """Removes data URIs from markdown to save space."""
    # Remove images embedded as data URIs
    text = re.sub(r'!\[.*?\]\(data:image\/.*?\)', '', text)
    return text

def run_ingestion_test():
    """Main execution flow."""
    # 1. Get documents
    docs = get_documents(limit=5) # Get 5, filter manually
    if not docs:
        print("No documents found.")
        return

    llama_docs = []

    for doc in docs:
        doc_id = doc['id']
        title = doc['title']
        filename = doc['original_file_name']
        
        # Skip known large/problematic files for system test
        if "SO39DSOwnersManual" in filename:
            print(f"⚠️ Skipping {filename} (Too large for quick test)")
            continue
            
        print(f"\n--- Processing [{doc_id}] {title} ---")
        
        # Limit to 2 document for this test
        if len(llama_docs) >= 2:
            break
        
        # 2. Download
        fpath = download_document(doc_id, filename)
        if not fpath: continue
        
        # 3. Convert with Docling
        data = process_docling_async(fpath)
        
        # Clean up temp file
        if os.path.exists(fpath): os.remove(fpath)

        if data and 'document' in data:
            raw_md = data['document'].get('md_content', '')
            clean_md = clean_markdown(raw_md)
            
            print(f"📄 DEBUG: First 500 chars of content:\n{clean_md[:500]}")
            
            # 4. Create LlamaDocument
            # Metadata structure matching what we want in Qdrant
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "filename": filename,
                "source": "paperless",
                "created": doc['created'],
                "tags": doc.get('tags', []) # List of IDs, usually
            }
            
            llama_doc = Document(text=clean_md, metadata=metadata)
            llama_docs.append(llama_doc)
        else:
            print(f"❌ Skipping {doc_id} due to Docling failure.")

    # 5. Ingest into Pipeline
    if llama_docs:
        print(f"\n🚀 Sending {len(llama_docs)} document(s) to Ingestion Pipeline...")
        ingest_documents(llama_docs)
    else:
        print("No documents successfully processed.")

if __name__ == "__main__":
    run_ingestion_test()
