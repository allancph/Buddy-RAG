import os
import requests
import json
import time
import re
from dotenv import load_dotenv
from llama_index.core import Document
from ingestion import ingest_documents

# Load environment variables
load_dotenv(dotenv_path='/root/Buddy-RAG/.env')

# Configuration
PAPERLESS_URL = "http://192.168.0.102:8000"
PAPERLESS_TOKEN = "497dcbd54f4bc168678662947c857d4295603dd7"
DOCLING_URL = "http://192.168.0.110:5001"
PAPERLESS_HEADERS = {"Authorization": f"Token {PAPERLESS_TOKEN}"}

# --- Funktioner kopieret fra run_ingestion.py ---

def get_document_details(doc_id):
    """Fetch specific document details from Paperless."""
    print(f"📥 Fetching details for document ID {doc_id} from Paperless...")
    url = f"{PAPERLESS_URL}/api/documents/{doc_id}/"
    try:
        res = requests.get(url, headers=PAPERLESS_HEADERS)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"❌ Failed to fetch document details for ID {doc_id}: {e}")
        return None

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
        print(f"✅ Download complete: {temp_path}")
        return temp_path
    except Exception as e:
        print(f"❌ Failed to download document {doc_id}: {e}")
        return None

def process_docling_async(file_path):
    """Processes a file with Docling Async API."""
    submit_url = f"{DOCLING_URL}/v1/convert/file/async"
    data = {"options": json.dumps({"do_picture_description": True, "do_table_structure": True, "image_export_mode": "placeholder"})}
    files = [('files', (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf'))]
    
    try:
        print(f"🚀 Sending {os.path.basename(file_path)} to Docling...")
        res = requests.post(submit_url, files=files, data=data)
        res.raise_for_status()
        task_id = res.json()['task_id']
    except Exception as e:
        print(f"❌ Docling upload failed: {e}")
        return None

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
                print(f"❌ Docling task failed. Full response: {requests.get(result_url).text}")
                return None
            else:
                print(f"⏳ Docling status: {status}... waiting")
                time.sleep(10)
                 
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(10)
            
    print("❌ Docling timeout.")
    return None

def clean_markdown(text):
    """Removes data URIs from markdown to save space."""
    return re.sub(r'!\[.*?\]\(data:image\/.*?\)', '', text)

# --- Hovedfunktion ---

def main(doc_id_to_process):
    """Main execution flow for a single document."""
    # 1. Get document details
    doc = get_document_details(doc_id_to_process)
    if not doc:
        return

    doc_id = doc['id']
    title = doc['title']
    filename = doc['original_file_name']
        
    print(f"\n--- Processing [{doc_id}] {title} ---")
    
    # 2. Download
    fpath = download_document(doc_id, filename)
    if not fpath: return
    
    # 3. Convert with Docling
    data = process_docling_async(fpath)
    
    # Clean up temp file
    if os.path.exists(fpath): os.remove(fpath)

    if data and 'document' in data:
        raw_md = data['document'].get('md_content', '')
        clean_md = clean_markdown(raw_md)
        
        # 4. Create LlamaDocument
        metadata = {
            "doc_id": doc_id,
            "title": title,
            "filename": filename,
            "source": "paperless",
            "created": doc['created'],
            "tags": doc.get('tags', [])
        }
        
        llama_doc = Document(text=clean_md, metadata=metadata)
        
        # 5. Ingest into Pipeline
        print(f"\n🚀 Sending document to Ingestion Pipeline...")
        ingest_documents([llama_doc])
    else:
        print(f"❌ Skipping {doc_id} due to Docling failure.")

if __name__ == "__main__":
    DOCUMENT_ID = 3
    main(DOCUMENT_ID)
