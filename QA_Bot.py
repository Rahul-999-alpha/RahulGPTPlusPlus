import os
import gradio as gr
import hashlib
import json
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import fitz  # PyMuPDF
from ctransformers import AutoModelForCausalLM
import tiktoken

# Constants
VECTORSTORE_DIR = "vectorstore"
HASH_STORE_FILE = "hash_store.json"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Load or initialize hash store
if os.path.exists(HASH_STORE_FILE):
    with open(HASH_STORE_FILE, "r") as f:
        existing_hashes = set(json.load(f))
else:
    existing_hashes = set()

# Load embedding model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = AutoModelForCausalLM.from_pretrained(
    "models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0  # Set >0 if GPU is available
)

def extract_text_from_pdf(pdf_path: str) -> List[tuple[str, int]]:
    chunks = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start = 1):
            text = page.get_text()
            chunks.append((text,i))
    return chunks

def local_llm_answer(prompt: str) -> str:
    # You can tweak this prompt format as needed
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return llm(formatted_prompt).strip()

def compress_text(text: str) -> str:
    # Rule-based simplification (safe for legal docs)
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]  # remove blanks
    text = " ".join(lines)

    # Optional: limit redundant phrases or verbose structures
    replacements = {
        "in accordance with the provisions of": "under",
        "pursuant to": "under",
        "with regard to": "regarding",
        "has been decided that": "decided",
        "the following shall apply": "applies",
    }
    for phrase, replacement in replacements.items():
        text = text.replace(phrase, replacement)

    # Truncate overly long strings if needed (backup safety)
    return text[:1200]  # keep under chunk size

def process_pdfs(pdf_files: List[gr.File]) -> str:
    new_chunks = []
    new_hashes = []

    for pdf in pdf_files:
        pages = extract_text_from_pdf(pdf.name)
        content_hash = hashlib.sha256("".join(text for text, _ in pages).encode()).hexdigest()

        if content_hash in existing_hashes:
            continue  # Skip duplicates

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        
        for text, page_num in pages:
            chunks = splitter.split_text(text)
            compressed_chunks = [compress_text(chunk) for chunk in chunks]
            new_chunks.extend([
                Document(page_content=chunk, metadata = {"source": os.path.basename(pdf.name), "page": page_num})
                for chunk in compressed_chunks
            ])
        new_hashes.append(content_hash)

    if new_chunks:
        if os.path.exists("faiss_index"):
            db = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)
            db.add_documents(new_chunks)
        else:
            db  = FAISS.from_documents(new_chunks, embedding_function)
        db.save_local("faiss_index")
        
        existing_hashes.update(new_hashes)
        # Save updated hash store
        with open(HASH_STORE_FILE, "w") as f:
            json.dump(list(existing_hashes), f)

        return f"{len(new_chunks)} new chunks added to the vector DB."
    else:
        return "No new documents processed (duplicates skipped)."

def build_limited_prompt(docs, question, max_tokens=512, reserved_output_tokens=100):
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-like tokenizer
    prompt_header = "Use the following RBI circulars to answer the user's question. Cite source and page.\n\n"
    prompt_question = f"\n\nQuestion: {question}\nAnswer:"
    
    context_chunks = []
    total_tokens = len(encoding.encode(prompt_header + prompt_question))

    for doc in docs:
        chunk = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        tagged_chunk = f"[{source} - page {page}]\n{chunk}"
        chunk_tokens = len(encoding.encode(tagged_chunk))
        if total_tokens + chunk_tokens > max_tokens - reserved_output_tokens:
            break
        context_chunks.append(tagged_chunk)
        total_tokens += chunk_tokens
        
    context = "\n\n".join(context_chunks)
    return f"{prompt_header}{context}{prompt_question}"

def query_rag(user_query: str) -> str:
    if not os.path.exists("faiss_index"):
        return "No documents available to search. Please upload some PDFs first."

    db = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(user_query)

    if not docs or all (len(doc.page_content.strip()) < 100 for doc in docs):
        return f"(Fallback response from LLM)\n\n{local_llm_answer(user_query)}"

    prompt = build_limited_prompt(docs, user_query, max_tokens = 512, reserved_output_tokens = 100)
    return llm(prompt)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 RBI Circulars Q&A Chatbot")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload RBI Circular PDFs")
    upload_output = gr.Textbox(visible=False)

    with gr.Row():
        query = gr.Textbox(label="Ask a Question")
        output = gr.Textbox(label="Answer")

    # Automatically process PDFs once uploaded
    pdf_input.upload(fn=process_pdfs, inputs=pdf_input, outputs=upload_output)

    # Respond to query
    query.submit(fn=query_rag, inputs=query, outputs=output)

demo.launch()
