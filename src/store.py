import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf,
    text_split,
    download_huggingface_embeddings
)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# ==========================
# 1. Load PDF
# ==========================
extracted_data = load_pdf("Data/")
print("PDF Loaded")

# ==========================
# 2. Split into chunks
# ==========================
text_chunks = text_split(extracted_data)
print(f"Chunks Created: {len(text_chunks)}")

# ==========================
# 3. Load Embeddings
# ==========================
embeddings = download_huggingface_embeddings()
print("Embeddings Ready")

# ==========================
# 4. Connect Pinecone
# ==========================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# ==========================
# 5. Create Index if needed
# ==========================
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:

    print("Creating Pinecone Index...")

    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 output size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index Created Successfully")

else:
    print("Index Already Exists")

# ==========================
# 6. Upload Documents
# ==========================
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Pinecone Upload Complete")