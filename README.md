# Medical Chatbot Gen AI

A medical question-and-answer assistant built with Streamlit, LangChain, Pinecone, and Groq. This project creates a searchable medical knowledge base from PDF documents, embeds the content with Hugging Face sentence transformers, and uses a large language model to generate detailed, context-aware medical responses.

## Features

- Streamlit web interface for asking medical questions
- PDF ingestion pipeline to extract medical documents from `Data/`
- Text splitting into vector-friendly chunks
- Hugging Face embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Pinecone vector store for fast retrieval
- Groq LLM powered answer generation
- Domain-specific prompt template for medical diagnosis, symptoms, treatment, and prevention

## Repository Structure

- `app.py` - Streamlit application for query input and answer display
- `src/helper.py` - PDF loading, text splitting, and embedding helper functions
- `src/prompt.py` - Medical question prompt template
- `src/store.py` - Pinecone index creation and document upload pipeline
- `Data/` - Folder for medical PDF source documents
- `requirements.txt` - Python dependencies
- `setup.py` - package metadata

## Requirements

- Python 3.10+ recommended
- `streamlit`
- `python-dotenv`
- `sentence-transformers`
- `langchain`
- `pinecone[grpc]`
- `langchain-pinecone`
- `langchain_community`
- `pypdf`
- `groq` (used by `app.py`)

Install dependencies:

```bash
pip install -r requirements.txt
pip install groq
```

Alternatively, install the package in editable mode:

```bash
pip install -e .
```

## Environment Variables

Create a `.env` file at the project root with values for:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

## Setup

1. Add medical PDF documents to the `Data/` folder.
2. Create or update `.env` with your API keys.
3. Build the vector store by running:

```bash
python src/store.py
```

This script:

- loads PDFs from `Data/`
- chunks the extracted text
- creates or reuses a Pinecone index named `medicalbot`
- uploads document embeddings to Pinecone

## Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit and ask medical questions to retrieve answers from the indexed knowledge base.

## Customization

- Add new PDFs to `Data/` and rerun `python src/store.py` to refresh the Pinecone index.
- Adjust prompt behavior in `src/prompt.py`.
- Modify model settings or retrieval logic in `app.py`.

## Notes

- The app is intended for educational or prototype use only. It is not a substitute for professional medical advice.
- Ensure your Pinecone account and Groq API access are active and configured correctly.
