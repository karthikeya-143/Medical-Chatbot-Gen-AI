import os
import streamlit as st
from dotenv import load_dotenv

from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical AI Assistant")
st.write("Ask questions from your Medical Knowledge Base")

# ----------------------------------
# EMBEDDINGS
# ----------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ----------------------------------
# VECTOR STORE
# ----------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medicalbot",
        embedding=embeddings
    )

    return docsearch

# ----------------------------------
# GROQ CLIENT
# ----------------------------------

@st.cache_resource
def load_llm():

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    return client

# ----------------------------------
# LOAD COMPONENTS
# ----------------------------------

docsearch = load_vectorstore()

retriever = docsearch.as_retriever(
    search_kwargs={"k": 5}
)

client = load_llm()

# ----------------------------------
# USER INPUT
# ----------------------------------

query = st.text_input(
    "Ask a Medical Question"
)

if query:

    with st.spinner("Searching Medical Knowledge Base..."):

        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an expert medical assistant.

Use the provided medical context to answer the user's question.

Instructions:
- Give a detailed answer in paragraph form.
- Explain the condition clearly.
- Include causes, symptoms, and treatment if available.
- Do not simply copy the context.
- If the answer is not found, say:
  "I could not find sufficient information in the medical knowledge base."

Medical Context:
{context}

Question:
{query}

Detailed Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        st.success("Answer Generated")

        st.write(answer)