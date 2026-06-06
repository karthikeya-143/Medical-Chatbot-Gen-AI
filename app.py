import streamlit as st
from dotenv import load_dotenv

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
# LOAD RETRIEVER
# ----------------------------------

docsearch = load_vectorstore()

retriever = docsearch.as_retriever(
    search_kwargs={"k": 3}
)

# ----------------------------------
# USER INPUT
# ----------------------------------

query = st.text_input(
    "Ask a Medical Question"
)

if query:

    with st.spinner("Searching..."):

        docs = retriever.get_relevant_documents(query)

        answer = ""

        for doc in docs:
            answer += doc.page_content + "\n\n"

        st.success("Answer Generated")

        st.markdown(answer)