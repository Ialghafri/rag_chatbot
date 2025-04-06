import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings 
from qdrant_client import QdrantClient
import os
import sys
from pathlib import Path

# Get the absolute path of the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import from server
from server.tools import query_rag_system, client, openai_api_key, process_uploaded_file, upsert_documents_to_qdrant

# Streamlit frontend
st.title("RAG Chatbot")
st.write("Ask any questions and get answers from the knowledge base")

# File uploader for adding documents to the knowledge base

with st.container():
    st.subheader("Upload a Document")
    uploaded_file = st.file_uploader("Upload a document to add to the knowledge base", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:

        # # Saving the uploaded file and storing it for processing
        file_path = Path("internal_documents") / uploaded_file.name

        with open(file_path, "wb") as f:
          f.write(uploaded_file.getbuffer())

        docs = process_uploaded_file(uploaded_file)

        embeddings = OpenAIEmbeddings()
        upsert_documents_to_qdrant(docs, embeddings)

        st.success(f"Successfully uploaded and processed {uploaded_file.name}.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

embeddings = OpenAIEmbeddings()

# User input
query = st.chat_input("Enter your question:")

# if st.button("Ask"):
if query:

    st.session_state.messages.append({"role": "user", "content": query})
    response = query_rag_system(query, client, "test_collection", embeddings, openai_api_key)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    #with st.chat_message("assistant"):
    #    st.markdown(response.content)
    st.rerun()





