import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings 
from qdrant_client import QdrantClient
import os
import sys

# Get the absolute path of the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import from server
from server.tools import query_rag_system, client, embeddings, openai_api_key


# Streamlit frontend
st.title("RAG Chatbot")
st.write("Ask any questions and get answers from the knowledge base")

# User input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        
        response = query_rag_system(query, client, "test_collection", embeddings, openai_api_key)
        st.subheader("Response:")
        st.write(response.content)
    else:
        st.warning("Please enter a question")





