import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.tools import process_and_store_documents

# Load environment variables from .env file
load_dotenv()

# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")

# Run the document processing pipeline
if __name__ == "__main__":
    folder_path = "internal_documents"
    collection_name = "internal_documents"
    process_and_store_documents(folder_path, collection_name)




# Deleting an existing collection to create a new one
# # Get existing collections
# existing_collections = client.get_collections().collections
# collection_names = [col.name for col in existing_collections]

# # Delete the collection if it exists
# if collection_name in collection_names:
#     print(f"Deleting existing collection '{collection_name}' to update vector size.")
#     client.delete_collection(collection_name=collection_name)

# # Create a new collection with the correct vector size
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
# )

# print(f"Collection '{collection_name}' created with vector size 1536.")