import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# Load environment variables from .env file
load_dotenv()

# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
quadrant_api_key = os.getenv("QUADRANT_API_KEY")


# loading all PDFs, text files and word documents in a folder
def load_documents(folder_path):
    loaders = [
        DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents

docs = load_documents("internal_documents/")

print(f"Loader {len(docs)} documents.")







