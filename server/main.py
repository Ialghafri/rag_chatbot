import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# Load environment variables from .env file
load_dotenv()

# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")


# loading all PDFs, text files and word documents in a folder
def load_documents(folder_path):
    loaders = [
        DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
    ]

    documents = []
    for loader in loaders:
        
        print(f"Loaded files with {loader.glob}:")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            print(f" - {doc.metadata.get('source', 'No source info')}")

        # Removing duplicates by checking the document paths
        seen_files = set()

        for doc in loaded_docs:
            file_path = doc.metadata.get('source')
            if file_path not in seen_files:
                documents.append(doc)
                seen_files.add(file_path)

        print(f"Total files loaded: {len(documents)}")

    return documents

docs = load_documents("internal_documents")









