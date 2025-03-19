import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models



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

# Splitting large documents into smaller chunks for retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Generate embeddings using openAI
embeddings = OpenAIEmbeddings()
chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])  # Access 'page_content' of each chunk


# Initialize quadrant client
client = QdrantClient(url="http://localhost:6333")

collection_name = "test_collection"

# Check if the collection exists
existing_collections = client.get_collections().collections
collection_names = [col.name for col in existing_collections]

# If collection exists, skips creation
if collection_name not in collection_names:
    print(f"Creating new collection '{collection_name}' with vector size 1536.")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")


# prepare points for upsert
points = []
for idx, chunk in enumerate(chunks):
    vector = chunk_vectors[idx]
    points.append(PointStruct(id=idx+1, vector=vector, payload={"chunk_id": f"chunk_{idx+1}", "text": chunk.page_content}))


# Upsert data to Qdrant
operation_info = client.upsert(
    collection_name="test_collection",
    wait=True,
    points=points,
)

print(operation_info)


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