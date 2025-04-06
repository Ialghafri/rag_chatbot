from openai import OpenAI
import sys
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from main import collection_name
#from server.main import openai_api_key
#from server.main import client
#from server.main import embeddings

# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")

client = QdrantClient(url="http://localhost:6333")


# loading all PDFs, text files and word documents in a folder
def load_documents(folder_path):
    loaders = [
        DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
    ]

    documents = []
    seen_files = set()

    for loader in loaders:
        
        try:
            # print(f"Loaded files with {loader.glob}:")
            loaded_docs = loader.load()

            for doc in loaded_docs:
                print(f" - {doc.metadata.get('source', 'No source info')}")
    
            # Removing duplicates by checking the document paths
            for doc in loaded_docs:
                file_path = doc.metadata.get('source')
                if file_path not in seen_files:
                    documents.append(doc)
                    seen_files.add(file_path)
        except Exception as e:
            print(f"Error loading documents with {loader.glob}: {e}")

    print(f"Total files loaded: {len(documents)}")
    return documents

# Process and store documents in Qdrant
def process_and_store_documents(folder_path, collection_name):


    docs = load_documents("internal_documents")

    if not docs:
        print("No documents found. Exiting.")
        return

    # Splitting large documents into smaller chunks for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        print("No document chunks generated. Exiting.")
        return

    # Generate embeddings using openAI
    embeddings = OpenAIEmbeddings()
    chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])  # Access 'page_content' of each chunk


    # Initialize quadrant client
    client = QdrantClient(url="http://localhost:6333")

    collection_name = "test_collection"

    try:
        # Check if the collection exists
        existing_collections = client.get_collections().collections
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return
    
    collection_names = [col.name for col in existing_collections]
    # vector_size = len(chunk_vectors[0]) if chunk_vectors else 1536

    # If collection exists, skips creation
    if collection_name not in collection_names:
        print(f"Creating new collection '{collection_name}' with vector size 1536.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")


    # prepare data for upsert to qdrant
    points = []
    for idx, chunk in enumerate(chunks):
        vector = chunk_vectors[idx]
        points.append(PointStruct(id=idx+1, vector=vector, payload={"chunk_id": f"chunk_{idx+1}", "text": chunk.page_content}))

    try: 
        operation_info = client.upsert(
        collection_name="test_collection",
        wait=True,
        points=points,
    )
    except Exception as e:
        print(f"Error upserting to Qdrant: {e}")

    # # Upsert data to Qdrant
    # operation_info = client.upsert(
    #     collection_name="test_collection",
    #     wait=True,
    #     points=points,
    # )

    #print(operation_info)

def process_uploaded_file(uploaded_file):
    """Process and save the uploaded file, then return the loaded documents"""

    # Save the uploaded file to the local directory
    file_path = Path("internal_documents") / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # load the newly uploaded document
    docs = load_documents("internal_documents")
    return docs

def upsert_documents_to_qdrant(docs, embeddings, collection_name="test_collection"):
    """Split documents into chunks, generate embeddings, and upsert into Qdrant."""

    # Splitting large documents into smaller chunks for retrival 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Generate embeddings for chunks
    chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])

    # Initialize quadrant client
    client = QdrantClient(url="http://localhost:6333")

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

    print(f"Upsert operation completed. Total {len(points)} chunks/points added.")

    return operation_info

def search_qdrant(query_text, client, collection_name, embeddings):
    """Search Qdrant for the most relevant document chunks."""

    query_vector = embeddings.embed_query(query_text)

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
    )

    # Extract retrieved chunks
    retrieved_texts = [result.payload["text"] for result in search_results]
    return retrieved_texts



def format_context(retrieved_texts):
    """Format retrieved document chunks for OpenAI's LLM."""

    context = "\n\n".join(retrieved_texts)
    return f"Use the following information to answer the user's question:\n'n{context}\n\nAnswer concisely."



def ask_openai(query, retrieved_texts, openai_api_key):
    """Query OpenAI's GPT model using retrieved context."""

    context = format_context(retrieved_texts)  # Ensure this function correctly formats context

    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"{context}\n\nUser Query: {query}"}
        ],
        temperature=0.7,
    )

    return response.choices[0].message


def query_rag_system(query_text, client, collection_name, embeddings, openai_api_key):
    """End-to-end function: search Qdrant → get context → call OpenAI → return answer."""

    retrieved_texts = search_qdrant(query_text, client, collection_name, embeddings)

    if not retrieved_texts:
        return "I couldn't find relevant information in the knowledge base."
    
    response = ask_openai(query_text, retrieved_texts, openai_api_key)

    return response


# query = "What are the contents of the Team Collaboration & Project Guidelines Document? Can you summerise it for me?"

# response = query_rag_system(query, client, "test_collection", embeddings, openai_api_key)
# print(response.content)





# # loading all PDFs, text files and word documents in a folder
# def load_documents(folder_path):
#     loaders = [
#         DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader),
#         DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader),
#         DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
#     ]

#     documents = []
#     for loader in loaders:
        
#         print(f"Loaded files with {loader.glob}:")
#         loaded_docs = loader.load()

#         for doc in loaded_docs:
#             print(f" - {doc.metadata.get('source', 'No source info')}")

#         # Removing duplicates by checking the document paths
#         seen_files = set()

#         for doc in loaded_docs:
#             file_path = doc.metadata.get('source')
#             if file_path not in seen_files:
#                 documents.append(doc)
#                 seen_files.add(file_path)

#         print(f"Total files loaded: {len(documents)}")

#     return documents