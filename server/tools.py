from main import collection_name
from main import openai_api_key
from langchain_community.chat_models import ChatOpenAI
from main import client
from main import embeddings
from openai import OpenAI


def search_qdrant(query_text, client, collection_name, embeddings):
    """Search Qdrant for the most relevant docuemnt chunks."""

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


query = "What are the contents of the Team Collaboration & Project Guidelines Document? Can you summerise it for me?"

response = query_rag_system(query, client, "test_collection", embeddings, openai_api_key)
print(response.content)

