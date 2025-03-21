from qdrant_client.http import models
from main import client
from main import collection_name
from main import embeddings
from main import query_rag_system
from main import openai_api_key



# # Testing retreival with a query 
# query_text = "What are the departments roles and repsonibilities?"
# query_vector = embeddings.embed_query(query_text)

# search_results = client.search(
#     collection_name="test_collection",
#     query_vector=query_vector,
#     limit=5  # Retrieve top 5 matches
# )

# for result in search_results:
#     print(f"Score: {result.score}, Text: {result.payload['text']}")

query = "What are the contents of the Team Collaboration & Project Guidelines Document? Can you summerise it for me?"

response = query_rag_system(query, client, "test_collection", embeddings, openai_api_key)
print(response)