from sentence_transformers import SentenceTransformer
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Max Length:', model.max_seq_length)

file_path = 'files/docs.csv'
target_column = 'text_data'

documents : list[str] = [
    "The quick brown rabbit jumps over the lazy frogs.",
    "A fast tan hare leaps above sleepy toads.",
    "Artificial intelligence is transforming various industries.",
    "AI is having a significant impact on the future of work.",
    "The weather today is sunny and warm.",
    "It's a beautiful day with clear skies and high temperatures.",
    "The old wooden door creaked loudly in the wind.",
    "A fluffy white cat slept peacefully on the sunny windowsill.",
    "Freshly baked bread filled the kitchen with a warm aroma.",
    "The little girl giggled as she chased butterflies in the garden.",
    "Heavy rain poured down, creating puddles on the pavement.",
    "A tall green tree swayed gently in the light breeze.",
    "The scientist carefully mixed the colorful liquids in the lab.",
    "A delicious cup of coffee helped him start his busy day.",
    "Bright stars twinkled in the dark night sky.",
    "The new book quickly became a bestseller."
]

document_embeddings = model.encode(documents)
vector_database = list(zip(documents, document_embeddings))

print(f"Shape of document embeddings: {document_embeddings.shape}")
# Output will be something like: (16, 384) - 16 documents, each with a 384-dimensional vector


from typing import Any
from sklearn.metrics.pairwise import cosine_similarity

def search_documents(query: str, vector_database: list[tuple[str, Any]], model, top_n=3):
    query_embedding = model.encode(query)
    similarity_scores = cosine_similarity([query_embedding], [embedding for doc, embedding in vector_database])[0]
    ranked_results = sorted(zip(vector_database, similarity_scores), key=lambda x: x[1], reverse=True)
    return [(doc, score) for (doc, _), score in ranked_results[:top_n]]

def print_result(results : list[tuple[str, Any]], query):
    print(f"\nTop relevant documents for query: '{query}'")
    for doc, score in results:
        print(f"- '{doc}' (Score: {score:.4f})")


query = "What are the impacts of AI?"
results = search_documents(query, vector_database, model)
print_result(results, query)

query_weather = "Tell me about the weather."
results = search_documents(query_weather, vector_database, model)
print_result(results, query_weather)

query_cat = "A cat is sleeping."
results = search_documents(query_cat, vector_database, model)
print_result(results, query_cat)

