from typing import Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def search_documents(query: str, vector_database: list[tuple[str, Any]], model : SentenceTransformer, top_n : int=3) -> list[tuple[str, Any]]:
    query_embedding = model.encode(query)
    similarity_scores = cosine_similarity([query_embedding], [embedding for doc, embedding in vector_database])[0]
    ranked_results = sorted(zip(vector_database, similarity_scores), key=lambda x: x[1], reverse=True)
    return [(doc, score) for (doc, _), score in ranked_results[:top_n]]