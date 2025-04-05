from sentence_transformers import SentenceTransformer
from typing import Any
import numpy as np
from helper.csv_extractor_version1 import extract_column_to_list
from helper.vector_functions import search_documents

def print_result(results : list[tuple[str, Any]], query):
    print(f"\nTop relevant documents for query: '{query}'")
    for doc, score in results:
        print(f"- '{doc}' (Score: {score:.4f})")


if __name__ == "__main__":
    print("-----------------------------------------------------------------")
    print("Simple Vector Database")
    print("Using sentence-transformer and Sklearn-cosine_similarity")
    print("-----------------------------------------------------------------")
    print("Loading....")

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Model Loaded.")
    
    file_path = 'files/docs.csv'
    target_column = 'text_data'
    print("Loading Data.")
    documents = extract_column_to_list(file_path, target_column)

    document_embeddings = model.encode(documents)
    vector_database = list(zip(documents, document_embeddings))
    print(f"Data Loaded. Document Shape:  {document_embeddings.shape}")
    
    print("-----------------------------------------------------------------\n\n")

    while True:
        print("-----------------------------------------------------------------")
        print("Enter Text (type 'done' to finish):")
        line = input("> ")
        if line.lower() == 'done':
            break
        
        query = line
        results = search_documents(query, vector_database, model)
        print_result(results, query)
        print("\n")
