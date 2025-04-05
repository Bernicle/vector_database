from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
print(f"Model's maximum sequence length:",end='')
print(SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length)