from sentence_transformers import SentenceTransformer
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Max Length:', model.max_seq_length)

sentence = sentences[0]
tokenizer = model.tokenizer
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.encode(sentence)
decoded_sentence = tokenizer.decode(token_ids)

print("Original Sentence:", sentence)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Decoded Sentence:", decoded_sentence)
