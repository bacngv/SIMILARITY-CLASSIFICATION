import numpy as np
from numpy.linalg import norm

# Load pre-trained word embeddings
def load_word_embeddings(filepath):
    word_embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            try:
                vector = np.array([float(x) for x in parts[1:]])
                word_embeddings[word] = vector
            except ValueError:
                continue
    return word_embeddings

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Test on ViSim-400 dataset
def test_cosine_similarity(word_embeddings, test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word1, word2 = parts[0], parts[1]
            if word1 in word_embeddings and word2 in word_embeddings:
                vec1 = word_embeddings[word1]
                vec2 = word_embeddings[word2]
                similarity = cosine_similarity(vec1, vec2)
                print(f"Similarity between {word1} and {word2}: {similarity}")
