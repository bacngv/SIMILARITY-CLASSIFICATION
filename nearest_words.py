import heapq
from word_embeddings import cosine_similarity

def get_k_nearest_words(word, word_embeddings, k=5):
    if word not in word_embeddings:
        return None
    word_vec = word_embeddings[word]
    
    similarities = []
    for other_word, other_vec in word_embeddings.items():
        if other_word != word and word_embeddings[other_word].size == 150:
            sim = cosine_similarity(word_vec, other_vec)
            heapq.heappush(similarities, (-sim, other_word))
    
    # Get top k words
    nearest_words = [heapq.heappop(similarities)[1] for _ in range(k)]
    return nearest_words
