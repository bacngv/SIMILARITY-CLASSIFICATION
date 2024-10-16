import os
import gdown
from word_embeddings import load_word_embeddings, test_cosine_similarity
from nearest_words import get_k_nearest_words
from classifier import load_antonym_synonym_data, train_classifier, test_classifier_on_vicon

def main():
    # Create the directory if it doesn't exist
    os.makedirs('./SIMILARITY-CLASSIFICATION', exist_ok=True)
    
    # Download W2V_150.txt from Google Drive
    gdown.download('https://drive.google.com/uc?id=1rumN4auP1vAc6e3pARZ76vCTYrUwPSkF', 
                   './SIMILARITY-CLASSIFICATION/W2V_150.txt', quiet=False)
    
    # Load word embeddings
    word_embeddings = load_word_embeddings('./SIMILARITY-CLASSIFICATION/W2V_150.txt')
    
    # Test cosine similarity with ViSim-400 dataset
    test_cosine_similarity(word_embeddings, './SIMILARITY-CLASSIFICATION/datasets/ViSim-400/Visim-400.txt')

    # Example usage of k-nearest words
    word = 'tôi'
    nearest_words = get_k_nearest_words(word, word_embeddings, k=5)
    print(f"Top 5 words similar to {word}: {nearest_words}")

    # Load the dataset and train the classifier
    X, y = load_antonym_synonym_data('./SIMILARITY-CLASSIFICATION/antonym-synonym set/Antonym_vietnamese.txt', 
                                       './SIMILARITY-CLASSIFICATION/antonym-synonym set/Synonym_vietnamese.txt', 
                                       word_embeddings)
    classifier = train_classifier(X, y)

    # Test on ViCon-400 dataset
    test_classifier_on_vicon(classifier, './SIMILARITY-CLASSIFICATION/datasets/ViCon-400', word_embeddings)

if __name__ == "__main__":
    main()