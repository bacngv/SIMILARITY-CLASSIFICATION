import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Load antonym and synonym data from separate files
def load_antonym_synonym_data(antonym_file, synonym_file, word_embeddings):
    X, y = [], []
    
    # Load antonym data
    with open(antonym_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            try:
                word1, word2 = parts[0], parts[1]
                if word1 in word_embeddings and word2 in word_embeddings:
                    vec1 = word_embeddings[word1]
                    vec2 = word_embeddings[word2]
                    feature = np.concatenate((vec1, vec2))
                    X.append(feature)
                    y.append(0)  # ANT = 0
            except IndexError:
                print(f"Skipping line in antonym file due to insufficient parts: {line.strip()}")

    # Load synonym data
    with open(synonym_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            try:
                word1, word2 = parts[0], parts[1]
                if word1 in word_embeddings and word2 in word_embeddings:
                    vec1 = word_embeddings[word1]
                    vec2 = word_embeddings[word2]
                    feature = np.concatenate((vec1, vec2))
                    X.append(feature)
                    y.append(1)  # SYN = 1
            except IndexError:
                continue

    return np.array(X), np.array(y)

# Train synonym-antonym classifier
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate precision, recall, and F1
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    
    return clf

# Test classifier on ViCon-400 dataset
def test_classifier_on_vicon(clf, test_dir, word_embeddings):
    X_test, y_test = [], []
    
    # List of files in the test directory
    test_files = ['400_noun_pairs.txt', '400_verb_pairs.txt', '600_adj_pairs.txt']
    
    for test_file in test_files:
        file_path = os.path.join(test_dir, test_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    word1, word2, relation = parts[0], parts[1], parts[2]
                    if word1 in word_embeddings and word2 in word_embeddings:
                        vec1 = word_embeddings[word1]
                        vec2 = word_embeddings[word2]
                        feature = np.concatenate((vec1, vec2))
                        X_test.append(feature)
                        y_test.append(1 if relation == 'SYN' else 0)
                except IndexError:
                    print(f"Skipping line in test file {test_file} due to insufficient parts: {line.strip()}")
    
    # Convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate precision, recall, and F1
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'ViCon-400 Test - Precision: {precision}')
    print(f'ViCon-400 Test - Recall: {recall}')
    print(f'ViCon-400 Test - F1 Score: {f1}')
