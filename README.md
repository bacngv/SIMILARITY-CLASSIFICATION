# Similarity Classification Project

## Overview
This project focuses on calculating the cosine similarity between Vietnamese word pairs, finding k-nearest words, and classifying synonyms and antonyms. The project implements machine learning techniques using pre-trained word embeddings.

## A. Problem

1. **Cosine Similarity**: 
   - Given pre-trained embeddings of Vietnamese words, implement a function for calculating cosine similarity between word pairs. 
   - Test your program using word pairs in the **ViSim-400** dataset (located in `Datasets/ViSim-400`).

2. **K-Nearest Words**: 
   - Given a word \( w \), find \( k \) most-similar words of \( w \) using the cosine similarity function implemented in question 1.

3. **Synonym-Antonym Classification**: 
   - Implement a classifier (Logistic Regression, Multi-layer Perceptron, etc.) for distinguishing synonyms and antonyms. 
   - For training, use the dataset available in the directory `antonym-synonym set`: [Antonym-Synonym Set](https://github.com/NLP-Projects/Word-Similarity/tree/master/antonymsynonym%20set).
   - For testing, use the **ViCon-400** dataset (located in `Datasets/ViCon-400`).
   - Experimental results are evaluated by precision scores, recall scores, and F-measure (F1).

## B. Programming Language and Tools

1. **Programming Language**: Python
2. **Machine Learning Library**: [Scikit-learn](https://scikit-learn.org/)

## C. Data Sets

1. **Pre-trained Word2Vec**: Located in the directory `Word2vec`.
   - Example representation of the word “sinh_viên” as a 150-dimension vector:
   ```
   sinh_viên -0.16830535 -0.46649584 -0.09095726 0.26220384 ...
   ```

2. **ViSim-400 Dataset**: Located in `Datasets/ViSim-400`.
   - Line format and examples (Sim1 represents human rating of similarity in the interval [0,4], Sim2 represents human rating of similarity in the interval [0,6]):
   ```
   Word1 Word2 POS Sim1 Sim2 STD
   biến ngập V 3.13 5.22 0.72
   động tĩnh V 0.6 1.0 0.95
   ...
   ```

3. **VCon-400 Dataset**: Located in `Datasets/ViCon-400`.
   - Line format and examples:
   ```
   Word1 Word2 Relation
   hời_hợt nông_cạn SYN
   thảnh_thơi ưu_tư ANT
   ...
   ```

4. **Reference Documents**: Located in the `Reference` directory.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/bacngv/SIMILARITY-CLASSIFICATION
   ```

2. Install the required packages:
   ```bash
   pip install -r ./SIMILARITY-CLASSIFICATION/requirements.txt
   ```

3. Run the main program:
   ```bash
   python ./SIMILARITY-CLASSIFICATION/main.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.