# Similarity Classification Project

## A. Problem

This project aims to implement several functionalities related to word similarity and classification of synonyms and antonyms using pre-trained word embeddings of Vietnamese words.

1. **Cosine Similarity**: 
   - Implement a function to calculate cosine similarity between word pairs using pre-trained embeddings.
   - Test the implementation with word pairs from the **ViSim-400** dataset located in the `Datasets/ViSim-400` directory.

2. **K-Nearest Words**: 
   - Given a word \( w \), find \( k \) most-similar words to \( w \) using the cosine similarity function implemented in the first point.

3. **Synonym-Antonym Classification**:
   - Implement a classifier (e.g., Logistic Regression, Multi-layer Perceptron) to distinguish between synonyms and antonyms.
   - For training, use the dataset available in the **antonym-synonym set** directory:
     - [Antonym-Synonym Set Dataset](https://github.com/NLP-Projects/Word-Similarity/tree/master/antonymsynonym%20set)
   - For testing, use the **ViCon-400** dataset located in the `Datasets/ViCon-400` directory.
   - Evaluate the experimental results using precision, recall, and F-measure (F1):
     - **Precision**: 
       \[
       \text{Precision} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}
       \]
     - **Recall**: 
       \[
       \text{Recall} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}
       \]
     - **F1 Score**: 
       \[
       F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
       \]

## B. Programming Language and Tool

1. **Programming Language**: Python
2. **Machine Learning Library**: Scikit-learn ([scikit-learn.org](https://scikit-learn.org/))

## C. Data Sets

Data sets are available at: [NLP-Projects Word-Similarity](https://github.com/NLP-Projects/Word-Similarity)

1. **Pre-trained Word2Vec**: 
   - Located in the `Word2vec` directory. For example, the word `sinh_viên` is represented by a 150-dimensional vector:
   ```
   sinh_viên -0.16830535 -0.46649584 ... -0.17137058
   ```

2. **ViSim-400 Dataset**: 
   - Located in the `Datasets/ViSim-400` directory.
   - Line format and examples:
   ```
   Word1 Word2 POS Sim1 Sim2 STD
   biến ngập V 3.13 5.22 0.72
   động tĩnh V 0.6 1.0 0.95
   ```

3. **ViCon-400 Dataset**: 
   - Located in the `Datasets/ViCon-400` directory.
   - Line format and examples:
   ```
   Word1 Word2 Relation
   hời_hợt nông_cạn SYN
   thảnh_thơi ưu_tư ANT
   ```

4. **Reference Documents**: 
   - Located in the `Reference` directory.

## D. Installation and Usage

### Installation

To install the required packages, run the following command:

```bash
pip install -r ./SIMILARITY-CLASSIFICATION/requirements.txt
```

### Running the Project

To execute the main program, run:

```bash
python ./SIMILARITY-CLASSIFICATION/main.py
```

## E. Project Structure

- `./SIMILARITY-CLASSIFICATION/`: Main project directory.
- `./SIMILARITY-CLASSIFICATION/main.py`: Main script to run the project.
- `./SIMILARITY-CLASSIFICATION/word_embeddings.py`: Contains functions for loading word embeddings and calculating cosine similarity.
- `./SIMILARITY-CLASSIFICATION/nearest_words.py`: Contains functions for finding k-nearest words.
- `./SIMILARITY-CLASSIFICATION/classifier.py`: Contains functions for loading datasets and training/testing the classifier.
- `./SIMILARITY-CLASSIFICATION/datasets/`: Directory containing datasets for testing.
- `./SIMILARITY-CLASSIFICATION/antonym-synonym set/`: Directory containing antonym and synonym datasets.
- `./SIMILARITY-CLASSIFICATION/requirements.txt`: File listing the required Python packages.

## F. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.