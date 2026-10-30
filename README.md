# NLP Course Assignments

This repository contains five assignments completed as part of the **Natural Language Processing** course. The assignments cover Persian text clustering, neural network hyperparameter tuning, Naive Bayes classification, Word Embeddings, Recurrent Neural Networks (RNN/LSTM), and Convolutional Neural Networks (CNN) for text classification.
Each assignment is documented separately, and its corresponding implementation files and experimental results are uploaded using the assignment number.

---

## Assignment 1 – Persian Text Clustering Using K-Means++

### Overview
The first assignment focuses on clustering Persian text using the **K-Means++ algorithm** and evaluating clustering performance under different parameter configurations.

Two Persian text datasets are used:
- **Persian-Text-Sentiment:** A dataset containing positive and negative comments.
- **persian_news_dataset:** A dataset containing news articles from six different categories.

### Objectives
- Implement the K-Means++ clustering algorithm.
- Convert Persian text into numerical feature vectors using TF-IDF.
- Investigate the effect of different `min_df` and `item_count` values on clustering performance.
- Evaluate clustering quality using the Calinski–Harabasz Index.
- Explore the impact of Persian language preprocessing and Part-of-Speech filtering on clustering performance.

### Tasks

**Part A – Sentiment Dataset Clustering**

Apply K-Means++ to the Persian-Text-Sentiment dataset using two clusters (`k=2`), representing positive and negative comments.

Perform six experiments using:
- Three different `min_df` values selected from 2, 3, 5, 10, 50, and 100.
- Two different `item_count` values selected from 100, 200, 500, and 1000.

Use TF-IDF for text vectorization and evaluate each experiment using the Calinski–Harabasz Index.

**Part B – News Dataset Clustering**

Repeat the same experimental procedure on the `persian_news_dataset`, using six clusters (`k=6`), with each cluster representing a news category.

Compare the results with Part A and discuss the differences in clustering performance.

**Part C – POS-Based Text Filtering**

Use the Persian language preprocessing tool introduced in Part 3 of the course to filter words based on their POS tags before clustering.

Repeat the clustering process on both datasets using the best parameter configurations obtained in Parts A and B.

### Results and Documentation
A PDF file containing the experimental results, explanations, comparisons, and interpretations has been uploaded.

---

## Assignment 2 – Neural Network Parameter Evaluation for Spam Text Detection

### Overview
The second assignment focuses on investigating the initial parameters and architecture of a neural network for **spam text detection**.

### Objectives
- Investigate the effect of neural network architecture on classification performance.
- Understand the impact of the number and size of hidden layers.
- Evaluate different vocabulary sizes.
- Compare different activation functions.
- Identify the best combination of hyperparameters.

### Tasks

Each parameter is evaluated independently while keeping the other parameters unchanged.

**Part A – Number of Hidden Layers**

Test at least three configurations selected from:
- 0 hidden layers
- 1 hidden layer
- 2 hidden layers
- 3 hidden layers
- 4 hidden layers
- 5 hidden layers

**Part B – Hidden Layer Size**

Evaluate at least three different hidden layer sizes within the range of 10 to 1024 neurons.

**Part C – Vocabulary Size**

Test at least three different vocabulary configurations by changing the vocabulary size from 20% to 80%.

**Part D – Activation Functions**

Compare at least three activation functions in the hidden layers:
- ReLU
- Sigmoid
- Linear

### Final Experiment – Best Parameter Combination

Select the best-performing configuration for each parameter, combine them into a single neural network model, and evaluate its performance.

### Evaluation and Submission
- Present the experimental results in tables and, if necessary, charts.
- Compare and analyze the different configurations.
- Perform all experiments using Google Colab.
- Upload the executed notebook in `.ipynb` format.
- Upload an analysis report of no more than three pages.

---

## Assignment 3 – Naive Bayes Text Classification

### Overview
The third assignment focuses on implementing and evaluating two Naive Bayes classification approaches: **standard Naive Bayes and Gaussian Naive Bayes**.

The implementations are tested on three text datasets:
- **IMDB:** English movie reviews.
- **Persian-Text-Sentiment:** Persian positive and negative comments.
- **persian_news_dataset:** Persian news articles.

### Objectives
- Understand and implement Naive Bayes classification.
- Compare standard Naive Bayes with Gaussian Naive Bayes.
- Evaluate classification performance across English and Persian datasets.
- Analyze the strengths and weaknesses of both approaches.

### Tasks

**Part A – Standard Naive Bayes**

Implement the standard Naive Bayes classifier and evaluate it on all three datasets.

**Part B – Gaussian Naive Bayes**

Implement Gaussian Naive Bayes and perform the same experiments on all three datasets.

### Evaluation Metrics
For both classification approaches and all three datasets, calculate and report:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Analysis
Compare the performance of standard Naive Bayes and Gaussian Naive Bayes across the three datasets and discuss the differences between English and Persian text classification results.

---

## Assignment 4 – Neural Networks with Word Embeddings, RNN, and LSTM

### Overview
The fourth assignment extends the neural network implementation from Assignment 2 by incorporating **Word Embeddings and recurrent neural network layers (SimpleRNN and LSTM)** before the hidden layers.

The experiments are performed on three datasets:
- **IMDB:** English movie reviews.
- **Persian-Text-Sentiment:** Persian positive and negative comments.
- **persian_news_dataset:** Persian news articles.

### Objectives
- Understand how Word Embeddings represent words as dense numerical vectors.
- Investigate the impact of embedding dimensions on classification performance.
- Compare randomly initialized embeddings with pre-trained embeddings.
- Explore the effect of SimpleRNN and LSTM layers on text classification.

### Tasks

**Part A – Trainable Word Embeddings**

Add an Embedding layer to the neural network with `input_length=50`.

```python
model.add(Embedding(max_words, embedding_dim, input_length=50))
```

Evaluate at least three different embedding dimensions selected from:
- 8
- 10
- 16
- 32
- 64
- 100

**Part B – Pre-trained Word Embeddings**

Use pre-trained Word Embeddings such as:
- GloVe
- Word2Vec

Load the pre-trained embedding matrix into the Embedding layer and freeze its weights:

```python
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
```

**Part C – SimpleRNN Layers**

Add two SimpleRNN layers with 16 and 32 units to the best-performing configuration from Part A.

Example:

```python
model.add(SimpleRNN(32))
```

Evaluate the resulting model and compare its performance with the previous configurations.

**Part D – LSTM Layers**

Add LSTM layers with 32 and 64 units to the best-performing configuration from Part B.

Example:

```python
model.add(LSTM(32))
```

Evaluate the resulting model and analyze the impact of LSTM layers on classification performance.

### Evaluation Metrics
Calculate and report the following metrics for all experiments:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Analysis
Compare the results across the three datasets and investigate the effects of trainable embeddings, pre-trained embeddings, SimpleRNN, and LSTM architectures.

---

## Assignment 5 – Text Classification Using CNN and LSTM-CNN

### Overview
The fifth assignment extends Assignment 4 by incorporating **One-Dimensional Convolutional Neural Networks (1D CNN)** into the text classification architecture.

The objective is to investigate how CNN layers can extract local patterns and features from text sequences and how combining CNN with LSTM affects classification performance.

### Objectives
- Understand the application of 1D CNN layers in text classification.
- Investigate the effect of filter sizes and kernel sizes.
- Evaluate the impact of MaxPooling1D.
- Compare CNN-based architectures with recurrent neural network architectures.
- Explore hybrid CNN-LSTM models.

### Tasks

**Part A – Adding CNN Layers**

Similar to Part C of Assignment 4, replace the SimpleRNN layers with 1D CNN layers.

Evaluate eight possible configurations by varying the following parameters:

- Number of CNN layers: one or two.
- Number of output filters: 16 or 32.
- Kernel size: 5 or 7.
- MaxPooling1D: enabled or disabled.

The experiments should follow the CNN architecture described in Listing 6.46 of the course textbook.

Compare the eight configurations and identify the best-performing CNN architecture.

**Part B – Combining CNN and LSTM**

Take the best-performing CNN configuration from Part A and add it to the LSTM architecture developed in Part D of Assignment 4.

Evaluate the resulting hybrid CNN-LSTM model and compare its performance with the previous CNN and LSTM models.

### Evaluation Metrics
Calculate and report the following metrics for all experiments:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Analysis
Compare the performance of different CNN configurations and the hybrid CNN-LSTM model. Discuss how convolutional and recurrent layers contribute to text classification.

---

## General Repository Structure

The repository is organized by assignment number. Each assignment contains its corresponding implementation, notebooks and reports.

```text
NLP-Course-Assignments/
│
├── README.md
│
├── Assignment-1/
│   ├── Assignment-1.ipynb
│   └── reports.pdf
├── Assignment-2/
│   ├── Assignment-2.ipynb
│   └── reports.pdf
│
├── Assignment-3/
│  ├── Assignment-3.ipynb
│  └── reports.pdf
│
├── Assignment-4/
│   ├── A.ipynb
│   └── B.ipynb
|   └── C.ipynb
│   └── D.ipynb
|   └── reports.pdf
|
└── Assignment-5/
|    ├── A.ipynb
│    └── B.ipynb
|    └── reports.pdf
```
---

## Overall Learning Outcomes

These five assignments provide practical experience with fundamental NLP and machine learning techniques, including:

- Unsupervised text clustering using K-Means++.
- Text representation using TF-IDF and Word Embeddings.
- Traditional machine learning classification using Naive Bayes.
- Neural network architecture design and hyperparameter tuning.
- Recurrent neural networks, including SimpleRNN and LSTM.
- Convolutional neural networks for text classification.
- Hybrid CNN-LSTM architectures.
- Model evaluation using classification metrics and confusion matrices.

The assignments collectively explore different approaches to English and Persian text processing, clustering, and classification.
