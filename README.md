
### Overview

This repository delves into community detection within the GitHub developer network through binary node classification. The main goal is to discern whether a GitHub user belongs to the web development or machine learning community. The project utilizes Graph Neural Networks (GNN), Naive Bayes, and Logistic Regression models for classification.

### Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
    - [Importing the Dataset](#importing-the-dataset)
    - [Preprocessing](#preprocessing)
        - [Feature Extraction](#feature-extraction)
        - [Feature Frequency Analysis](#feature-frequency-analysis)
    - [Data Encoding](#data-encoding)
        - [One-Hot Encoding](#one-hot-encoding-for-feature-representation)
        - [Creating Data Structures for Encoding](#creating-data-structures-for-encoding)
    - [Constructing a Graph](#constructing-a-graph)

3. [Model Development](#model-development)
    - [Graph Neural Network (GNN)](#gnn-model)
        - [Description of the GNN Model](#description-of-the-gnn-model)
        - [Training the GNN Model](#training-the-gnn-model)
          - [Model Architecture](#model-architecture)
          - [Loss Function](#loss-function)
          - [Training](#training)
          - [Results](#results)
    - [Naive Bayes Model](#naive-bayes)
        - [Description of the Naive Bayes Model](#description-of-the-naive-bayes-model)
        - [Training the Naive Bayes Model](#training-the-naive-bayes-model)
    - [Logistic Regression Model](#logistic-regression)
        - [Description of the Logistic Regression Model](#description-of-the-logistic-regression-model)
        - [Training the Logistic Regression Model](#training-the-logistic-regression-model)

4. [Results and Analysis](#results-and-analysis)
    - [Graph Neural Network Results](#gnn-results)
    - [Naive Bayes Results](#naive-bayes-results)
    - [Logistic Regression Results](#logistic-regression-results)

5. [Comparison of the Three Models](#comparison-of-the-three-models)
6. [Conclusion](#conclusion)
7. [Future Directions and Considerations](#future-directions-and-considerations)
8. [Collaboration](#Collaboration)

## Introduction

In the ever-evolving landscape of software development, understanding the diverse communities and roles within the GitHub ecosystem is crucial. This project aims to perform social network analysis on GitHub developers using a dataset collected from the public API in June 2019. The focus is on binary node classification, distinguishing between web developers and machine learning developers. The tools employed include Graph Neural Networks (GNN), Naive Bayes, and Logistic Regression, with a comprehensive analysis and comparison of their results.

## Dataset

### Importing the Dataset

To initiate the project, the first step is exploring and loading the dataset. The data, collected from GitHub's public API, includes information about developers' locations, starred repositories, employers, and email addresses.

<img width="413" alt="image" src="https://github.com/rakshita003/CSI4900/assets/43514952/56d4e81c-439a-4ce8-98b8-24e033982055">


### Preprocessing

#### Feature Extraction

In this phase, features are extracted from all 37,700 nodes in the dataset. Features are consolidated into lists, laying the foundation for subsequent analysis and model development.

#### Feature Frequency Analysis

A feature frequency analysis is performed to count the occurrence of each feature, identifying patterns and influential features.

### Data Encoding

Data encoding plays a pivotal role in preparing the dataset for machine learning models. One-hot encoding is employed to transform heterogeneous features into a unified format.

### Constructing a Graph

The GitHub developer network is represented as a graph, crucial for modeling relationships and conducting graph-based tasks.

<img width="675" alt="image" src="https://github.com/rakshita003/CSI4900/assets/43514952/78a3ca88-63ae-4fc7-a37f-028a5b48cade">


## Model Development

### Graph Neural Network (GNN)

#### Description of the GNN Model

GNNs, specifically Graph Convolutional Networks (GCNs), are utilized for analyzing graph-structured data. The model is designed to capture relationships within the GitHub developer network. The SocialGNN is designed to predict social interactions in a graph-based setting. It utilizes Graph Convolutional Networks (GCN) to capture the relationships and interactions between entities represented as nodes in the graph. The model is trained to classify the presence or absence of social interactions.

#### Model Architecture

The architecture of the SocialGNN consists of two Graph Convolutional Layers:

1. **Convolutional Layer 1:**
   - Input Features: Number of features associated with each node.
   - Output Features: Configurable parameter `f`.
   - Activation Function: Rectified Linear Unit (ReLU).

2. **Convolutional Layer 2:**
   - Input Features: `f` features from the first layer.
   - Output Features: 2, representing binary classification (presence or absence of social interaction).
   - Activation Function: None (linear).

The model utilizes these layers to capture the complex relationships within the graph and make predictions.

#### Loss Function

The loss function used for training is the Cross-Entropy Loss. Additionally, a masking mechanism is employed to focus the loss computation only on the relevant parts of the training data. This is particularly useful in scenarios where the graph data is partially labeled.

#### Training

The training process involves optimizing the model using the Adam optimizer. A learning rate scheduler is applied to adaptively adjust the learning rate during training, with a step size of 10 epochs and a decay factor of 0.9.

#### Results

Training progress is monitored over epochs, including training and validation losses, as well as accuracies. The model's performance is evaluated on a test set to assess its generalization capabilities. Visualizations of the training and validation metrics are provided for analysis.

### Naive Bayes Model

The Naive Bayes classifier is a probabilistic model based on Bayes' theorem, assuming independence among features. This implementation uses the Gaussian Naive Bayes variant, suitable for continuous data. The dataset is split into training and testing sets using 4-Fold Cross-Validation. This technique helps assess the model's generalization performance by training and evaluating it on different subsets of the data. The Naive Bayes classifier is trained on each fold of the cross-validation. The accuracy of the classifier is computed for each iteration, and confusion matrices are generated to provide insights into its predictive capabilities. The code outputs accuracy scores for each fold, as well as an average accuracy score across all folds. Additionally, average confusion matrices are calculated and visualized to show the distribution of predicted labels compared to true labels. This Naive Bayes classifier implementation with 4-fold cross-validation provides a robust evaluation of the model's performance on the given dataset. The provided code is customizable for different datasets and can be extended for further experimentation.

### Logistic Regression Model

#### Description of the Logistic Regression Model

Logistic Regression is configured for binary classification, demonstrating effectiveness in predicting whether a user is a web developer or a machine learning developer. Logistic Regression is a classification algorithm that predicts the probability of an instance belonging to a particular class. This implementation uses the Logistic Regression model with specified parameters. The dataset is split into training and testing sets using 4-Fold Cross-Validation. This technique helps assess the model's generalization performance by training and evaluating it on different subsets of the data.

#### Classifier Training

The Logistic Regression classifier is trained on each fold of the cross-validation. The accuracy of the classifier is computed for each iteration, and confusion matrices are generated to provide insights into its predictive capabilities. The code outputs accuracy scores for each fold, as well as an average accuracy score across all folds. Additionally, an average confusion matrix is calculated and visualized to show the distribution of predicted labels compared to true labels. This Logistic Regression classifier implementation with 4-fold cross-validation provides a robust evaluation of the model's performance on the given dataset. The provided code is customizable for different datasets and can be extended for further experimentation.

## Results and Analysis

### Graph Neural Network Results

The GNN model achieves an impressive average accuracy of 87.82%. Learning rate adjustments contribute to its ability to capture evolving social dynamics within the GitHub community.

<img width="415" alt="image" src="https://github.com/rakshita003/CSI4900/assets/43514952/050915d4-14c9-41f5-9ef8-ad7ae6326c06">


### Naive Bayes Results

The Naive Bayes model, while simple and efficient, struggles with the dataset's complexity and imbalance, resulting in an accuracy of 44.45%.

<img width="386" alt="image" src="https://github.com/rakshita003/CSI4900/assets/43514952/64460cd8-d604-4c15-bbf6-9b4b218cfdbc">

### Logistic Regression Results

Logistic Regression achieves a notable accuracy of 83.42%, effectively handling imbalanced class distributions.

<img width="389" alt="image" src="https://github.com/rakshita003/CSI4900/assets/43514952/f95ddfc5-2e14-4a87-8e8e-81dbd0115d0d">


## Comparison of the Three Models

The GNN outperforms Naive Bayes and Logistic Regression, showcasing its ability to capture complex relationships in graph-structured data.

<img width="408" alt="image" src="https://github.com/rakshita003/CSI4900/assets/43514952/f229399e-130a-465e-9bd5-de697eb7ddc2">


## Conclusion

The project provides insights into GitHub community dynamics, leveraging GNN, Naive Bayes, and Logistic Regression models. GNN excels in capturing intricate relationships, while Logistic Regression offers a reliable and interpretable solution. 

You can find the [Final Code](https://colab.research.google.com/drive/1-UUDyyiHna9oxSG85Gz0qzFyKgvMaCzg?usp=sharing) here.
 
## Future Directions and Considerations

Future explorations could involve refining GNN architectures, expanding classification scope, exploring alternative models, and incorporating diverse datasets for a broader understanding of community detection in different contexts.

## Collaboration

This project was developed in collaboration with Rakshita Mathur and Meet Mehta. It served as our final year honors project at the University of Ottawa Fall 2023

#### Contributors

- [Rakshita Mathur](https://github.com/rakshita003)
- [Meet Mehta](https://github.com/mehtameet12)

#### Acknowledgments

We would like to express our gratitude to Amiya Nayak, our project supervisor, for their guidance and support throughout the development of this project.
