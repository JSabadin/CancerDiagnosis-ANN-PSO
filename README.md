# CancerDiagnosis-ANN-PSO

## Project Overview
This repository contains a cancer diagnosis system that leverages an Artificial Neural Network (ANN) optimized by Particle Swarm Optimization (PSO). The project focuses on classifying breast cancer based on a set of features, aiming to assist in early detection and treatment planning.

## Background
Breast cancer is a prevalent disease with significant impact on health. Early diagnosis is crucial for effective treatment. This project applies advanced machine learning techniques to improve diagnostic accuracy.

## Features
- Use of ANN for accurate classification of cancerous conditions.
- Optimization of ANN using PSO for enhanced performance.
- Analysis of various features to determine cancer presence and type.

## Dataset
For this project, we utilize the Diagnostic Breast Cancer (WDBC) dataset. This dataset is publicly available and is commonly used in the machine learning community for classification tasks. It consists of 569 instances with 32 attributes each, including ID, diagnosis, and 30 real-valued input features. These features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image.

The dataset includes measurements such as radius, texture, perimeter, and area of the cell nuclei. Each instance in the dataset is labeled with a diagnosis: M (malignant) or B (benign). This labeling assists in training the ANN to classify breast cancer effectively.

The data was preprocessed to normalize the feature values and split into training and testing sets, ensuring an unbiased evaluation of the model's performance.

Source: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## Technologies Used
- Machine Learning
- Artificial Neural Networks
- Particle Swarm Optimization

## File Description
- [CancerDiagnosis-ANN-PSO Methodology (PDF)](BreastCancerClassification.pdf): A comprehensive document detailing the theoretical background, methodology, and results of the project.
