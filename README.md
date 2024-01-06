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
For this project, we utilize the Diagnostic Breast Cancer (WDBC) dataset. This dataset is publicly available and is widely used in the machine learning community for classification tasks. It includes 569 instances with 32 attributes, such as ID, diagnosis, and 30 real-valued input features derived from digitized images of fine needle aspirates of breast masses.

The dataset features include various measurements like radius, texture, perimeter, and area of the cell nuclei. Each instance is labeled with a diagnosis: M (malignant) or B (benign), which aids in training the ANN for effective breast cancer classification.

Preprocessing involved normalizing the feature values and splitting the data into training and testing sets for unbiased model evaluation.

Source: Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

## Technologies Used
- Machine Learning
- Artificial Neural Networks
- Particle Swarm Optimization

## File Description
- [CancerDiagnosis-ANN-PSO Methodology (PDF)](BreastCancerClassification.pdf): A comprehensive document detailing the theoretical background, methodology, and results of the project.
