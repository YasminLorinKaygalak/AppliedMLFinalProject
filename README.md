# Fake News Detection with Machine Learning

This project presents a comprehensive machine learning pipeline for detecting fake news articles using Natural Language Processing (NLP) and classical classification algorithms. Built as a final project for *CSC 4505: Applied Machine Learning* at Villanova University, the system compares a wide range of models using both frequency- and embedding-based text representations.

## 🔍 Project Overview

- **Objective**: Classify news articles as real or fake using NLP-based feature extraction and machine learning.
- **Dataset**: [ISOT Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), containing 44,919 labeled news articles with metadata (title, content, subject, date).
- **Models Tested**:
  - Logistic Regression
  - Naïve Bayes
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Random Forest Classifier 🌟 (Best performing)
  - Gradient Boost Classifier
  - Word2Vec + Logistic Regression
  - Simple RNN
  - (Planned) LSTM

## 🛠️ Features

- Preprocessing: Tokenization, stopword removal, TF-IDF transformation, Word2Vec embedding.
- Model Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- Ensemble Learning: Demonstrated the effectiveness of bagging (Random Forest) and boosting (Gradient Boost).
- Visualization: Confusion matrices and distribution insights for data exploration.

## 📈 Results

| Model                  | Accuracy | F1-Score |
|-----------------------|----------|----------|
| Random Forest          | 99.8%    | 0.998    |
| Gradient Boosting      | 99.6%    | 0.996    |
| SVM (Linear Kernel)    | 99.5%    | 0.995    |
| MLP Classifier         | 99.1%    | 0.991    |
| Logistic Regression    | 98.9%    | 0.988    |
| Naïve Bayes            | 93.3%    | 0.929    |

## 🧠 Future Work

- Implement and evaluate LSTM and transformer-based models (e.g., BERT).
- Perform cross-validation and hyperparameter optimization (e.g., GridSearchCV).
- Explore larger and more diverse datasets to test model generalization.

## 👥 Authors

- **Yasmin Lorin Kaygalak** – Feature engineering, model implementation, TF-IDF & Word2Vec pipelines, reporting.
- **Natalie Bohmbach** – Data cleaning, neural network implementation (MLP, RNN), exploratory analysis, presentation design.
- **Dr. Justin DeBenedetto** – Project advisor and course instructor.

## 📝 License

This project is for academic and research purposes. For external use, please cite accordingly or contact the authors.



