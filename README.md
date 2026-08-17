# Machine Learning Assignment 2 - Classification Models and Streamlit Deployment

## Problem Statement
The objective of this project is to build an end-to-end machine learning classification workflow. Multiple classification models are trained on the same dataset, evaluated using common classification metrics, and demonstrated through an interactive Streamlit web application.

## Dataset Description
- Dataset: UCI Breast Cancer Wisconsin Diagnostic dataset
- Source used in implementation: `sklearn.datasets.load_breast_cancer`
- Problem type: Binary classification
- Classes: malignant and benign
- Number of instances: 569
- Number of features: 30 numeric features
- Target column: `target`, where 0 = malignant and 1 = benign



## GitHub Repository Link

`https://github.com/UmaAlebale/ML_Assignement2`

## Live Streamlit App Link

`https://mlassignement2.streamlit.app/`

## Models Used
The following classification models were implemented on the same train-test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier
6. Support Vector Machine Classifier

## Evaluation Metrics

| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression      |     0.986  | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.97   |
| Decision Tree            |     0.9371 | 0.9186 |      0.9551 |   0.9444 | 0.9497 | 0.8657 |
| kNN                      |     0.979  | 0.9845 |      0.9677 |   1      | 0.9836 | 0.9555 |
| Naive Bayes              |     0.9371 | 0.9893 |      0.9263 |   0.9778 | 0.9514 | 0.865  |
| Random Forest (Ensemble) |     0.958  | 0.9949 |      0.9565 |   0.9778 | 0.967  | 0.9098 |
| Support Vector Machine   |     0.979  | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong baseline. Standard scaling helps this linear model separate malignant and benign classes effectively. |
| Decision Tree | Good interpretability but slightly lower generalization because a single tree can overfit selected splits. |
| kNN | High performance after scaling. It benefits from similarity among standardized numeric tumor measurements. |
| Naive Bayes | Fast and simple. Performance is good, but independence assumptions can limit it when features are correlated. |
| Random Forest (Ensemble) | Best or near-best overall. It reduces single-tree variance and handles non-linear patterns well. |
| Support Vector Machine | Strong performance due to RBF kernel and scaling. It is competitive for this medium-size numeric dataset. |
| Overall Winner for this dataset | Random Forest and SVM are the strongest candidates. Random Forest is selected as the overall winner because it provides strong performance, robustness, and good handling of non-linear feature interactions. |

## Streamlit App Features
The Streamlit application includes:

- CSV upload option for test data
- Model selection dropdown
- Display of Accuracy, AUC, Precision, Recall, F1, and MCC
- Confusion matrix visualization
- Classification report
- Prediction output table

## Repository Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- breast_cancer_full_dataset.csv
|-- model_metrics.csv
|-- model_reports.json
|-- model/
    |-- train_models.py
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- random_forest_ensemble.joblib
    |-- support_vector_machine.joblib
```

## BITS Virtual Lab Execution Screenshot
Images/BITSLabSnap.png
