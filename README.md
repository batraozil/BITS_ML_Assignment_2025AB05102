# HAR Multi-Model Classification (Streamlit Deployment)

## a. Problem statement
Build and compare multiple ML classification models to recognize human activities from smartphone sensor features, and deploy an interactive Streamlit app.

## b. Dataset description
Dataset: UCI Human Activity Recognition Using Smartphones  
- Instances: 10,299  
- Features: 561 numeric features  
- Classes: 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)  
Source: UCI Machine Learning Repository

## c. Models used (with metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9854 | 0.9994 | 0.9855 | 0.9854 | 0.9854 | 0.9825 |
| Decision Tree | 0.9369 | 0.9622 | 0.9373 | 0.9369 | 0.937 | 0.9241 |
| kNN | 0.9626 | 0.998 | 0.964 | 0.9626 | 0.9625 | 0.9554 |
| Naive Bayes (Gaussian) | 0.7243 | 0.9614 | 0.7898 | 0.7243 | 0.7125 | 0.6877 |
| Random Forest (Ensemble) | 0.9816 | 0.9996 | 0.9817 | 0.9816 | 0.9816 | 0.9778 |
| XGBoost (Ensemble) | 0.9942 | 0.9999 | 0.9942 | 0.9942 | 0.9942 | 0.993 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong baseline with near top performance, indicating the dataset is highly separable. Performs competitively with ensemble models while remaining simple and interpretable. |
| Decision Tree | Noticeably weaker than other models, especially ensembles. Likely suffers from overfitting and limited generalization. |
| kNN | Solid performance with high AUC, but slightly lower accuracy and MCC than ensembles. Sensitive to feature scaling and computationally expensive at inference time. |
| Naive Bayes (Gaussian) | Poor overall performance. The Gaussian independence assumption does not hold well for HAR sensor data. |
| Random Forest (Ensemble) | Strong and stable performance across all metrics. Offers a good balance between robustness and accuracy, though marginally below XGBoost. |
| XGBoost (Ensemble) | Best-performing model across all metrics. Captures complex feature interactions effectively, but at the cost of higher computational complexity and tuning effort. |
