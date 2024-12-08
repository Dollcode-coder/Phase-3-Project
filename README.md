# Predict Customer Retention at Syriatel Mobile Telecom Using Machine Learning
## Business overview

SyriaTel, a leading telecommunications provider, faces significant challenges in retaining customers. The goal of this project is to predict customer churn using machine learning techniques and provide actionable insights to reduce churn rates. By identifying customers likely to leave, SyriaTel can proactively implement targeted retention strategies, minimizing revenue loss and enhancing customer loyalty.

### Stakeholder
1. SyriaTel Mobile Telecom

### Problem statement
 
 Customer Churn: SyriaTel customer churn is very unpredictable; therefore, the business struggles to anticipate which customers are likely to stop using its services and why. This unpredictability results in reactive strategies, where the company focuses on damage control after customers have already decided to leave, rather than proactively preventing churn.
 ### Success Criteria
 
 Model Performance:
High accuracy to ensure high reliability in identifying customers likely to churn.

Actionable Insights:
Provide interpretable patterns or features that indicate customer churn, allowing SyriaTel to implement effective intervention strategies.

### Constraints

Data Availability and Quality:
The prediction model depends on access to clean, up-to-date, and comprehensive customer data, including demographics, usage patterns, and complaints.

Regulatory Compliance:
Ensure data handling complies with telecommunications regulations and privacy laws.

## Data Understanding

This dataset was sourced from kaggle and it has 3333 rows and 21 columns.The dataset has been used by other data scientists before and is accessible for anyone to look through it, it is updated often. The data is in csv format inside a folder named data. I then read through the data using the panda's library in order to get a data frame as our output . The dataset has data recorded in different data type including float, intergers and objects.

## Data Analysis

We will use univariate, bivariate, and multivariate analysis to perform a thorough investigation of the data in this section.

Finding potential correlations between the features and variable distribution is the goal of this kind of data exploration, which will be crucial for feature engineering and modelling. Features that have a high correlation with the target oare often good for building basline models.

## Model Performance

### 1.Logistic Model ( Baseline model)
Based on the results you've provided for the Logistic Regression model, here's an interpretation:

#### **Accuracy**:
- The model achieves an **accuracy of 0.90**, meaning it correctly classifies 90% of the total instances in the test set. This indicates good overall performance.

#### **Confusion Matrix**:
- **True Negatives (TN)**: 480 instances where the model correctly predicted the negative class (not churned).
- **False Positives (FP)**: 10 instances where the model incorrectly predicted the positive class (predicted churned when not churned).
- **False Negatives (FN)**: 48 instances where the model incorrectly predicted the negative class (predicted not churned when it actually churned).
- **True Positives (TP)**: 22 instances where the model correctly predicted the positive class (churned).

#### **Precision, Recall, F1-Score**:
##### Class 0 (Not Churned):
- **Precision**: 0.91 – The model is very precise in predicting class 0, meaning 91% of the predictions for class 0 are correct.
- **Recall**: 0.98 – The model has a high recall for class 0, meaning it correctly identifies 98% of the actual class 0 instances.
- **F1-Score**: 0.94 – The balance between precision and recall for class 0 is excellent.

##### Class 1 (Churned):
- **Precision**: 0.69 – The model correctly predicts churned cases 69% of the time, which is lower than for class 0.
- **Recall**: 0.31 – The model only correctly identifies 31% of the actual churned instances, indicating it misses a large portion of churned cases.
- **F1-Score**: 0.43 – The F1-score is lower for class 1, indicating a poor balance between precision and recall for the positive class.

#### **Macro and Weighted Averages**:
- **Macro Average**: The macro average precision (0.80), recall (0.65), and F1-score (0.69) give an overall sense of the model's performance on both classes. The macro average treats both classes equally, and the lower recall for class 1 is pulling the average down.
- **Weighted Average**: The weighted average metrics take into account the imbalance in the dataset (more class 0 instances). The weighted averages show high performance overall (precision: 0.88, recall: 0.90, F1: 0.88), reflecting the model's good performance on the majority class.

#### **Interpretation**:
- The model performs **very well for class 0 (not churned)** with high accuracy, precision, and recall.
- **Class 1 (churned)** has lower precision and recall, suggesting that the model struggles to correctly identify churned cases, which is typical in imbalanced datasets.
- If identifying churn (class 1) is critical (e.g., in a business context where predicting churn is important), you might consider improving the model's recall for class 1, possibly through techniques like oversampling, undersampling, or using different algorithms.

### 2.Tuned Model

The results for the trained logistic regression model after applying SMOTE are as follows:

#### Model Accuracy:
- **Accuracy**: 80.71% — The model correctly predicts 80.71% of all samples in the test set.

#### Classification Report:

1. **Class 0 (Not Churned)**:
   - **Precision**: 0.95 — When the model predicts class 0, it is correct 95% of the time.
   - **Recall**: 0.82 — 82% of the actual class 0 instances are correctly identified by the model.
   - **F1-Score**: 0.88 — This is the harmonic mean of precision and recall for class 0, showing a balanced performance.

2. **Class 1 (Churned)**:
   - **Precision**: 0.36 — When the model predicts class 1 (churned), it is correct only 36% of the time. This is low, suggesting that the model tends to predict the negative class (not churned) even when churned instances are present.
   - **Recall**: 0.70 — The model identifies 70% of the actual churned instances correctly.
   - **F1-Score**: 0.48 — The F1-score is relatively low, indicating that the model's performance on class 1 (churned) could be improved, especially in terms of precision.

#### Macro and Weighted Averages:
- **Macro Average**:
  - **Precision**: 0.66 — Averaging precision across both classes.
  - **Recall**: 0.76 — Averaging recall across both classes.
  - **F1-Score**: 0.68 — Averaging F1-scores for both classes.
  
- **Weighted Average**:
  - **Precision**: 0.88 — Weighted precision across the two classes, with more weight on class 0 (which has more samples).
  - **Recall**: 0.81 — Weighted recall across the two classes.
  - **F1-Score**: 0.83 — Weighted F1-score across both classes.

#### Confusion Matrix:
- **True Negatives (TN)**: 403 — The model correctly predicted 403 instances as not churned.
- **False Positives (FP)**: 87 — The model incorrectly predicted 87 instances as churned, although they were not.
- **False Negatives (FN)**: 21 — The model incorrectly predicted 21 instances as not churned, although they were churned.
- **True Positives (TP)**: 49 — The model correctly predicted 49 instances as churned.

#### Interpretation:
- **Class Imbalance**: Despite using SMOTE to oversample the minority class, the model still has difficulty predicting the positive class (churned). The low precision (0.36) for class 1 suggests the model frequently predicts class 0 instead of churned instances.
- **Impact of SMOTE**: SMOTE seems to have improved recall for class 1 (churned), but the model's ability to predict churned samples accurately (precision) is still low. This indicates that although more churned samples are present in the training data after resampling, the model is not distinguishing them well.
- **Next Steps**: Consider experimenting with hyperparameter tuning or trying other techniques like undersampling the majority class or using a different model (e.g., Random Forest or XGBoost) to improve performance on the minority class.

## 3.K-Nearest Neighbors

The results for the K-Nearest Neighbors (KNN) model are as follows:

#### Model Accuracy:
Accuracy: 89.11% — The model correctly predicts 89.11% of all samples in the test set.

#### Classification Report:
Class 0 (Not Churned):

Precision: 0.90 — When the model predicts class 0 (not churned), it is correct 90% of the time.
Recall: 0.99 — The model correctly identifies 99% of the actual non-churned instances.
F1-Score: 0.94 — The model performs very well in predicting the majority class, with a high F1-score indicating balanced precision and recall.
Class 1 (Churned):

Precision: 0.71 — When the model predicts class 1 (churned), it is correct 71% of the time.
Recall: 0.21 — The model only identifies 21% of the actual churned instances correctly, which is low.
F1-Score: 0.33 — The F1-score for class 1 is relatively low, suggesting the model struggles to identify churned instances.

#### Macro and Weighted Averages:
Macro Average:
Precision: 0.81 — Averaging precision across both classes.
Recall: 0.60 — Averaging recall across both classes.
F1-Score: 0.64 — Averaging F1-scores across both classes.
Weighted Average:
Precision: 0.88 — Weighted precision across the two classes, considering the higher number of instances for class 0.
Recall: 0.89 — Weighted recall across both classes.
F1-Score: 0.86 — Weighted F1-score across both classes.

#### Confusion Matrix:
True Negatives (TN): 484 — The model correctly predicted 484 instances as not churned.
False Positives (FP): 6 — The model incorrectly predicted 6 instances as churned, although they were not.
False Negatives (FN): 55 — The model incorrectly predicted 55 instances as not churned, although they were churned.
True Positives (TP): 15 — The model correctly predicted 15 instances as churned.

#### Interpretation:
Class Imbalance: The model is performing well for the majority class (not churned) but has difficulty predicting the minority class (churned). This is reflected in the low recall for class 1, where the model only captures 21% of the churned instances.
Precision for Churned Class: The precision for class 1 (churned) is higher than its recall, meaning that when the model predicts churn, it is relatively accurate, but it is not predicting enough churned instances.
Potential Improvement: Similar to the logistic regression model, the KNN model struggles with the minority class despite being overall accurate. Applying techniques like oversampling (e.g., SMOTE) or using a different classifier, such as Random Forest or XGBoost, might help to improve recall for churned customers.
Conclusion:
While the KNN model has a high overall accuracy, it shows significant weakness in detecting churned customers. There is a need for further adjustments or different models to improve the prediction of the minority class (churned).


## Comparison of the ROC curves

Here's the interpretation:

#### Models Compared:
Logistic Regression (Red curve, AUC = 0.81): The baseline logistic regression model.

Logistic Regression Variant (Blue curve, AUC = 0.81): A slightly modified version of logistic regression with similar performance.

Random Forest (Green curve, AUC = 0.86): The best-performing model in this comparison, with the highest area under the curve (AUC).

Random Guess (Dashed grey line, AUC = 0.5): Represents a random classifier, indicating no discriminatory ability.

#### Key Metric (AUC):
The Area Under the Curve (AUC) quantifies the overall ability of a model to distinguish between classes.

Higher AUC values indicate better model performance.

In this case, the Random Forest model (AUC = 0.86) outperforms the logistic regression models (both with AUC = 0.81), while the random guess (AUC = 0.5) serves as the baseline.


## Conclusion
The Random Forest model provides the best classification performance among the evaluated models, as it has the highest AUC and its ROC curve is closest to the top-left corner.

### Recommendations

1)Targeted Customer Retention Strategies: Identify customers who are likely to churn and offer them personalized deals or services to retain them.

2)Resource Allocation: Focus retention efforts on customers who are at risk of leaving, improving efficiency in marketing and customer support.

3)Revenue Growth: By reducing churn, Syriatel can increase customer lifetime value (CLV) and stabilize revenue growth.

