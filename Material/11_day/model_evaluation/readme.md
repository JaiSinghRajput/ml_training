# confusion matrix
- The confusion matrix evaluates classification models by comparing predicted and true labels.
- It consists of four components: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- Useful for multi-class problems, it highlights class-wise performance and areas for improvement.
- Key metrics derived include accuracy, precision, recall, F1 score, and more.
- Example matrix:
    | Actual\Predicted | Positive | Negative |
    |-------------------|----------|----------|
    | Positive          | TP       | FN       |
    | Negative          | FP       | TN       |
- Common formulas:
    - Accuracy = (TP + TN) / Total
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    - Specificity = TN / (TN + FP)
    - MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
- Confusion matrices provide a detailed view of model performance and guide improvements.

# Roc curve
- The ROC curve visualizes a model's performance across different thresholds, plotting True Positive Rate (TPR) against False Positive Rate (FPR).
