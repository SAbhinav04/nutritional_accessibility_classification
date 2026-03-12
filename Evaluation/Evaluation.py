from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
MODELS_DIR = BASE_DIR.parent / "models"
BASE_DIR = Path(__file__).resolve().parent

# Load test labels and model predictions
y_test = pd.read_csv(MODELS_DIR / "y_test.csv").squeeze("columns")
predictions = pd.read_csv(BASE_DIR / "random_forest_predictions.csv").squeeze("columns")

# Keep labels in a fixed order for the report and confusion matrices
labels = sorted(y_test.unique())

# Main summary metrics
metrics_summary = pd.DataFrame(
    [
        ("Accuracy", accuracy_score(y_test, predictions)),
        ("Balanced Accuracy", balanced_accuracy_score(y_test, predictions)),
        ("Macro Precision", precision_score(y_test, predictions, average="macro")),
        ("Macro Recall", recall_score(y_test, predictions, average="macro")),
        ("Macro F1", f1_score(y_test, predictions, average="macro")),
        ("Weighted F1", f1_score(y_test, predictions, average="weighted")),
    ],
    columns=["metric", "score"],
)

# Detailed class-wise report
report_df = pd.DataFrame(
    classification_report(y_test, predictions, output_dict=True)
).transpose().round(4)

# Raw and normalized confusion matrices
cm = confusion_matrix(y_test, predictions, labels=labels)
cm_normalized = confusion_matrix(y_test, predictions, labels=labels, normalize="true")

metrics_summary.to_csv(BASE_DIR / "random_forest_evaluation_metrics.csv", index=False)
report_df.to_csv(BASE_DIR / "random_forest_classification_report.csv")
pd.DataFrame(cm, index=labels, columns=labels).to_csv(
    BASE_DIR / "random_forest_confusion_matrix.csv"
)
pd.DataFrame(cm_normalized, index=labels, columns=labels).round(4).to_csv(
    BASE_DIR / "random_forest_confusion_matrix_normalized.csv"
)

print("Random Forest Evaluation")
print(metrics_summary.to_string(index=False))
print("\nClassification Report:\n")
print(report_df.to_string())

# Plot raw confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE_DIR / "random_forest_confusion_matrix.png")
plt.close()

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE_DIR / "random_forest_confusion_matrix_normalized.png")
plt.close()
