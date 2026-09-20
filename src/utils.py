import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_pred):
    """Computes TP, FP, FN, TN, Precision, Recall/Sensitivity, Specificity, F1, and Balanced Accuracy."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": precision,
        "Recall_Sensitivity": recall,
        "Specificity": specificity,
        "F1_Score": f1,
        "Balanced_Accuracy": balanced_acc,
    }


def balanced_accuracy_from_probs(y_true, y_prob, threshold=0.5):
    """Calculates balanced accuracy from continuous probability predictions."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return compute_metrics(y_true, y_pred)["Balanced_Accuracy"]


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plots and saves a 2x2 confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HS", "NHS"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HS", "NHS"])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_bar(benchmark_keys, model_scores, title, ylabel, save_path):
    """Plots a grouped bar chart comparing performance across benchmarks."""
    n_models = len(model_scores)
    x = np.arange(len(benchmark_keys))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, scores) in enumerate(model_scores.items()):
        ax.bar(x + i * width, scores, width, label=name)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(benchmark_keys)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_metrics_table_csv(rows, path):
    """Saves a list of dictionary metrics to a CSV file."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")
