"""Metrics, plotting, CSV output and latency helpers (no TensorFlow dependency)."""
import csv
import time

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_pred):
    """HS = positive class. Returns TP/FP/FN/TN, precision, recall, specificity, F1, balanced accuracy."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": precision, "Recall_Sensitivity": recall,
            "Specificity": specificity, "F1_Score": f1,
            "Balanced_Accuracy": (recall + specificity) / 2}


def balanced_accuracy_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return compute_metrics(y_true, y_pred)["Balanced_Accuracy"]


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Rows = ACTUAL class, columns = PREDICTED class (fixes the swapped axis labels of the first version)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])          # [[TP, FN], [FP, TN]]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HS", "NHS"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["HS", "NHS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_model_comparison_bar(benchmark_keys, model_scores, title, ylabel, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_scores)
    x = np.arange(len(benchmark_keys))
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, scores) in enumerate(model_scores.items()):
        ax.bar(x + i * width, scores, width, label=name)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(benchmark_keys)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def save_metrics_table_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows:                                   # tolerate rows with extra keys
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def measure_latency_ms(fn, X, n=200, warmup=10, seed=0):
    """Per-clip (batch size 1) wall-clock latency of fn(X[i:i+1]) in milliseconds."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    for i in idx[:warmup]:
        fn(X[i:i + 1])
    times = []
    for i in idx:
        t0 = time.perf_counter()
        fn(X[i:i + 1])
        times.append((time.perf_counter() - t0) * 1e3)
    times = np.asarray(times)
    return {"N": int(len(times)), "Mean_ms": float(times.mean()),
            "Median_ms": float(np.median(times)), "Std_ms": float(times.std())}
