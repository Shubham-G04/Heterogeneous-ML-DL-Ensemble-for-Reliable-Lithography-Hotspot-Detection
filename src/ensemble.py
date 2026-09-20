import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import compute_metrics, balanced_accuracy_from_probs


def majority_vote(prob_matrix, threshold=0.5):
    """Computes majority (hard) voting across base model predictions."""
    votes = (prob_matrix >= threshold).astype(int)
    return (votes.mean(axis=1) >= 0.5).astype(int)


def weighted_soft_vote(prob_matrix, weights):
    """Computes weighted soft voting across base model probabilities."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    combined_prob = prob_matrix @ weights
    return combined_prob, (combined_prob >= 0.5).astype(int)


def compute_validation_weights(val_prob_matrix, y_val):
    """Computes model weights proportional to validation balanced accuracy."""
    weights = []
    for col in range(val_prob_matrix.shape[1]):
        weights.append(balanced_accuracy_from_probs(y_val, val_prob_matrix[:, col]))
    return np.clip(np.array(weights), 1e-3, None)


def fit_stacking_meta_learner(val_prob_matrix, y_val, seed=42):
    """Trains a class-balanced Logistic Regression meta-learner on validation predictions."""
    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    meta.fit(val_prob_matrix, y_val)
    return meta


def stacking_predict(meta_learner, prob_matrix, threshold=0.5):
    """Generates predictions from stacking meta-learner."""
    proba = meta_learner.predict_proba(prob_matrix)[:, 1]
    return proba, (proba >= threshold).astype(int)


def run_ablation_study(val_matrix, test_matrix, y_val, y_test, benchmark_key, base_models=["cnn", "svm", "random_forest", "ann"]):
    """Performs leave-one-base-model-out ablation analysis for weighted soft voting."""
    ablation_rows = []
    full_weights = compute_validation_weights(val_matrix, y_val)
    _, full_pred = weighted_soft_vote(test_matrix, full_weights)
    full_metrics = compute_metrics(y_test, full_pred)

    for drop in base_models:
        keep_idx = [i for i, n in enumerate(base_models) if n != drop]
        sub_weights = compute_validation_weights(val_matrix[:, keep_idx], y_val)
        _, sub_pred = weighted_soft_vote(test_matrix[:, keep_idx], sub_weights)
        sub_metrics = compute_metrics(y_test, sub_pred)
        ablation_rows.append({
            "Benchmark": benchmark_key,
            "Removed_Model": drop,
            "Remaining_Models": ",".join(n for n in base_models if n != drop),
            "Balanced_Accuracy": sub_metrics["Balanced_Accuracy"],
            "F1_Score": sub_metrics["F1_Score"],
        })

    ablation_rows.append({
        "Benchmark": benchmark_key,
        "Removed_Model": "(none - full ensemble)",
        "Remaining_Models": ",".join(base_models),
        "Balanced_Accuracy": full_metrics["Balanced_Accuracy"],
        "F1_Score": full_metrics["F1_Score"],
    })

    return ablation_rows
