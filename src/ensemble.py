"""Combination strategies, leave-one-out ablation and two optional single-model controls.

All combiners take a probability matrix [n_samples x n_models]. Weights / meta-learner are
fitted on the VALIDATION split only; the test split is never used for fitting.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .dataset import RANDOM_SEED
from .utils import balanced_accuracy_from_probs, compute_metrics


# ----------------------------------------------------------------------------- combiners
def majority_vote(prob_matrix, threshold=0.5):
    """Hard vote. Each member votes HS if p >= threshold; a clip is HS if at least HALF of the
    members vote HS - with 4 members a 2-2 tie is therefore resolved as HOTSPOT."""
    votes = (prob_matrix >= threshold).astype(int)
    return (votes.mean(axis=1) >= 0.5).astype(int)


def weighted_soft_vote(prob_matrix, weights, threshold=0.5):
    """Weighted average of member probabilities, thresholded at 0.5."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    combined = prob_matrix @ weights
    return combined, (combined >= threshold).astype(int)


def compute_validation_weights(val_prob_matrix, y_val):
    """Weight of each member = its balanced accuracy on the validation split (threshold 0.5).
    BA lies in [0.5, 1], so the largest possible weight ratio between two members is 2 : 1."""
    w = [balanced_accuracy_from_probs(y_val, val_prob_matrix[:, c]) for c in range(val_prob_matrix.shape[1])]
    return np.clip(np.array(w), 1e-3, None)


def fit_stacking_meta_learner(val_prob_matrix, y_val, seed=RANDOM_SEED):
    """Logistic regression (balanced class weights) on the members' validation probabilities."""
    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    meta.fit(val_prob_matrix, y_val)
    return meta


def stacking_predict(meta_learner, prob_matrix):
    proba = meta_learner.predict_proba(prob_matrix)[:, 1]
    return proba, (proba >= 0.5).astype(int)


# ----------------------------------------------------------------------------- ablation
def run_ablation_study(val_matrix, test_matrix, y_val, y_test, benchmark_key, model_names):
    """Leave-one-member-out on the WEIGHTED-VOTE ensemble (weights re-derived on validation data).
    The last row is the full ensemble."""
    rows = []
    for drop in model_names:
        keep = [i for i, n in enumerate(model_names) if n != drop]
        w = compute_validation_weights(val_matrix[:, keep], y_val)
        _, pred = weighted_soft_vote(test_matrix[:, keep], w)
        m = compute_metrics(y_test, pred)
        rows.append({"Benchmark": benchmark_key, "Removed_Model": drop,
                     "Remaining_Models": ",".join(n for n in model_names if n != drop),
                     "Balanced_Accuracy": m["Balanced_Accuracy"], "F1_Score": m["F1_Score"]})
        print(f"    without {drop:15s} -> Balanced Acc: {m['Balanced_Accuracy']:.4f}")
    w = compute_validation_weights(val_matrix, y_val)
    _, pred = weighted_soft_vote(test_matrix, w)
    m = compute_metrics(y_test, pred)
    rows.append({"Benchmark": benchmark_key, "Removed_Model": "(none - full ensemble)",
                 "Remaining_Models": ",".join(model_names),
                 "Balanced_Accuracy": m["Balanced_Accuracy"], "F1_Score": m["F1_Score"]})
    return rows


# ----------------------------------------------------------------------------- controls (optional)
# Question: is the stacking gain due to combining DIFFERENT models, or just to re-scaling / re-thresholding
# the CNN probability on the validation split? These two controls use the CNN ALONE.
def fit_single_model_calibrator(val_prob, y_val, seed=RANDOM_SEED):
    """Logistic regression (balanced class weights) on the CNN probability alone."""
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    lr.fit(np.asarray(val_prob).reshape(-1, 1), y_val)
    return lr


def calibrated_predict(calibrator, prob):
    p = calibrator.predict_proba(np.asarray(prob).reshape(-1, 1))[:, 1]
    return p, (p >= 0.5).astype(int)


def best_threshold_on_val(val_prob, y_val):
    """Threshold maximising validation BA (ties -> closest to 0.5). Noisy when the validation split has
    only a handful of hotspots (e.g. ICCAD-5: ~4)."""
    val_prob = np.asarray(val_prob)
    cands = np.unique(np.concatenate([[0.5], np.quantile(val_prob, np.linspace(0.01, 0.99, 99))]))
    scores = np.array([balanced_accuracy_from_probs(y_val, val_prob, t) for t in cands])
    best = cands[scores == scores.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])
