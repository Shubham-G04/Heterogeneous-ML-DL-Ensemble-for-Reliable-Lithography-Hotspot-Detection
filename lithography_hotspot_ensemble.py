

# ==============================================================================
# Next notebook code cell
# ==============================================================================import os, sys, subprocess, zipfile, gc

# Detect environment (works on both Kaggle and Colab)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

WORK_DIR = "/content" if IN_COLAB else "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
import gdown

FILE_ID = "1jx7gDR92sqoIw2Nh4NGwwZwC-qNp2osd"
ZIP_PATH = os.path.join(WORK_DIR, "iccad12_dataset.archive")
EXTRACT_DIR = os.path.join(WORK_DIR, "iccad12_dataset")

if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 10_000:
    print("Downloading ICCAD-12 dataset ...")
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)
else:
    print("Archive already present, skipping download.")

print("Archive size (bytes):", os.path.getsize(ZIP_PATH))

with open(ZIP_PATH, "rb") as f:
    header = f.read(8)


def extract_archive(archive_path, extract_dir, header):
    os.makedirs(extract_dir, exist_ok=True)

    if header.startswith(b"PK"):
        print("Detected ZIP archive, extracting ...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return

    if header.startswith(b"Rar!"):
        print("Detected RAR archive, extracting with unrar ...")
        if subprocess.run(["which", "unrar"], capture_output=True).returncode != 0:
            subprocess.run(["apt-get", "update", "-qq"], check=False)
            subprocess.run(["apt-get", "install", "-y", "-qq", "unrar"], check=False)
        result = subprocess.run(["unrar", "x", "-y", archive_path, extract_dir + "/"],
                                 capture_output=True, text=True)
        print(result.stdout[-800:])
        if result.returncode != 0:
            print("unrar failed, trying 7z as a fallback ...")
            subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=False)
            result2 = subprocess.run(["7z", "x", f"-o{extract_dir}", "-y", archive_path],
                                      capture_output=True, text=True)
            print(result2.stdout[-800:])
            if result2.returncode != 0:
                raise RuntimeError("Both unrar and 7z failed to extract the archive:\n"
                                    + result.stderr + "\n" + result2.stderr)
        return

    raise ValueError(
        f"Unrecognized archive format (header bytes: {header}). "
        "The downloaded file may be an HTML error/quota page rather than the real "
        "dataset -- open it in a text viewer to check."
    )


if not os.path.isdir(EXTRACT_DIR) or len(os.listdir(EXTRACT_DIR)) == 0:
    extract_archive(ZIP_PATH, EXTRACT_DIR, header)
else:
    print("Already extracted, skipping.")

print("Done. Extracted to:", EXTRACT_DIR)


def print_tree(root, max_depth=5, max_entries=15):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        n_img = sum(1 for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        label = os.path.basename(dirpath) or dirpath
        extra = f"   [{n_img} images]" if n_img else ""
        print(f"{indent}{label}/{extra}")
        dirnames.sort()
        if len(dirnames) > max_entries:
            print(f"{indent}  ... ({len(dirnames) - max_entries} more subfolders, truncated)")
            dirnames[:] = dirnames[:max_entries]

print_tree(EXTRACT_DIR)


import numpy as np
from PIL import Image

# From the reference paper (Verma, Rao, Hegde -- Fig. 3, ICCAD-12 statistics).
# Used only to sanity-check that our folder-discovery logic found the right
# images -- your zip should match these exactly since it's the same official
# ICCAD-12 benchmark the paper (and its GitHub repo) links to.
REFERENCE_STATS = {
    "iccad1": {"train_hs": 99,  "train_nhs": 340,  "test_hs": 226,  "test_nhs": 3869},
    "iccad2": {"train_hs": 174, "train_nhs": 5285, "test_hs": 498,  "test_nhs": 41298},
    "iccad3": {"train_hs": 909, "train_nhs": 4643, "test_hs": 1808, "test_nhs": 46333},
    "iccad4": {"train_hs": 95,  "train_nhs": 4452, "test_hs": 177,  "test_nhs": 31890},
    "iccad5": {"train_hs": 26,  "train_nhs": 2716, "test_hs": 41,   "test_nhs": 19327},
}

# NHS keywords checked FIRST (since "hs" is a substring of "nhs")
NHS_KEYWORDS = ["nhs", "non_hotspot", "nonhotspot", "non-hotspot", "negative", "non_hs", "non-hs", "non"]
HS_KEYWORDS = ["hotspot", "hot_spot", "hot-spot", "hs", "positive", "pos"]
TRAIN_KEYWORDS = ["train", "training"]
TEST_KEYWORDS = ["test", "testing"]


def _match_any(segment, keywords):
    seg = segment.lower()
    return any(k in seg for k in keywords)


def find_benchmark_root(extract_dir, bench_name):
    """Finds the folder anywhere under extract_dir whose name matches
    bench_name (e.g. 'iccad1'), ignoring case and -/_ separators."""
    target = bench_name.lower().replace("-", "").replace("_", "")
    candidates = []
    for dirpath, dirnames, _ in os.walk(extract_dir):
        for d in dirnames:
            norm = d.lower().replace("-", "").replace("_", "")
            if norm == target:
                candidates.append(os.path.join(dirpath, d))
    if not candidates:
        raise FileNotFoundError(f"Could not find a folder matching '{bench_name}' under {extract_dir}")
    candidates.sort(key=lambda p: p.count(os.sep))  # prefer shallowest match
    return candidates[0]


def discover_image_files(bench_root):
    """Walks bench_root and classifies every image file into
    (split, label) purely from folder-name keywords.
    Returns {'train': {'hs': [paths], 'nhs': [paths]},
             'test':  {'hs': [paths], 'nhs': [paths]}}"""
    result = {"train": {"hs": [], "nhs": []}, "test": {"hs": [], "nhs": []}}
    unclassified = []
    for dirpath, _, filenames in os.walk(bench_root):
        imgs = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not imgs:
            continue
        rel_parts = os.path.relpath(dirpath, bench_root).split(os.sep)

        split = None
        for part in rel_parts:
            if _match_any(part, TRAIN_KEYWORDS):
                split = "train"; break
            if _match_any(part, TEST_KEYWORDS):
                split = "test"; break

        label = None
        for part in rel_parts:
            if _match_any(part, NHS_KEYWORDS):
                label = "nhs"; break
        if label is None:
            for part in rel_parts:
                if _match_any(part, HS_KEYWORDS):
                    label = "hs"; break

        if split is None or label is None:
            unclassified.append((dirpath, len(imgs)))
            continue

        result[split][label].extend(os.path.join(dirpath, f) for f in imgs)

    if unclassified:
        print("  WARNING: could not classify these folders (skipped):")
        for d, n in unclassified[:20]:
            print(f"    {d}  ({n} images)")

    return result


def sanity_check(bench_name, discovered):
    ref = REFERENCE_STATS.get(bench_name)
    found = {
        "train_hs": len(discovered["train"]["hs"]),
        "train_nhs": len(discovered["train"]["nhs"]),
        "test_hs": len(discovered["test"]["hs"]),
        "test_nhs": len(discovered["test"]["nhs"]),
    }
    print(f"  Found:     {found}")
    if ref:
        print(f"  Reference: {ref}")
        if all(found[k] == ref[k] for k in ref):
            print("  MATCH -- folder discovery is correct.")
        else:
            print("  MISMATCH -- check the WARNING above / the tree printout; "
                  "adjust NHS_KEYWORDS / HS_KEYWORDS / TRAIN_KEYWORDS / TEST_KEYWORDS if needed.")
    return found


BENCHMARK_NAMES = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

discovered_all = {}
benchmark_roots = {}
for name in BENCHMARK_NAMES:
    print(f"\n--- {name} ---")
    root = find_benchmark_root(EXTRACT_DIR, name)
    benchmark_roots[name] = root
    print("Folder:", root)
    d = discover_image_files(root)
    discovered_all[name] = d
    sanity_check(name, d)

print("\nIf every benchmark above says MATCH, move on to the next cell.")
print("If any says MISMATCH, paste me its folder-tree section from the earlier cell's "
      "output and I'll fix the keyword lists.")


from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG -- tune these if you hit memory/time limits
# ---------------------------------------------------------------------------
IMG_SIZE = 48
IMG_CHANNELS = 1
VAL_SPLIT = 0.15
RANDOM_SEED = 42

CNN_EPOCHS = 15          # lower to 2-3 for a first quick test run
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3
USE_CLASS_WEIGHTS = True

ENSEMBLE_BASE_MODELS = ["cnn", "svm", "random_forest", "ann"]
RANDOM_FOREST_N_ESTIMATORS = 300
SVM_KERNEL = "rbf"
ANN_HIDDEN_LAYERS = (128, 64)
ANN_MAX_ITER = 500

OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
for d in [OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def _load_image_list(paths, img_size, channels):
    if len(paths) == 0:
        return np.zeros((0, img_size, img_size, channels), dtype=np.float32)
    mode = "L" if channels == 1 else "RGB"
    out = np.zeros((len(paths), img_size, img_size, channels), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert(mode).resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if channels == 1:
            arr = arr[..., np.newaxis]
        out[i] = arr
    return out


def build_dataset(discovered, img_size=IMG_SIZE, channels=IMG_CHANNELS,
                   val_split=VAL_SPLIT, seed=RANDOM_SEED, verbose=True):
    train_hs = _load_image_list(discovered["train"]["hs"], img_size, channels)
    train_nhs = _load_image_list(discovered["train"]["nhs"], img_size, channels)
    test_hs = _load_image_list(discovered["test"]["hs"], img_size, channels)
    test_nhs = _load_image_list(discovered["test"]["nhs"], img_size, channels)

    X_train_full = np.concatenate([train_hs, train_nhs], axis=0)
    y_train_full = np.concatenate([np.ones(len(train_hs)), np.zeros(len(train_nhs))]).astype(np.int32)
    X_test = np.concatenate([test_hs, test_nhs], axis=0)
    y_test = np.concatenate([np.ones(len(test_hs)), np.zeros(len(test_nhs))]).astype(np.int32)

    del train_hs, train_nhs, test_hs, test_nhs
    gc.collect()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_split,
        stratify=y_train_full, random_state=seed
    )
    del X_train_full, y_train_full
    gc.collect()

    if verbose:
        print(f"  train={len(X_train)} (HS={int(y_train.sum())}) | "
              f"val={len(X_val)} (HS={int(y_val.sum())}) | "
              f"test={len(X_test)} (HS={int(y_test.sum())}, NHS={int((y_test == 0).sum())})")

    return {"X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks


def build_cnn(img_size, channels, embedding_dim=64):
    inputs = layers.Input(shape=(img_size, img_size, channels))

    def basic_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="elu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="elu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation=None)(x)
        x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = layers.Activation("elu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        return x

    x = basic_block(inputs, 16)
    x = basic_block(x, 32)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = layers.Dropout(0.3)(embedding)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    full_model = models.Model(inputs, output, name="lightweight_cnn")
    embedding_model = models.Model(inputs, embedding, name="cnn_embedding_extractor")
    return full_model, embedding_model


def train_cnn(X_train, y_train, X_val, y_val, verbose=1):
    tf.random.set_seed(RANDOM_SEED)
    full_model, embedding_model = build_cnn(IMG_SIZE, IMG_CHANNELS)

    full_model.compile(
        optimizer=optimizers.Nadam(learning_rate=CNN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    class_weight = None
    if USE_CLASS_WEIGHTS:
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    history = full_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return full_model, embedding_model, history


def cnn_predict_proba(full_model, X, batch_size=256):
    return full_model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)


def extract_embeddings(embedding_model, X, batch_size=256):
    return embedding_model.predict(X, batch_size=batch_size, verbose=0)


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_classical_base_models(X_train_feat, y_train):
    scaler = StandardScaler().fit(X_train_feat)
    X_scaled = scaler.transform(X_train_feat)

    svm = SVC(kernel=SVM_KERNEL, probability=True, class_weight="balanced", random_state=RANDOM_SEED)
    svm.fit(X_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                                 class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_feat, y_train)

    ann = MLPClassifier(hidden_layer_sizes=ANN_HIDDEN_LAYERS, max_iter=ANN_MAX_ITER,
                         random_state=RANDOM_SEED, early_stopping=True)
    ann.fit(X_scaled, y_train)

    return {"svm": (svm, scaler), "random_forest": (rf, None), "ann": (ann, scaler)}


def classical_predict_proba(model, scaler, X_feat):
    X_in = scaler.transform(X_feat) if scaler is not None else X_feat
    return model.predict_proba(X_in)[:, 1]


def get_all_base_probabilities(cnn_full_model, classical_models, X_images, X_feat):
    probs = {"cnn": cnn_predict_proba(cnn_full_model, X_images)}
    for name in ["svm", "random_forest", "ann"]:
        model, scaler = classical_models[name]
        probs[name] = classical_predict_proba(model, scaler, X_feat)
    matrix = np.column_stack([probs[name] for name in ENSEMBLE_BASE_MODELS])
    return matrix, probs


def majority_vote(prob_matrix, threshold=0.5):
    votes = (prob_matrix >= threshold).astype(int)
    return (votes.mean(axis=1) >= 0.5).astype(int)


def weighted_soft_vote(prob_matrix, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    combined_prob = prob_matrix @ weights
    return combined_prob, (combined_prob >= 0.5).astype(int)


def compute_validation_weights(val_prob_matrix, y_val):
    weights = []
    for col in range(val_prob_matrix.shape[1]):
        weights.append(balanced_accuracy_from_probs(y_val, val_prob_matrix[:, col]))
    return np.clip(np.array(weights), 1e-3, None)


def fit_stacking_meta_learner(val_prob_matrix, y_val):
    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    meta.fit(val_prob_matrix, y_val)
    return meta


def stacking_predict(meta_learner, prob_matrix):
    proba = meta_learner.predict_proba(prob_matrix)[:, 1]
    return proba, (proba >= 0.5).astype(int)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Precision": precision, "Recall_Sensitivity": recall,
        "Specificity": specificity, "F1_Score": f1,
        "Balanced_Accuracy": balanced_acc,
    }


def balanced_accuracy_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return compute_metrics(y_true, y_pred)["Balanced_Accuracy"]


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HS", "NHS"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["HS", "NHS"])
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_model_comparison_bar(benchmark_keys, model_scores, title, ylabel, save_path):
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
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


import time
import itertools
import pandas as pd


def run_one_benchmark(benchmark_key, discovered):
    print(f"\n{'=' * 70}\n{benchmark_key}\n{'=' * 70}")
    t0 = time.time()

    print("Loading images ...")
    data = build_dataset(discovered)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("-- Training baseline CNN --")
    cnn_full, cnn_embed, _ = train_cnn(X_train, y_train, X_val, y_val, verbose=0)

    cnn_test_prob = cnn_predict_proba(cnn_full, X_test)
    cnn_test_pred = (cnn_test_prob >= 0.5).astype(int)
    baseline_metrics = compute_metrics(y_test, cnn_test_pred)
    print(f"Baseline CNN  -> Balanced Acc: {baseline_metrics['Balanced_Accuracy']:.4f}")

    print("-- Extracting CNN embeddings & training SVM / RF / ANN --")
    train_feat = extract_embeddings(cnn_embed, X_train)
    val_feat = extract_embeddings(cnn_embed, X_val)
    test_feat = extract_embeddings(cnn_embed, X_test)

    classical = train_classical_base_models(train_feat, y_train)

    val_matrix, val_probs = get_all_base_probabilities(cnn_full, classical, X_val, val_feat)
    test_matrix, test_probs = get_all_base_probabilities(cnn_full, classical, X_test, test_feat)

    per_model_metrics = {}
    for name in ENSEMBLE_BASE_MODELS:
        pred = (test_probs[name] >= 0.5).astype(int)
        per_model_metrics[name] = compute_metrics(y_test, pred)
        print(f"  {name:15s} -> Balanced Acc: {per_model_metrics[name]['Balanced_Accuracy']:.4f}")

    print("-- Building ensembles (voting / weighted voting / stacking) --")
    vote_pred = majority_vote(test_matrix)
    vote_metrics = compute_metrics(y_test, vote_pred)

    weights = compute_validation_weights(val_matrix, y_val)
    _, wvote_pred = weighted_soft_vote(test_matrix, weights)
    wvote_metrics = compute_metrics(y_test, wvote_pred)

    meta = fit_stacking_meta_learner(val_matrix, y_val)
    _, stack_pred = stacking_predict(meta, test_matrix)
    stack_metrics = compute_metrics(y_test, stack_pred)

    print(f"  Majority vote   -> Balanced Acc: {vote_metrics['Balanced_Accuracy']:.4f}")
    print(f"  Weighted vote   -> Balanced Acc: {wvote_metrics['Balanced_Accuracy']:.4f}")
    print(f"  Stacking        -> Balanced Acc: {stack_metrics['Balanced_Accuracy']:.4f}")

    ensemble_variants = {
        "Ensemble_MajorityVote": vote_metrics,
        "Ensemble_WeightedVote": wvote_metrics,
        "Ensemble_Stacking": stack_metrics,
    }
    best_variant_name = max(ensemble_variants, key=lambda k: ensemble_variants[k]["Balanced_Accuracy"])
    best_ensemble_pred = {"Ensemble_MajorityVote": vote_pred,
                           "Ensemble_WeightedVote": wvote_pred,
                           "Ensemble_Stacking": stack_pred}[best_variant_name]

    print("-- Ablation: leave-one-base-model-out (weighted voting) --")
    ablation_rows = []
    all_names = ENSEMBLE_BASE_MODELS
    for drop in all_names:
        keep_idx = [i for i, n in enumerate(all_names) if n != drop]
        sub_weights = compute_validation_weights(val_matrix[:, keep_idx], y_val)
        _, sub_pred = weighted_soft_vote(test_matrix[:, keep_idx], sub_weights)
        sub_metrics = compute_metrics(y_test, sub_pred)
        ablation_rows.append({
            "Benchmark": benchmark_key, "Removed_Model": drop,
            "Remaining_Models": ",".join(n for n in all_names if n != drop),
            "Balanced_Accuracy": sub_metrics["Balanced_Accuracy"], "F1_Score": sub_metrics["F1_Score"],
        })
        print(f"  without {drop:15s} -> Balanced Acc: {sub_metrics['Balanced_Accuracy']:.4f}")
    ablation_rows.append({
        "Benchmark": benchmark_key, "Removed_Model": "(none - full ensemble)",
        "Remaining_Models": ",".join(all_names),
        "Balanced_Accuracy": wvote_metrics["Balanced_Accuracy"], "F1_Score": wvote_metrics["F1_Score"],
    })

    plot_confusion_matrix(y_test, cnn_test_pred, f"{benchmark_key}: Baseline CNN",
                           os.path.join(FIGURES_DIR, f"{benchmark_key}_baseline_cnn_cm.png"))
    plot_confusion_matrix(y_test, best_ensemble_pred, f"{benchmark_key}: Best Ensemble ({best_variant_name})",
                           os.path.join(FIGURES_DIR, f"{benchmark_key}_best_ensemble_cm.png"))

    del data, X_train, X_val, X_test, train_feat, val_feat, test_feat
    gc.collect()

    print(f"[{benchmark_key}] done in {time.time() - t0:.1f}s")
    return {
        "benchmark": benchmark_key, "baseline_metrics": baseline_metrics,
        "per_model_metrics": per_model_metrics, "ensemble_variants": ensemble_variants,
        "best_variant_name": best_variant_name, "ablation_rows": ablation_rows,
    }


all_results = []
for name in BENCHMARK_NAMES:
    result = run_one_benchmark(name.upper().replace("ICCAD", "ICCAD-"), discovered_all[name])
    all_results.append(result)

# ---------------- Summary table ----------------
summary_rows = []
for r in all_results:
    bk = r["benchmark"]
    summary_rows.append({"Benchmark": bk, "Model": "Baseline_CNN", **r["baseline_metrics"]})
    for name, m in r["per_model_metrics"].items():
        summary_rows.append({"Benchmark": bk, "Model": name, **m})
    for name, m in r["ensemble_variants"].items():
        summary_rows.append({"Benchmark": bk, "Model": name, **m})

df = pd.DataFrame(summary_rows)
save_metrics_table_csv(summary_rows, os.path.join(RESULTS_DIR, "per_benchmark_results.csv"))

avg_df = df.groupby("Model")[["Balanced_Accuracy", "Precision", "Recall_Sensitivity",
                               "Specificity", "F1_Score"]].mean().reset_index()
avg_df.insert(0, "Benchmark", "AVERAGE")
avg_df.to_csv(os.path.join(RESULTS_DIR, "average_results.csv"), index=False)

print("\n" + "=" * 70)
print("FINAL SUMMARY (Balanced Accuracy, averaged over 5 benchmarks)")
print("=" * 70)
print(avg_df[["Model", "Balanced_Accuracy"]].sort_values("Balanced_Accuracy", ascending=False).to_string(index=False))

ablation_all = list(itertools.chain.from_iterable(r["ablation_rows"] for r in all_results))
save_metrics_table_csv(ablation_all, os.path.join(RESULTS_DIR, "ablation_results.csv"))
abl_df = pd.DataFrame(ablation_all)
abl_avg = abl_df.groupby("Removed_Model")["Balanced_Accuracy"].mean().reset_index().sort_values(
    "Balanced_Accuracy", ascending=False)
abl_avg.to_csv(os.path.join(RESULTS_DIR, "ablation_average.csv"), index=False)
print("\nAblation (avg Balanced Accuracy across 5 benchmarks, weighted voting):")
print(abl_avg.to_string(index=False))

benchmark_keys = [r["benchmark"] for r in all_results]
plot_models = ["Baseline_CNN", "Ensemble_MajorityVote", "Ensemble_WeightedVote", "Ensemble_Stacking"]
model_scores = {
    m: [df[(df.Benchmark == bk) & (df.Model == m)]["Balanced_Accuracy"].values[0] for bk in benchmark_keys]
    for m in plot_models
}
plot_model_comparison_bar(benchmark_keys, model_scores,
                           "Baseline CNN vs. Ensemble Variants (Balanced Accuracy)",
                           "Balanced Accuracy",
                           os.path.join(FIGURES_DIR, "baseline_vs_ensemble_comparison.png"))

print(f"\nAll results saved under: {RESULTS_DIR}")
print(f"All figures saved under: {FIGURES_DIR}")


import shutil
import os

zip_path = shutil.make_archive(
    os.path.join(WORK_DIR, "g6_results"),
    "zip",
    OUTPUT_DIR
)

print("Results zipped to:", zip_path)
print("On Kaggle, the ZIP file is available in the notebook's Output/Files section.")