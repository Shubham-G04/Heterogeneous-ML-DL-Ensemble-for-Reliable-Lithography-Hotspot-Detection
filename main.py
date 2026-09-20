"""Run the CNN baseline and the heterogeneous ensembles on the REAL ICCAD-12 benchmarks 1-5.

Just run it - no arguments needed:

    python main.py

What happens automatically:
  1. finds the ICCAD-12 dataset (--data_dir, $ICCAD12_DIR, ./data, Kaggle inputs, Colab /content) and
     otherwise downloads the official one with gdown;
  2. checks the image counts against the official ICCAD-12 benchmark and REFUSES to run on anything else
     (demo / wrong data) unless --allow_mismatch is given;
  3. trains and evaluates on ICCAD-1..5, writes CSVs, figures and run_config.json, and zips everything
     to g6_results.zip next to the output folder.
Optional: --benchmarks iccad1 iccad5 --seed 7 --controls --latency --save_predictions --no_zip
"""
import argparse
import gc
import itertools
import json
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd

from src import dataset as ds
from src import ensemble as ens
from src import models as md
from src import utils as ut
from src.dataset import BENCHMARK_NAMES, build_dataset, discover_image_files, find_benchmark_root, sanity_check
from src.models import (ENSEMBLE_BASE_MODELS, classical_predict_proba, cnn_predict_proba, extract_embeddings,
                        get_all_base_probabilities, train_classical_base_models, train_cnn)


def library_versions():
    from importlib import metadata
    out = {"python": platform.python_version()}
    for pkg in ("tensorflow", "keras", "scikit-learn", "numpy", "pandas", "pillow"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def experiment_config(seed):
    return {
        "seed": seed, "img_size": ds.IMG_SIZE, "channels": ds.IMG_CHANNELS, "val_split": ds.VAL_SPLIT,
        "cnn": {"epochs": md.CNN_EPOCHS, "batch_size": md.CNN_BATCH_SIZE, "lr": md.CNN_LEARNING_RATE,
                "optimizer": "Nadam", "loss": "binary_crossentropy", "class_weights": md.USE_CLASS_WEIGHTS,
                "early_stopping": f"val_auc, patience={md.EARLY_STOP_PATIENCE}, restore_best_weights",
                "embedding_dim": md.EMBEDDING_DIM},
        "svm": {"kernel": md.SVM_KERNEL, "probability": True, "class_weight": "balanced"},
        "random_forest": {"n_estimators": md.RANDOM_FOREST_N_ESTIMATORS, "class_weight": "balanced"},
        "ann": {"hidden_layers": list(md.ANN_HIDDEN_LAYERS), "max_iter": md.ANN_MAX_ITER,
                "early_stopping": True, "class_weight": "none (not supported by MLPClassifier)"},
        "majority_vote": "hard vote, tie (2-2) -> hotspot",
        "weighted_vote": "weights = validation BA of each member (normalised), threshold 0.5",
        "stacking": "LogisticRegression(class_weight='balanced') on member probabilities, fitted on validation split",
        "members_input": "SVM/RF/ANN use the CNN's 64-D embedding, not raw pixels",
        "libraries": library_versions(),
    }


def run_single_benchmark(benchmark_key, discovered, output_dir, seed=ds.RANDOM_SEED,
                         controls=False, latency=False, save_predictions=False):
    print(f"\n{'=' * 70}\nRunning Benchmark: {benchmark_key}\n{'=' * 70}")
    t0 = time.time()
    figures_dir = os.path.join(output_dir, "figures")
    results_dir = os.path.join(output_dir, "results")
    for d in (output_dir, figures_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    print("Loading images and constructing datasets ...")
    data = build_dataset(discovered, seed=seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("-- Training baseline CNN --")
    cnn_full, cnn_embed, _ = train_cnn(X_train, y_train, X_val, y_val, seed=seed, verbose=0)
    cnn_test_prob = cnn_predict_proba(cnn_full, X_test)
    cnn_test_pred = (cnn_test_prob >= 0.5).astype(int)
    baseline_metrics = ut.compute_metrics(y_test, cnn_test_pred)
    print(f"Baseline CNN -> Balanced Acc: {baseline_metrics['Balanced_Accuracy']:.4f}")

    print("-- Extracting CNN embeddings & training SVM / RF / ANN --")
    train_feat = extract_embeddings(cnn_embed, X_train)
    val_feat = extract_embeddings(cnn_embed, X_val)
    test_feat = extract_embeddings(cnn_embed, X_test)
    classical = train_classical_base_models(train_feat, y_train, seed=seed)
    val_matrix, val_probs = get_all_base_probabilities(cnn_full, classical, X_val, val_feat, ENSEMBLE_BASE_MODELS)
    test_matrix, test_probs = get_all_base_probabilities(cnn_full, classical, X_test, test_feat, ENSEMBLE_BASE_MODELS)

    per_model_metrics = {}
    for name in ENSEMBLE_BASE_MODELS:
        pred = (test_probs[name] >= 0.5).astype(int)
        per_model_metrics[name] = ut.compute_metrics(y_test, pred)
        print(f"  {name:15s} -> Balanced Acc: {per_model_metrics[name]['Balanced_Accuracy']:.4f}")

    print("-- Building ensembles (Majority Vote / Weighted Soft Vote / Stacking) --")
    vote_pred = ens.majority_vote(test_matrix)
    weights = ens.compute_validation_weights(val_matrix, y_val)
    _, wvote_pred = ens.weighted_soft_vote(test_matrix, weights)
    meta = ens.fit_stacking_meta_learner(val_matrix, y_val, seed=seed)
    _, stack_pred = ens.stacking_predict(meta, test_matrix)
    ensemble_variants = {"Ensemble_MajorityVote": ut.compute_metrics(y_test, vote_pred),
                         "Ensemble_WeightedVote": ut.compute_metrics(y_test, wvote_pred),
                         "Ensemble_Stacking": ut.compute_metrics(y_test, stack_pred)}
    ensemble_preds = {"Ensemble_MajorityVote": vote_pred, "Ensemble_WeightedVote": wvote_pred,
                      "Ensemble_Stacking": stack_pred}
    for k, m in ensemble_variants.items():
        print(f"  {k:22s} -> Balanced Acc: {m['Balanced_Accuracy']:.4f}")
    print(f"  val weights (cnn, svm, rf, ann): {np.round(weights / weights.sum(), 3).tolist()}")
    print(f"  stacking coefficients          : {np.round(meta.coef_[0], 3).tolist()}")

    # NOTE: the 'best' variant is picked by TEST balanced accuracy and is used ONLY to choose which
    # confusion-matrix figure to draw. Report all three variants in the paper, not just the best one.
    best_variant_name = max(ensemble_variants, key=lambda k: ensemble_variants[k]["Balanced_Accuracy"])

    print("-- Ablation: Leave-One-Base-Model-Out (Weighted Soft Vote) --")
    ablation_rows = ens.run_ablation_study(val_matrix, test_matrix, y_val, y_test, benchmark_key, ENSEMBLE_BASE_MODELS)

    control_rows = []
    if controls:
        print("-- Controls: CNN alone, re-scaled / re-thresholded on the validation split --")
        cal = ens.fit_single_model_calibrator(val_probs["cnn"], y_val, seed=seed)
        _, cal_pred = ens.calibrated_predict(cal, test_probs["cnn"])
        thr = ens.best_threshold_on_val(val_probs["cnn"], y_val)
        thr_pred = (test_probs["cnn"] >= thr).astype(int)
        for name, pred, extra in (("Control_CNN_PlattBalanced", cal_pred, {"Threshold": 0.5}),
                                  ("Control_CNN_ValThreshold", thr_pred, {"Threshold": thr})):
            m = ut.compute_metrics(y_test, pred)
            control_rows.append({"Benchmark": benchmark_key, "Model": name, **m, **extra})
            print(f"  {name:27s} -> Balanced Acc: {m['Balanced_Accuracy']:.4f}")

    latency_rows = []
    if latency:
        latency_rows = measure_benchmark_latency(benchmark_key, cnn_full, classical, meta, X_test)

    if save_predictions:
        np.savez_compressed(os.path.join(results_dir, f"predictions_{benchmark_key}.npz"),
                            y_true=y_test, member_order=np.array(ENSEMBLE_BASE_MODELS),
                            test_probs=test_matrix, cnn_pred=cnn_test_pred,
                            majority_pred=vote_pred, weighted_pred=wvote_pred, stacking_pred=stack_pred)

    ut.plot_confusion_matrix(y_test, cnn_test_pred, f"{benchmark_key}: Baseline CNN",
                             os.path.join(figures_dir, f"{benchmark_key}_baseline_cnn_cm.png"))
    ut.plot_confusion_matrix(y_test, ensemble_preds[best_variant_name],
                             f"{benchmark_key}: Best Ensemble ({best_variant_name})",
                             os.path.join(figures_dir, f"{benchmark_key}_best_ensemble_cm.png"))

    del data, X_train, X_val, X_test, train_feat, val_feat, test_feat
    gc.collect()
    print(f"[{benchmark_key}] completed in {time.time() - t0:.1f}s")
    return {"benchmark": benchmark_key, "baseline_metrics": baseline_metrics,
            "per_model_metrics": per_model_metrics, "ensemble_variants": ensemble_variants,
            "best_variant_name": best_variant_name, "ablation_rows": ablation_rows,
            "control_rows": control_rows, "latency_rows": latency_rows}


def measure_benchmark_latency(benchmark_key, cnn_full, classical, meta, X_test, n=200):
    """Batch-size-1 inference latency: CNN alone vs. the full stacking ensemble (one CNN forward pass
    yields both the embedding and the CNN probability, then SVM + RF + ANN + logistic meta-learner)."""
    joint = md.build_joint_model(cnn_full)
    classical["random_forest"][0].set_params(n_jobs=1)          # avoid thread start-up cost per clip

    def cnn_fn(x):
        return cnn_full(x, training=False).numpy()

    def ens_fn(x):
        emb, p = joint(x, training=False)
        emb, p = emb.numpy(), p.numpy().reshape(-1)
        cols = [p] + [classical_predict_proba(*classical[k], emb) for k in ("svm", "random_forest", "ann")]
        return ens.stacking_predict(meta, np.column_stack(cols))

    rows = []
    for name, fn, params in (("CNN_alone", cnn_fn, cnn_full.count_params()),
                             ("Ensemble_Stacking", ens_fn, None)):
        r = ut.measure_latency_ms(fn, X_test, n=n)
        rows.append({"Benchmark": benchmark_key, "Model": name, "CNN_params": cnn_full.count_params(), **r})
        print(f"  latency {name:18s}: mean {r['Mean_ms']:.2f} ms / median {r['Median_ms']:.2f} ms per clip")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Heterogeneous ML-DL Ensemble for ICCAD-12 Hotspot Detection "
                                            "(runs on the real dataset automatically)")
    p.add_argument("--work_dir", default=None, help="download/extract folder (default: auto by environment)")
    p.add_argument("--data_dir", default=None, help="folder that already contains the extracted ICCAD-12 data")
    p.add_argument("--output_dir", default=None, help="results folder (default: auto by environment)")
    p.add_argument("--benchmarks", nargs="+", default=BENCHMARK_NAMES)
    p.add_argument("--seed", type=int, default=ds.RANDOM_SEED)
    p.add_argument("--allow_mismatch", action="store_true",
                   help="do NOT abort when image counts differ from the official ICCAD-12 benchmark")
    p.add_argument("--controls", action="store_true", help="also run single-CNN calibration/threshold controls")
    p.add_argument("--latency", action="store_true", help="measure per-clip inference latency (needs TensorFlow)")
    p.add_argument("--save_predictions", action="store_true", help="save per-clip predictions (.npz) for paired tests")
    p.add_argument("--no_zip", action="store_true", help="do not zip the outputs to g6_results.zip")
    args = p.parse_args(argv)
    args.work_dir = args.work_dir or ds.default_work_dir()
    args.output_dir = args.output_dir or ds.default_output_dir()
    unknown = [b for b in args.benchmarks if b not in BENCHMARK_NAMES]
    if unknown:
        p.error(f"unknown benchmark(s) {unknown}; choose from {BENCHMARK_NAMES}")

    print("=" * 70 + "\nICCAD-12 hotspot detection - REAL dataset run\n" + "=" * 70)
    extract_dir = ds.ensure_dataset(args.work_dir, args.benchmarks, explicit=args.data_dir)
    discovered_all = {}
    n_match = 0
    for name in args.benchmarks:
        print(f"\n--- {name} ---")
        root = find_benchmark_root(extract_dir, name)
        print("Folder:", root)
        discovered_all[name] = discover_image_files(root)
        sanity_check(name, discovered_all[name])
        n_match += ds.verify_dataset(name, discovered_all[name], strict=not args.allow_mismatch)
    print(f"\nDataset check: {n_match}/{len(args.benchmarks)} benchmarks match the official ICCAD-12 counts.")

    results_dir = os.path.join(args.output_dir, "results")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump(experiment_config(args.seed), f, indent=2)

    all_results = [run_single_benchmark(n.upper().replace("ICCAD", "ICCAD-"), discovered_all[n], args.output_dir,
                                        seed=args.seed, controls=args.controls, latency=args.latency,
                                        save_predictions=args.save_predictions)
                   for n in args.benchmarks]

    rows = []
    for r in all_results:
        bk = r["benchmark"]
        rows.append({"Benchmark": bk, "Model": "Baseline_CNN", **r["baseline_metrics"]})
        rows += [{"Benchmark": bk, "Model": n, **m} for n, m in r["per_model_metrics"].items()]
        rows += [{"Benchmark": bk, "Model": n, **m} for n, m in r["ensemble_variants"].items()]
    df = pd.DataFrame(rows)
    ut.save_metrics_table_csv(rows, os.path.join(results_dir, "per_benchmark_results.csv"))

    cols = ["Balanced_Accuracy", "Precision", "Recall_Sensitivity", "Specificity", "F1_Score"]
    avg_df = df.groupby("Model")[cols].mean().reset_index()
    avg_df.insert(0, "Benchmark", "AVERAGE")
    avg_df.to_csv(os.path.join(results_dir, "average_results.csv"), index=False)
    print("\n" + "=" * 70 + "\nFINAL SUMMARY (Balanced Accuracy, macro-average over benchmarks)\n" + "=" * 70)
    print(avg_df[["Model", "Balanced_Accuracy"]].sort_values("Balanced_Accuracy", ascending=False).to_string(index=False))

    ablation_all = list(itertools.chain.from_iterable(r["ablation_rows"] for r in all_results))
    ut.save_metrics_table_csv(ablation_all, os.path.join(results_dir, "ablation_results.csv"))
    (pd.DataFrame(ablation_all).groupby("Removed_Model")["Balanced_Accuracy"].mean().reset_index()
        .sort_values("Balanced_Accuracy", ascending=False)
        .to_csv(os.path.join(results_dir, "ablation_average.csv"), index=False))

    control_all = list(itertools.chain.from_iterable(r["control_rows"] for r in all_results))
    if control_all:
        ut.save_metrics_table_csv(control_all, os.path.join(results_dir, "control_results.csv"))
        print("\nControl average BA:\n" + pd.DataFrame(control_all).groupby("Model")["Balanced_Accuracy"].mean().to_string())
    latency_all = list(itertools.chain.from_iterable(r["latency_rows"] for r in all_results))
    if latency_all:
        ut.save_metrics_table_csv(latency_all, os.path.join(results_dir, "latency_results.csv"))

    keys = [r["benchmark"] for r in all_results]
    scores = {m: [df[(df.Benchmark == k) & (df.Model == m)]["Balanced_Accuracy"].values[0] for k in keys]
              for m in ("Baseline_CNN", "Ensemble_MajorityVote", "Ensemble_WeightedVote", "Ensemble_Stacking")}
    ut.plot_model_comparison_bar(keys, scores, "Baseline CNN vs. Ensemble Variants (Balanced Accuracy)",
                                 "Balanced Accuracy", os.path.join(figures_dir, "baseline_vs_ensemble_comparison.png"))
    print(f"\nAll results saved under: {results_dir}\nAll figures saved under: {figures_dir}")
    if not args.no_zip:
        out_abs = os.path.abspath(args.output_dir)
        zip_path = shutil.make_archive(os.path.join(os.path.dirname(out_abs), "g6_results"), "zip", out_abs)
        print("Results zipped to:", zip_path)


if __name__ == "__main__":
    main()
