import os
import time
import gc
import argparse
import itertools
import pandas as pd

from src.dataset import (
    download_iccad12_dataset,
    find_benchmark_root,
    discover_image_files,
    build_dataset,
    BENCHMARK_NAMES if "BENCHMARK_NAMES" in locals() else ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"],
)
from src.models import (
    train_cnn,
    cnn_predict_proba,
    extract_embeddings,
    train_classical_base_models,
    get_all_base_probabilities,
)
from src.ensemble import (
    majority_vote,
    weighted_soft_vote,
    compute_validation_weights,
    fit_stacking_meta_learner,
    stacking_predict,
    run_ablation_study,
)
from src.utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison_bar,
    save_metrics_table_csv,
)

BENCHMARK_NAMES = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
ENSEMBLE_BASE_MODELS = ["cnn", "svm", "random_forest", "ann"]


def run_single_benchmark(benchmark_key, discovered, output_dir):
    print(f"\n{'=' * 70}\nRunning Benchmark: {benchmark_key}\n{'=' * 70}")
    t0 = time.time()

    figures_dir = os.path.join(output_dir, "figures")
    results_dir = os.path.join(output_dir, "results")
    for d in [output_dir, figures_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print("Loading images and constructing datasets ...")
    data = build_dataset(discovered)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("-- Training Baseline CNN --")
    cnn_full, cnn_embed, _ = train_cnn(X_train, y_train, X_val, y_val, verbose=0)

    cnn_test_prob = cnn_predict_proba(cnn_full, X_test)
    cnn_test_pred = (cnn_test_prob >= 0.5).astype(int)
    baseline_metrics = compute_metrics(y_test, cnn_test_pred)
    print(f"Baseline CNN  -> Balanced Acc: {baseline_metrics['Balanced_Accuracy']:.4f}")

    print("-- Extracting CNN Embeddings & Training SVM / RF / ANN --")
    train_feat = extract_embeddings(cnn_embed, X_train)
    val_feat = extract_embeddings(cnn_embed, X_val)
    test_feat = extract_embeddings(cnn_embed, X_test)

    classical = train_classical_base_models(train_feat, y_train)

    val_matrix, val_probs = get_all_base_probabilities(cnn_full, classical, X_val, val_feat, ENSEMBLE_BASE_MODELS)
    test_matrix, test_probs = get_all_base_probabilities(cnn_full, classical, X_test, test_feat, ENSEMBLE_BASE_MODELS)

    per_model_metrics = {}
    for name in ENSEMBLE_BASE_MODELS:
        pred = (test_probs[name] >= 0.5).astype(int)
        per_model_metrics[name] = compute_metrics(y_test, pred)
        print(f"  {name:15s} -> Balanced Acc: {per_model_metrics[name]['Balanced_Accuracy']:.4f}")

    print("-- Building Ensembles (Majority Vote / Weighted Soft Vote / Stacking) --")
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
    best_ensemble_pred = {
        "Ensemble_MajorityVote": vote_pred,
        "Ensemble_WeightedVote": wvote_pred,
        "Ensemble_Stacking": stack_pred,
    }[best_variant_name]

    print("-- Ablation: Leave-One-Base-Model-Out (Weighted Soft Vote) --")
    ablation_rows = run_ablation_study(val_matrix, test_matrix, y_val, y_test, benchmark_key, ENSEMBLE_BASE_MODELS)

    plot_confusion_matrix(
        y_test, cnn_test_pred, f"{benchmark_key}: Baseline CNN", os.path.join(figures_dir, f"{benchmark_key}_baseline_cnn_cm.png")
    )
    plot_confusion_matrix(
        y_test,
        best_ensemble_pred,
        f"{benchmark_key}: Best Ensemble ({best_variant_name})",
        os.path.join(figures_dir, f"{benchmark_key}_best_ensemble_cm.png"),
    )

    del data, X_train, X_val, X_test, train_feat, val_feat, test_feat
    gc.collect()

    print(f"[{benchmark_key}] completed in {time.time() - t0:.1f}s")
    return {
        "benchmark": benchmark_key,
        "baseline_metrics": baseline_metrics,
        "per_model_metrics": per_model_metrics,
        "ensemble_variants": ensemble_variants,
        "best_variant_name": best_variant_name,
        "ablation_rows": ablation_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous ML-DL Ensemble for ICCAD-12 Hotspot Detection")
    parser.add_argument("--work_dir", type=str, default="./data", help="Directory to store extracted dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save figures and CSV results")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARK_NAMES, help="Benchmarks to evaluate")
    args = parser.parse_args()

    extract_dir = download_iccad12_dataset(args.work_dir)

    discovered_all = {}
    for name in args.benchmarks:
        root = find_benchmark_root(extract_dir, name)
        discovered_all[name] = discover_image_files(root)

    all_results = []
    for name in args.benchmarks:
        bench_key = name.upper().replace("ICCAD", "ICCAD-")
        res = run_single_benchmark(bench_key, discovered_all[name], args.output_dir)
        all_results.append(res)

    results_dir = os.path.join(args.output_dir, "results")
    figures_dir = os.path.join(args.output_dir, "figures")

    summary_rows = []
    for r in all_results:
        bk = r["benchmark"]
        summary_rows.append({"Benchmark": bk, "Model": "Baseline_CNN", **r["baseline_metrics"]})
        for name, m in r["per_model_metrics"].items():
            summary_rows.append({"Benchmark": bk, "Model": name, **m})
        for name, m in r["ensemble_variants"].items():
            summary_rows.append({"Benchmark": bk, "Model": name, **m})

    df = pd.DataFrame(summary_rows)
    save_metrics_table_csv(summary_rows, os.path.join(results_dir, "per_benchmark_results.csv"))

    avg_df = (
        df.groupby("Model")[["Balanced_Accuracy", "Precision", "Recall_Sensitivity", "Specificity", "F1_Score"]]
        .mean()
        .reset_index()
    )
    avg_df.insert(0, "Benchmark", "AVERAGE")
    avg_df.to_csv(os.path.join(results_dir, "average_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY (Balanced Accuracy, averaged over benchmarks)")
    print("=" * 70)
    print(avg_df[["Model", "Balanced_Accuracy"]].sort_values("Balanced_Accuracy", ascending=False).to_string(index=False))

    ablation_all = list(itertools.chain.from_iterable(r["ablation_rows"] for r in all_results))
    save_metrics_table_csv(ablation_all, os.path.join(results_dir, "ablation_results.csv"))
    abl_df = pd.DataFrame(ablation_all)
    abl_avg = abl_df.groupby("Removed_Model")["Balanced_Accuracy"].mean().reset_index().sort_values("Balanced_Accuracy", ascending=False)
    abl_avg.to_csv(os.path.join(results_dir, "ablation_average.csv"), index=False)

    benchmark_keys = [r["benchmark"] for r in all_results]
    plot_models = ["Baseline_CNN", "Ensemble_MajorityVote", "Ensemble_WeightedVote", "Ensemble_Stacking"]
    model_scores = {
        m: [df[(df.Benchmark == bk) & (df.Model == m)]["Balanced_Accuracy"].values[0] for bk in benchmark_keys]
        for m in plot_models
    }
    plot_model_comparison_bar(
        benchmark_keys,
        model_scores,
        "Baseline CNN vs. Ensemble Variants (Balanced Accuracy)",
        "Balanced Accuracy",
        os.path.join(figures_dir, "baseline_vs_ensemble_comparison.png"),
    )

    print(f"\nAll results saved under: {results_dir}")
    print(f"All figures saved under: {figures_dir}")


if __name__ == "__main__":
    main()
