"""DEMO / SMOKE TEST ONLY - this is NOT how you run the project (use `python main.py` for the real dataset).

Runs WITHOUT TensorFlow or the real dataset.

Creates a tiny synthetic ICCAD-style folder tree, replaces the CNN by a fixed random-projection stand-in and
runs the full pipeline (discovery -> split -> SVM/RF/ANN -> voting/stacking -> ablation -> CSV/figures).
It checks plumbing and output schemas, NOT the scientific results.
   python tests/smoke_test.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M                                   # noqa: E402
from src import dataset as ds                       # noqa: E402
from src import ensemble as ens                     # noqa: E402
from src import utils as ut                         # noqa: E402


def make_fake_dataset(root, seed=0):
    rng = np.random.RandomState(seed)
    counts = {"train": {"HS": 14, "NHS": 90}, "test": {"HS": 10, "NHS": 120}}
    for b in range(1, 6):
        for split, c in counts.items():
            for lab, n in c.items():
                d = os.path.join(root, "ICCAD-Data", f"ICCAD{b}", split, lab)
                os.makedirs(d)
                for i in range(n):
                    base = 200 if lab == "HS" else 120
                    arr = np.clip(rng.normal(base, 40, (60, 60)), 0, 255).astype(np.uint8)
                    Image.fromarray(arr, "L").save(os.path.join(d, f"{i}.png"))


class FakeKeras:
    """Stand-in for a Keras model: .predict(X, batch_size, verbose)."""
    def __init__(self, kind, rng):
        self.kind, self.W = kind, rng.normal(size=(48 * 48, 64)).astype(np.float32) / 48.0

    def predict(self, X, batch_size=256, verbose=0):
        f = X.reshape(len(X), -1) @ self.W
        if self.kind == "embed":
            return np.maximum(f, 0)
        z = (X.reshape(len(X), -1).mean(1) - 0.55) * 12
        return (1 / (1 + np.exp(-z))).reshape(-1, 1)


def fake_train_cnn(X_train, y_train, X_val, y_val, seed=42, verbose=0):
    rng = np.random.RandomState(seed)
    return FakeKeras("full", rng), FakeKeras("embed", rng), None


class _T:                       # tiny tensor stand-in with .numpy()
    def __init__(self, a): self.a = np.asarray(a)
    def numpy(self): return self.a


def test_latency_path():
    """Exercises measure_benchmark_latency with a fake `tensorflow` module (no real TF needed)."""
    import types
    from src import models as md
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    W = rng.normal(size=(48 * 48, 64)).astype(np.float32) / 48.0

    class FakeLayer:
        output = "embedding_tensor"

    class FakeFull:
        input, output = "in", "out"
        def get_layer(self, name):
            assert name == "embedding"
            return FakeLayer()
        def count_params(self): return 323169
        def __call__(self, x, training=False):
            return _T(1 / (1 + np.exp(-(x.reshape(len(x), -1).mean(1, keepdims=True) - 0.5))))

    class FakeJoint:
        def __call__(self, x, training=False):
            return _T(np.maximum(x.reshape(len(x), -1) @ W, 0)), _T(0.5 * np.ones((len(x), 1)))

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(Model=lambda inputs, outputs: FakeJoint())
    sys.modules["tensorflow"] = fake_tf
    try:
        X = rng.rand(30, 48, 48, 1).astype(np.float32)
        Xf = rng.normal(size=(200, 64)); y = (rng.rand(200) < 0.2).astype(int)
        classical = md.train_classical_base_models(Xf, y)
        meta = LogisticRegression(class_weight="balanced").fit(rng.rand(200, 4), y)
        rows = M.measure_benchmark_latency("ICCAD-X", FakeFull(), classical, meta, X, n=10)
        assert [r["Model"] for r in rows] == ["CNN_alone", "Ensemble_Stacking"]
        assert all(r["N"] == 10 and r["CNN_params"] == 323169 for r in rows)
    finally:
        del sys.modules["tensorflow"]


def main():
    tmp = tempfile.mkdtemp()
    try:
        make_fake_dataset(tmp)
        M.train_cnn = fake_train_cnn
        out = os.path.join(tmp, "out", "outputs")
        fake_root = os.path.join(tmp, "ICCAD-Data")

        # 1) the real-dataset guard must REJECT this demo data by default
        try:
            M.main(["--data_dir", fake_root, "--output_dir", out, "--benchmarks", "iccad1"])
            raise AssertionError("demo data was accepted - guard is broken")
        except ds.DatasetMismatchError as e:
            print("Guard OK - demo data rejected:", str(e)[:90], "...")

        # 2) explicit override runs the plumbing test
        M.main(["--data_dir", fake_root, "--output_dir", out, "--allow_mismatch",
                "--controls", "--save_predictions", "--benchmarks", "iccad1", "iccad2", "iccad3", "iccad4", "iccad5"])
        assert os.path.exists(os.path.join(tmp, "out", "g6_results.zip")), "results zip missing"

        # 3) automatic dataset location (env var) and download fallback
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            os.environ["ICCAD12_DIR"] = fake_root
            assert ds.locate_dataset(os.path.join(tmp, "nowhere")) == os.path.abspath(fake_root)
            del os.environ["ICCAD12_DIR"]
            assert ds.locate_dataset(os.path.join(tmp, "nowhere")) is None
            calls = []
            ds.download_iccad12_dataset = lambda wd: calls.append(wd) or "DOWNLOADED"
            assert ds.ensure_dataset(os.path.join(tmp, "nowhere")) == "DOWNLOADED" and calls
        finally:
            os.chdir(cwd)

        # 4) verify_dataset accepts exactly-official counts
        for b, ref in ds.REFERENCE_STATS.items():
            fake = {"train": {"hs": [0] * ref["train_hs"], "nhs": [0] * ref["train_nhs"]},
                    "test": {"hs": [0] * ref["test_hs"], "nhs": [0] * ref["test_nhs"]}}
            assert ds.verify_dataset(b, fake, strict=True) is True
        R = os.path.join(out, "results")
        per = pd.read_csv(os.path.join(R, "per_benchmark_results.csv"))
        assert set(per.Model) == {"Baseline_CNN", "cnn", "svm", "random_forest", "ann",
                                  "Ensemble_MajorityVote", "Ensemble_WeightedVote", "Ensemble_Stacking"}
        assert list(per.columns) == ["Benchmark", "Model", "TP", "FP", "FN", "TN", "Precision",
                                     "Recall_Sensitivity", "Specificity", "F1_Score", "Balanced_Accuracy"]
        assert (per[per.Model == "Baseline_CNN"].drop(columns="Model").values ==
                per[per.Model == "cnn"].drop(columns="Model").values).all(), "baseline must equal CNN member"
        assert (per.TP + per.FN).groupby(per.Benchmark).nunique().max() == 1          # same test set for every model
        abl = pd.read_csv(os.path.join(R, "ablation_results.csv"))
        assert len(abl) == 25 and "(none - full ensemble)" in set(abl.Removed_Model)
        assert os.path.exists(os.path.join(R, "control_results.csv"))
        assert os.path.exists(os.path.join(R, "run_config.json"))
        assert os.path.exists(os.path.join(R, "predictions_ICCAD-1.npz"))
        assert len(pd.read_csv(os.path.join(R, "average_results.csv"))) == 8
        assert os.path.exists(os.path.join(out, "figures", "ICCAD-3_best_ensemble_cm.png"))
        assert os.path.exists(os.path.join(out, "figures", "baseline_vs_ensemble_comparison.png"))

        # metric / combiner unit checks
        m = ut.compute_metrics(np.array([1, 1, 0, 0, 0, 0]), np.array([1, 0, 1, 0, 0, 0]))
        assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 3)
        assert abs(m["Balanced_Accuracy"] - (0.5 + 0.75) / 2) < 1e-12
        P = np.array([[0.9, 0.9, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.1]])
        assert ens.majority_vote(P).tolist() == [1, 0, 1], "2-2 tie must be HS"
        lat = ut.measure_latency_ms(lambda x: x.sum(), np.zeros((50, 4, 4, 1)), n=20, warmup=2)
        assert lat["N"] == 20 and lat["Mean_ms"] >= 0
        test_latency_path()
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
