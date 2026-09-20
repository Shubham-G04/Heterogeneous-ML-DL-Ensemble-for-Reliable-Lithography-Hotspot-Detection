"""ICCAD-12 dataset: download, folder discovery, preprocessing and splitting.

Pipeline (identical for every model, baseline included):
  * grayscale, bilinear resize to IMG_SIZE x IMG_SIZE, scaled to [0, 1]
  * official train / test partitions of each benchmark (benchmarks never merged)
  * 15 % stratified hold-out of the *training* partition = validation set
    (used for CNN early stopping, weighted-vote weights and the stacking meta-learner)
  * the test partition is used only for final scoring
"""
import gc
import os
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------- config
FILE_ID = "1jx7gDR92sqoIw2Nh4NGwwZwC-qNp2osd"   # Google-Drive id of the dataset used by the reference work
IMG_SIZE = 48
IMG_CHANNELS = 1
VAL_SPLIT = 0.15
RANDOM_SEED = 42
BENCHMARK_NAMES = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

# Expected image counts. Training counts and ICCAD-2..5 test counts are from the reference
# work's Fig. 3. It lists 3,869 test NHS clips for ICCAD-1, but the confusion matrices of
# our runs sum to 4,679, so 4,679 is used here (the 3,869 looks like a typo in the reference).
REFERENCE_STATS = {
    "iccad1": {"train_hs": 99,  "train_nhs": 340,  "test_hs": 226,  "test_nhs": 4679},
    "iccad2": {"train_hs": 174, "train_nhs": 5285, "test_hs": 498,  "test_nhs": 41298},
    "iccad3": {"train_hs": 909, "train_nhs": 4643, "test_hs": 1808, "test_nhs": 46333},
    "iccad4": {"train_hs": 95,  "train_nhs": 4452, "test_hs": 177,  "test_nhs": 31890},
    "iccad5": {"train_hs": 26,  "train_nhs": 2716, "test_hs": 41,   "test_nhs": 19327},
}

# NHS keywords are checked FIRST because "hs" is a substring of "nhs".
NHS_KEYWORDS = ["nhs", "non_hotspot", "nonhotspot", "non-hotspot", "negative", "non_hs", "non-hs", "non"]
HS_KEYWORDS = ["hotspot", "hot_spot", "hot-spot", "hs", "positive", "pos"]
TRAIN_KEYWORDS = ["train", "training"]
TEST_KEYWORDS = ["test", "testing"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ----------------------------------------------------------------------------- download
def _extract_archive(archive_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with open(archive_path, "rb") as f:
        header = f.read(8)
    if header.startswith(b"PK"):
        print("Detected ZIP archive, extracting ...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return
    if header.startswith(b"Rar!"):
        print("Detected RAR archive, extracting ...")
        if subprocess.run(["which", "unrar"], capture_output=True).returncode != 0:
            subprocess.run(["apt-get", "update", "-qq"], check=False)
            subprocess.run(["apt-get", "install", "-y", "-qq", "unrar"], check=False)
        res = subprocess.run(["unrar", "x", "-y", archive_path, extract_dir + "/"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print("unrar failed, trying 7z ...")
            subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=False)
            res2 = subprocess.run(["7z", "x", f"-o{extract_dir}", "-y", archive_path],
                                  capture_output=True, text=True)
            if res2.returncode != 0:
                raise RuntimeError("Both unrar and 7z failed:\n" + res.stderr + "\n" + res2.stderr)
        return
    raise ValueError(f"Unrecognised archive format (header {header!r}). The download may be a "
                     "Google-Drive quota/HTML page - download the dataset manually and use --data_dir.")


def download_iccad12_dataset(work_dir):
    """Download (gdown) and extract the dataset under work_dir. Returns the extraction folder."""
    os.makedirs(work_dir, exist_ok=True)
    archive = os.path.join(work_dir, "iccad12_dataset.archive")
    extract_dir = os.path.join(work_dir, "iccad12_dataset")
    if not os.path.exists(archive) or os.path.getsize(archive) < 10_000:
        manual = ("Download the ICCAD-12 dataset manually from "
                  f"https://drive.google.com/file/d/{FILE_ID}/view, extract it, then run: "
                  "python main.py --data_dir <extracted folder>  (or set ICCAD12_DIR).")
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
            try:
                import gdown
            except ImportError as e:
                raise RuntimeError("Could not import/install gdown (no internet?). " + manual) from e
        print("Downloading ICCAD-12 dataset ...")
        gdown.download(id=FILE_ID, output=archive, quiet=False)
        if not os.path.exists(archive) or os.path.getsize(archive) < 10_000:
            raise RuntimeError("Dataset download failed (Google-Drive quota or no internet). " + manual)
    else:
        print("Archive already present, skipping download.")
    if not os.path.isdir(extract_dir) or len(os.listdir(extract_dir)) == 0:
        _extract_archive(archive, extract_dir)
    else:
        print("Already extracted, skipping.")
    return extract_dir


# ----------------------------------------------------------------------------- environment / auto-location
class DatasetMismatchError(RuntimeError):
    """Raised when the folder found is not the official ICCAD-12 data (e.g. a demo / wrong dataset)."""


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def default_work_dir():
    """Where the dataset is downloaded: Kaggle -> /kaggle/working, Colab -> /content, else ./data."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    return "/content" if in_colab() else "./data"


def default_output_dir():
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/outputs"
    return "/content/outputs" if in_colab() else "./outputs"


def _candidate_roots(work_dir, explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("ICCAD12_DIR"):
        cands.append(os.environ["ICCAD12_DIR"])
    cands += [os.path.join(work_dir, "iccad12_dataset"), work_dir,
              "./data/iccad12_dataset", "./iccad12_dataset", "./data",
              "/content/iccad12_dataset", "/kaggle/working/iccad12_dataset"]
    if os.path.isdir("/kaggle/input"):                       # datasets attached to a Kaggle notebook
        cands += [os.path.join("/kaggle/input", d) for d in sorted(os.listdir("/kaggle/input"))]
    seen, out = set(), []
    for c in cands:
        ap = os.path.abspath(c)
        if ap not in seen and os.path.isdir(ap):
            seen.add(ap); out.append(ap)
    return out


def _has_benchmarks(root, benchmarks):
    try:
        for b in benchmarks:
            r = find_benchmark_root(root, b)
            if not any(f.lower().endswith(IMAGE_EXTS) for _, _, fs in os.walk(r) for f in fs):
                return False
        return True
    except FileNotFoundError:
        return False


def locate_dataset(work_dir, benchmarks=BENCHMARK_NAMES, explicit=None):
    """Return the first candidate folder that contains every requested benchmark with images, else None."""
    for root in _candidate_roots(work_dir, explicit):
        if _has_benchmarks(root, benchmarks):
            return root
    return None


def ensure_dataset(work_dir, benchmarks=BENCHMARK_NAMES, explicit=None):
    """Use an already-present ICCAD-12 folder if one is found automatically, otherwise download it."""
    root = locate_dataset(work_dir, benchmarks, explicit)
    if root:
        print(f"Using existing ICCAD-12 dataset at: {root}")
        return root
    if explicit:
        raise FileNotFoundError(f"--data_dir {explicit!r} does not contain benchmark folders "
                                f"{list(benchmarks)} with images.")
    print("No local ICCAD-12 dataset found - downloading the official one ...")
    return download_iccad12_dataset(work_dir)


def verify_dataset(bench_name, discovered, strict=True):
    """Guard against running on the wrong / demo data. Test-set counts must match the official ICCAD-12
    counts exactly (they are confirmed by the reported confusion matrices); a training-count deviation
    only produces a warning. Returns True if everything matches."""
    ref = REFERENCE_STATS[bench_name]
    found = {f"{s}_{l}": len(discovered[s][l]) for s in ("train", "test") for l in ("hs", "nhs")}
    bad_test = {k: (found[k], ref[k]) for k in ("test_hs", "test_nhs") if found[k] != ref[k]}
    bad_train = {k: (found[k], ref[k]) for k in ("train_hs", "train_nhs") if found[k] != ref[k]}
    if bad_test and strict:
        raise DatasetMismatchError(
            f"{bench_name}: test-set counts (found, expected) {bad_test} do not match the official ICCAD-12 "
            "benchmark, so this does not look like the real dataset. Fix the dataset location, or pass "
            "--allow_mismatch to run anyway.")
    if bad_test or bad_train:
        print(f"  WARNING {bench_name}: counts differ from the reference (found, expected): "
              f"{ {**bad_train, **bad_test} }")
        return False
    print(f"  {bench_name}: counts match the official ICCAD-12 benchmark.")
    return True


# ----------------------------------------------------------------------------- discovery
def _match_any(segment, keywords):
    seg = segment.lower()
    return any(k in seg for k in keywords)


def find_benchmark_root(extract_dir, bench_name):
    """Shallowest folder under extract_dir whose name matches bench_name ('iccad1'),
    ignoring case and '-'/'_' separators."""
    target = bench_name.lower().replace("-", "").replace("_", "")
    candidates = []
    for dirpath, dirnames, _ in os.walk(extract_dir):
        for d in dirnames:
            if d.lower().replace("-", "").replace("_", "") == target:
                candidates.append(os.path.join(dirpath, d))
    if not candidates:
        raise FileNotFoundError(f"No folder matching '{bench_name}' under {extract_dir}")
    candidates.sort(key=lambda p: p.count(os.sep))
    return candidates[0]


def discover_image_files(bench_root):
    """Classify every image under bench_root into (split, label) from folder-name keywords.
    Returns {'train': {'hs': [...], 'nhs': [...]}, 'test': {'hs': [...], 'nhs': [...]}}."""
    result = {"train": {"hs": [], "nhs": []}, "test": {"hs": [], "nhs": []}}
    unclassified = []
    for dirpath, _, filenames in os.walk(bench_root):
        imgs = [f for f in filenames if f.lower().endswith(IMAGE_EXTS)]
        if not imgs:
            continue
        parts = os.path.relpath(dirpath, bench_root).split(os.sep)
        split = None
        for part in parts:
            if _match_any(part, TRAIN_KEYWORDS):
                split = "train"; break
            if _match_any(part, TEST_KEYWORDS):
                split = "test"; break
        label = None
        for part in parts:
            if _match_any(part, NHS_KEYWORDS):
                label = "nhs"; break
        if label is None:
            for part in parts:
                if _match_any(part, HS_KEYWORDS):
                    label = "hs"; break
        if split is None or label is None:
            unclassified.append((dirpath, len(imgs)))
            continue
        result[split][label].extend(os.path.join(dirpath, f) for f in sorted(imgs))
    if unclassified:
        print("  WARNING: could not classify these folders (skipped):")
        for d, n in unclassified[:20]:
            print(f"    {d} ({n} images)")
    return result


def sanity_check(bench_name, discovered):
    """Compare discovered image counts with REFERENCE_STATS; prints MATCH / MISMATCH."""
    found = {f"{s}_{l}": len(discovered[s][l]) for s in ("train", "test") for l in ("hs", "nhs")}
    print(f"  Found:     {found}")
    ref = REFERENCE_STATS.get(bench_name)
    if ref:
        print(f"  Reference: {ref}")
        print("  MATCH" if all(found[k] == ref[k] for k in ref)
              else "  MISMATCH - check the warnings above / keyword lists")
    return found


# ----------------------------------------------------------------------------- arrays
def _load_image_list(paths, img_size, channels):
    if len(paths) == 0:
        return np.zeros((0, img_size, img_size, channels), dtype=np.float32)
    mode = "L" if channels == 1 else "RGB"
    out = np.zeros((len(paths), img_size, img_size, channels), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert(mode).resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        out[i] = arr[..., np.newaxis] if channels == 1 else arr
    return out


def build_dataset(discovered, img_size=IMG_SIZE, channels=IMG_CHANNELS,
                  val_split=VAL_SPLIT, seed=RANDOM_SEED, verbose=True):
    """Load one benchmark. Label 1 = hotspot (HS), 0 = non-hotspot (NHS)."""
    tr_hs = _load_image_list(discovered["train"]["hs"], img_size, channels)
    tr_nhs = _load_image_list(discovered["train"]["nhs"], img_size, channels)
    te_hs = _load_image_list(discovered["test"]["hs"], img_size, channels)
    te_nhs = _load_image_list(discovered["test"]["nhs"], img_size, channels)

    X_full = np.concatenate([tr_hs, tr_nhs], axis=0)
    y_full = np.concatenate([np.ones(len(tr_hs)), np.zeros(len(tr_nhs))]).astype(np.int32)
    X_test = np.concatenate([te_hs, te_nhs], axis=0)
    y_test = np.concatenate([np.ones(len(te_hs)), np.zeros(len(te_nhs))]).astype(np.int32)
    del tr_hs, tr_nhs, te_hs, te_nhs
    gc.collect()

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_split, stratify=y_full, random_state=seed)
    del X_full, y_full
    gc.collect()

    if verbose:
        print(f"  train={len(X_train)} (HS={int(y_train.sum())}) | val={len(X_val)} (HS={int(y_val.sum())}) | "
              f"test={len(X_test)} (HS={int(y_test.sum())}, NHS={int((y_test == 0).sum())})")
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}
