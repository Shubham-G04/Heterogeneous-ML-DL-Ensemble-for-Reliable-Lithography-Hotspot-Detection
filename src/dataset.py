import os
import sys
import gc
import zipfile
import subprocess
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

REFERENCE_STATS = {
    "iccad1": {"train_hs": 99, "train_nhs": 340, "test_hs": 226, "test_nhs": 3869},
    "iccad2": {"train_hs": 174, "train_nhs": 5285, "test_hs": 498, "test_nhs": 41298},
    "iccad3": {"train_hs": 909, "train_nhs": 4643, "test_hs": 1808, "test_nhs": 46333},
    "iccad4": {"train_hs": 95, "train_nhs": 4452, "test_hs": 177, "test_nhs": 31890},
    "iccad5": {"train_hs": 26, "train_nhs": 2716, "test_hs": 41, "test_nhs": 19327},
}

NHS_KEYWORDS = ["nhs", "non_hotspot", "nonhotspot", "non-hotspot", "negative", "non_hs", "non-hs", "non"]
HS_KEYWORDS = ["hotspot", "hot_spot", "hot-spot", "hs", "positive", "pos"]
TRAIN_KEYWORDS = ["train", "training"]
TEST_KEYWORDS = ["test", "testing"]


def download_iccad12_dataset(work_dir, file_id="1jx7gDR92sqoIw2Nh4NGwwZwC-qNp2osd"):
    """Downloads the ICCAD-12 dataset from Google Drive via gdown if not already present."""
    import gdown

    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, "iccad12_dataset.archive")
    extract_dir = os.path.join(work_dir, "iccad12_dataset")

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 10_000:
        print("Downloading ICCAD-12 dataset ...")
        gdown.download(id=file_id, output=zip_path, quiet=False)
    else:
        print("Archive already present, skipping download.")

    with open(zip_path, "rb") as f:
        header = f.read(8)

    if not os.path.isdir(extract_dir) or len(os.listdir(extract_dir)) == 0:
        extract_archive(zip_path, extract_dir, header)
    else:
        print("Already extracted, skipping extraction.")

    return extract_dir


def extract_archive(archive_path, extract_dir, header):
    """Extracts zip/rar archive formats."""
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
        result = subprocess.run(["unrar", "x", "-y", archive_path, extract_dir + "/"], capture_output=True, text=True)
        if result.returncode != 0:
            print("unrar failed, trying 7z fallback ...")
            subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=False)
            result2 = subprocess.run(["7z", "x", f"-o{extract_dir}", "-y", archive_path], capture_output=True, text=True)
            if result2.returncode != 0:
                raise RuntimeError("Extraction failed:\n" + result.stderr + "\n" + result2.stderr)
        return

    raise ValueError(f"Unrecognized archive format (header bytes: {header}).")


def _match_any(segment, keywords):
    seg = segment.lower()
    return any(k in seg for k in keywords)


def find_benchmark_root(extract_dir, bench_name):
    """Locates target benchmark folder (e.g. 'iccad1')."""
    target = bench_name.lower().replace("-", "").replace("_", "")
    candidates = []
    for dirpath, dirnames, _ in os.walk(extract_dir):
        for d in dirnames:
            norm = d.lower().replace("-", "").replace("_", "")
            if norm == target:
                candidates.append(os.path.join(dirpath, d))
    if not candidates:
        raise FileNotFoundError(f"Could not find folder matching '{bench_name}' under {extract_dir}")
    candidates.sort(key=lambda p: p.count(os.sep))
    return candidates[0]


def discover_image_files(bench_root):
    """Walks folder tree to categorize image files into split and label."""
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
                split = "train"
                break
            if _match_any(part, TEST_KEYWORDS):
                split = "test"
                break

        label = None
        for part in rel_parts:
            if _match_any(part, NHS_KEYWORDS):
                label = "nhs"
                break
        if label is None:
            for part in rel_parts:
                if _match_any(part, HS_KEYWORDS):
                    label = "hs"
                    break

        if split is None or label is None:
            unclassified.append((dirpath, len(imgs)))
            continue

        result[split][label].extend(os.path.join(dirpath, f) for f in imgs)

    if unclassified:
        print("  WARNING: unclassified folders skipped:")
        for d, n in unclassified[:10]:
            print(f"    {d}  ({n} images)")

    return result


def sanity_check(bench_name, discovered):
    """Sanity checks image counts against reference stats."""
    ref = REFERENCE_STATS.get(bench_name)
    found = {
        "train_hs": len(discovered["train"]["hs"]),
        "train_nhs": len(discovered["train"]["nhs"]),
        "test_hs": len(discovered["test"]["hs"]),
        "test_nhs": len(discovered["test"]["nhs"]),
    }
    print(f"  Found:     {found}")
    if ref:
        if all(found[k] == ref[k] for k in ref):
            print("  MATCH -- folder discovery is correct.")
        else:
            print("  MISMATCH -- check keyword rules.")
    return found


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


def build_dataset(discovered, img_size=48, channels=1, val_split=0.15, seed=42, verbose=True):
    """Loads images and builds train/val/test splits."""
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
        X_train_full, y_train_full, test_size=val_split, stratify=y_train_full, random_state=seed
    )
    del X_train_full, y_train_full
    gc.collect()

    if verbose:
        print(
            f"  train={len(X_train)} (HS={int(y_train.sum())}) | "
            f"val={len(X_val)} (HS={int(y_val.sum())}) | "
            f"test={len(X_test)} (HS={int(y_test.sum())}, NHS={int((y_test == 0).sum())})"
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
