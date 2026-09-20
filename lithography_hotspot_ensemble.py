"""One-file launcher (Kaggle / Colab / local): runs the full pipeline on the REAL ICCAD-12 dataset.

    python lithography_hotspot_ensemble.py            # same as: python main.py

It finds (or downloads) the official dataset, verifies the image counts, trains the CNN baseline and the
ensembles on ICCAD-1..5 and writes results, figures and g6_results.zip. Extra options are passed through,
e.g.  python lithography_hotspot_ensemble.py --benchmarks iccad1 iccad5 --controls
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import main  # noqa: E402

if __name__ == "__main__":
    main()
