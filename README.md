# Heterogeneous ML–DL Ensemble for Reliable Lithography Hotspot Detection on ICCAD-12 Benchmarks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)

This repository contains the official code, paper documentation, and benchmark results for **"Heterogeneous ML–DL Ensemble for Reliable Lithography Hotspot Detection on the ICCAD-12 Benchmarks"**.

---

## 📌 Abstract

Lithography hotspots are layout patterns that print with open or short defects during silicon fabrication. Finding them early in the physical design flow is critical because rigorous lithography simulation is computationally expensive for full-chip screening. A 12,873-parameter convolutional neural network (CNN) trained from scratch was recently shown to outperform deep transfer-learning pipelines on the ICCAD-12 benchmarks. 

This work investigates whether combining heterogeneous machine learning (ML) and deep learning (DL) models can make hotspot detection more reliable than a standalone CNN. We construct heterogeneous ensembles consisting of a **CNN**, a **Support Vector Machine (SVM)**, a **Random Forest (RF)**, and an **Artificial Neural Network (ANN)** using three combination strategies:
1. **Majority Voting (Hard Voting)**
2. **Weighted Soft Voting**
3. **Stacking Meta-Learner (Logistic Regression)**

Evaluating on all five official ICCAD-12 benchmarks under identical experimental splits, **Stacking achieves the best overall performance**:
- **Balanced Accuracy (BA)** increases from **92.09% to 92.97%**.
- **Missed Hotspots (False Negatives)** pooled across test sets drop from **234 to 148 (−37%)** while maintaining an uncompromised false-alarm rate (3,502 vs 3,500).
- Leave-one-out ablation confirms that the **CNN is the most critical model**, while fixed voting strategies over-constrain model sensitivity.

---

## 🏗️ Model Architecture & Ensemble Overview

```
                        ┌──────────────────────────────────────────┐
                        │   ICCAD-12 Layout Clip (48x48 Grayscale) │
                        └────────────────────┬─────────────────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │   Lightweight CNN   │ (12,873 params)
                                  └──────────┬──────────┘
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     │                       │                       │
           [Penultimate 64-D Embedding]      │                       │
                     │                       │                       │
         ┌───────────┼───────────┐           │                       │
         ▼           ▼           ▼           ▼                       │
     ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐                   │
     │  SVM  │   │   RF  │   │  ANN  │   │  CNN  │                   │
     └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘                   │
         │           │           │           │                       │
         └───────────┼───────────┼───────────┘                       │
                     │ (Probabilities)                               │
                     ▼                                               │
   ┌───────────────────────────────────┐                             │
   │      Ensemble Combiners            │                             │
   │  • Majority Hard Voting           │                             │
   │  • Weighted Soft Voting           │                             │
   │  • Stacking Meta-Learner (Logistic)│ ◄── Trained on Val Split   │
   └─────────────────┬─────────────────┘                             │
                     │                                               │
                     ▼                                               │
       [Final Hotspot / Non-Hotspot Prediction] ─────────────────────┘
```

---

## 📊 ICCAD-12 Benchmark Suite Statistics

The official ICCAD-2012 CAD Contest suite consists of five highly imbalanced benchmarks:

| Benchmark | Train HS | Train NHS | Test HS | Test NHS | NHS:HS Ratio (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ICCAD-1** | 99 | 340 | 226 | 3,869 | 17.1 : 1 |
| **ICCAD-2** | 174 | 5,285 | 498 | 41,298 | 82.9 : 1 |
| **ICCAD-3** | 909 | 4,643 | 1,808 | 46,333 | 25.6 : 1 |
| **ICCAD-4** | 95 | 4,452 | 177 | 31,890 | 180.2 : 1 |
| **ICCAD-5** | 26 | 2,716 | 41 | 19,327 | 471.4 : 1 |

---

## 📈 Experimental Results

### 1. Balanced Accuracy (%) per Benchmark

| Benchmark | Baseline CNN | SVM | Random Forest | ANN | Majority Vote | Weighted Vote | **Stacking (Best)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ICCAD-1** | 82.83% | 86.60% | 85.54% | 83.59% | 83.77% | 86.18% | **86.11%** |
| **ICCAD-2** | 90.79% | 87.89% | 87.21% | 86.09% | 90.78% | 88.90% | **92.66%** |
| **ICCAD-3** | 95.71% | 97.20% | 97.05% | 96.95% | 97.11% | 97.18% | **97.10%** |
| **ICCAD-4** | 93.73% | 86.34% | 82.57% | 79.22% | 88.06% | 85.58% | **92.19%** |
| **ICCAD-5** | 97.42% | 86.26% | 88.71% | 50.00% | 89.86% | 92.29% | **96.80%** |
| **AVERAGE** | **92.09%** | **88.86%** | **88.22%** | **79.17%** | **89.92%** | **90.03%** | **92.97%** |

### 2. Five-Benchmark Macro Average Performance Metrics (%)

| Model | Balanced Acc (BA) | Precision | Recall (Sensitivity) | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 92.09% | 37.72% | 92.28% | 91.91% | 46.75% |
| **SVM** | 88.86% | 39.63% | 84.05% | 93.67% | 49.94% |
| **Random Forest** | 88.22% | 42.58% | 82.86% | 93.57% | 51.66% |
| **ANN** | 79.17% | 37.48% | 65.76% | 92.57% | 43.36% |
| **Majority Vote** | 89.92% | 39.12% | 87.35% | 92.48% | 49.93% |
| **Weighted Vote** | 90.03% | 41.22% | 86.50% | 93.55% | 51.36% |
| **Stacking (Ours)** | **92.97%** | **37.26%** | **92.69%** | **93.25%** | **48.59%** |

---

## 🖼️ Visualizations & Figures

### Baseline CNN vs. Ensemble Performance Comparison
![Baseline vs Ensemble Bar Chart](figures/baseline_vs_ensemble_comparison.png)

### Sample Confusion Matrices (ICCAD-1 & ICCAD-3)
| Benchmark | Baseline CNN | Stacking Ensemble |
| :---: | :---: | :---: |
| **ICCAD-1** | ![ICCAD-1 Baseline](figures/ICCAD-1_baseline_cnn_cm.png) | ![ICCAD-1 Stacking](figures/ICCAD-1_best_ensemble_cm.png) |
| **ICCAD-3** | ![ICCAD-3 Baseline](figures/ICCAD-3_baseline_cnn_cm.png) | ![ICCAD-3 Stacking](figures/ICCAD-3_best_ensemble_cm.png) |

---

## 📂 Repository Structure

```
Heterogeneous-ML-DL-Ensemble-for-Reliable-Lithography-Hotspot-Detection/
├── README.md                          # Main project documentation & overview
├── LICENSE                            # MIT License
├── requirements.txt                   # Project Python dependencies
├── .gitignore                         # Git ignore configuration
├── main.py                            # CLI entry script to run benchmark evaluations
├── lithography_hotspot_ensemble.py     # Standalone Python script (Colab/Kaggle friendly)
├── lithography_hotspot_ensemble.ipynb  # Interactive Jupyter Notebook for Kaggle / Colab
├── src/                               # Modular Python source package
│   ├── __init__.py
│   ├── dataset.py                     # ICCAD-12 dataset loader & preprocessor
│   ├── models.py                      # CNN architecture & classical base learners
│   ├── ensemble.py                    # Voting & Stacking meta-learner strategies
│   └── utils.py                       # Metric evaluation & plotting utilities
├── paper/                             # Research paper documentation
│   ├── G6_Hotspot_Ensemble_IEEE_Paper.docx
│   └── IEEE_Paper.md                  # Markdown version of full paper
├── figures/                           # Generated evaluation plots & confusion matrices
│   ├── baseline_vs_ensemble_comparison.png
│   ├── ICCAD-1_baseline_cnn_cm.png
│   └── ...
└── results/                           # Quantitative output metrics (CSV format)
    ├── average_results.csv
    ├── per_benchmark_results.csv
    ├── ablation_results.csv
    └── ablation_average.csv
```

---

## 🚀 Quickstart & Usage

### Option 1: Local Command Line (CLI)

```bash
# Clone repository
git clone https://github.com/Shubham-G04/Heterogeneous-ML-DL-Ensemble-for-Reliable-Lithography-Hotspot-Detection.git
cd Heterogeneous-ML-DL-Ensemble-for-Reliable-Lithography-Hotspot-Detection

# Install dependencies
pip install -r requirements.txt

# Execute full evaluation across ICCAD 1-5 benchmarks
python main.py --work_dir ./data --output_dir ./outputs
```

### Option 2: Google Colab / Kaggle Notebook
1. Open `lithography_hotspot_ensemble.ipynb` in Google Colab or upload to Kaggle Notebooks.
2. Select a GPU runtime (e.g. NVIDIA T4 / P100).
3. Run all cells. The dataset will automatically download from Google Drive via `gdown` and run the full ensemble pipeline.

---

## 👥 Authors & Acknowledgments

**Group 6 · Digital Assignment II**  
*BEVD402L: AI and Machine Learning for IC Design*  
**Vellore Institute of Technology (VIT), Chennai, India**

- **Subham Prasad Gupta** (Reg. No. 23BVD1039) – Voting & Stacking Ensembles, Ablation & Error Analysis
- **Akhil Kumar Kudipudi** (Reg. No. 23BVD1042) – Literature Survey, ICCAD-12 Data Analysis
- **Tejasvini R** (Reg. No. 23BVD1044) – Baseline CNN Implementation, SVM/RF/ANN Members

**Faculty Advisor:** Dr. G. Lakshmi Priya (VIT Chennai)

---

## 📄 Citation

If you find this work or codebase useful in your research, please cite our project paper:

```bibtex
@article{gupta2026heterogeneous,
  title={Heterogeneous ML--DL Ensemble for Reliable Lithography Hotspot Detection on the ICCAD-12 Benchmarks},
  author={Gupta, Subham Prasad and Kudipudi, Akhil Kumar and Tejasvini, R},
  journal={Course Project Report, BEVD402L: AI and Machine Learning for IC, VIT Chennai},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
