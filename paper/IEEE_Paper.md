# Heterogeneous ML–DL Ensemble for Reliable Lithography Hotspot Detection on the ICCAD-12 Benchmarks

Heterogeneous ML–DL Ensemble for Reliable Lithography Hotspot Detection on the ICCAD-12 Benchmarks

Subham Prasad Gupta

Reg. No. 23BVD1039


### B.Tech. (BVD)


Vellore Institute of Technology, Chennai, India

Akhil Kumar Kudipudi

Reg. No. 23BVD1042


### B.Tech. (BVD)


Vellore Institute of Technology, Chennai, India

Tejasvini R

Reg. No. 23BVD1044


### B.Tech. (BVD)


Vellore Institute of Technology, Chennai, India

Group 6 · Digital Assignment II · BEVD402L: AI and Machine Learning for IC · Faculty: Dr. G. Lakshmi Priya


## Abstract—Lithography hotspots are layout patterns that print with open or short defects, and they must be found early in the design flow because rigorous lithography simulation is too slow for full-chip screening. A 12,873-parameter CNN trained from scratch was recently shown to beat VGG16 transfer-learning pipelines on the ICCAD-12 benchmarks. This work asks whether combining heterogeneous learners makes hotspot detection more reliable than that single CNN. We build ensembles of a CNN, an SVM, a Random Forest and an ANN using majority voting, weighted voting and stacking, and compare them with the CNN baseline on all five ICCAD-12 benchmarks under identical conditions, reporting balanced accuracy (BA), precision, recall, specificity, F1 and confusion matrices. Stacking is the best configuration: average BA rises from 92.09% to 92.97%, and the number of missed hotspots pooled over the five test sets falls from 234 to 148 (−37%) at an unchanged false-alarm count (3,502 vs 3,500). Majority and weighted voting do not improve BA (89.92% and 90.03%) but raise precision and F1 at the cost of recall. A leave-one-out ablation shows the CNN is the most valuable member, whereas the ANN, which collapses to predicting only non-hotspots on ICCAD-5, slightly hurts the weighted-vote ensemble. The gains are not uniform: stacking trails the baseline on the two most imbalanced benchmarks, and the average improvement is of the same order as the sampling uncertainty of a single run.



## Index Terms—lithography hotspot detection, ICCAD-12, ensemble learning, stacking, class imbalance, balanced accuracy, convolutional neural network.



## I. INTRODUCTION


Photolithography transfers a circuit layout onto silicon, but as feature sizes shrink relative to the exposure wavelength, diffraction distorts the printed pattern. Layout patterns that print with pinching (open-circuit) or bridging (short-circuit) defects are called lithography hotspots [1]. Rigorous lithography simulation finds them accurately but is far too slow for full-chip screening, which motivated fast pattern matching [2] and machine-learning detectors [3], [4], [5], [6], [7]. The ICCAD-2012 contest suite [8] has become the standard testbed. Its five benchmarks are extremely imbalanced—hotspots are rare—so plain accuracy is misleading and balanced accuracy (BA) together with confusion-matrix measures must be used [9].

The reference work supplied for this assignment [10] compares two routes on ICCAD-12: (i) features from a pre-trained VGG16 [11] fed to classical classifiers [12], [1], whose best variant (a linear SVM) reaches 80.94% average BA at 182 ms per clip, and (ii) a 12,873-parameter CNN trained from scratch [13] that reaches 95.3% average validation BA at 6.3 ms per clip. It concludes that the lightweight CNN is preferable. However, the models are treated only as competitors, they are never combined, and results are shown mainly as BA bar charts, without per-benchmark confusion matrices, precision or F1, so the trade-off between missed hotspots and false alarms cannot be judged.

Research gap. A CNN and classical learners such as SVM or Random Forest differ in inductive bias and may therefore make different mistakes, but this complementarity is left unexplored on ICCAD-12. Research question. Can combining predictions from heterogeneous ML/DL models (CNN, SVM, Random Forest, ANN) improve hotspot-detection reliability compared with the individual CNN baseline, and which error patterns does the combination reduce? We test two hypotheses: H1—a learned combination (stacking) achieves higher average BA than the best single model; H2—the ensemble reduces missed hotspots (false negatives) without a proportional rise in false alarms. The contributions are:

three combination strategies (majority vote, weighted vote, stacking) over four heterogeneous members, compared with the common baseline on all five benchmarks under identical conditions;

per-benchmark results with full confusion matrices and an analysis of which error type (false negative or false positive) each combination reduces;

a leave-one-out ablation quantifying each member’s contribution; and

an explicit account of where the ensemble does not help and of the statistical uncertainty of the comparison.


## II. BACKGROUND AND RELATED WORK



### A. Hotspot Detection


Early detectors rely on pattern matching against a library of known hotspots, which cannot flag unseen patterns [2]. Machine-learning methods replace the library with learned classifiers over engineered features, e.g., topological classification with critical-feature extraction [3], AdaBoost with simplified features [4], and information-theoretic feature optimisation with online learning [5]. Deep learning moves feature extraction into the network: imbalance-aware CNN training [6], feature-tensor inputs with biased learning [7], and comparisons from shallow to deep models [1]. Transfer learning from ImageNet-pretrained VGG16 [11] was studied in [12], whereas [13] showed that a very small CNN trained from scratch is sufficient; this CNN is the common baseline of [10] and of this work.


### B. Ensemble Learning


An ensemble improves reliability when its members are individually competent and make diverse errors [14]. Voting fixes the combination rule in advance, whereas stacked generalisation trains a meta-learner on the members’ held-out outputs so that it learns how far to trust each member [15]; a survey is given in [16]. Because all members here are trained on the same imbalanced data, we evaluate every combination with BA rather than accuracy.


## III. DATASET AND CLASS-IMBALANCE ANALYSIS



### A. ICCAD-12 Statistics


ICCAD-12 provides five benchmarks, each with a training and a test partition of hotspot (HS) and non-hotspot (NHS) layout clips [8]. Table I lists the counts. The benchmarks are treated separately throughout, as required, and are never merged. Imbalance grows from ICCAD-1 to ICCAD-5: the NHS:HS ratio in the training data rises from 3.4:1 to 104.5:1, and in every benchmark the test set is more imbalanced than the training set (20.7:1 to 471.4:1). ICCAD-5 has only 26 training and 41 test hotspots, so a single hotspot moves its recall by 2.4 percentage points (pp).


#### TABLE I


ICCAD-12 class distribution

Bench.

Train HS

Train NHS

Ratio

Test HS

Test NHS

Ratio

B1

99

340

3.4:1

226

4,679*

20.7:1

B2

174

5,285

30.4:1

498

41,298

82.9:1

B3

909

4,643

5.1:1

1,808

46,333

25.6:1

B4

95

4,452

46.9:1

177

31,890

180.2:1

B5

26

2,716

104.5:1

41

19,327

471.4:1

Training counts as listed in [10]; test counts are TP+FN and FP+TN of our confusion matrices. *[10] lists 3,869 NHS test clips for ICCAD-1, whereas our pipeline yields 4,679; all ICCAD-1 metrics here use our own counts.


### B. Evaluation Metrics and Why Accuracy Fails


With TP, FP, FN and TN taken from the confusion matrix (HS is the positive class), we use precision P = TP/(TP+FP), recall (sensitivity) R = TP/(TP+FN), specificity S = TN/(TN+FP), F1 = 2PR/(P+R) and the primary metric

(1)

Accuracy is uninformative here. On ICCAD-5 the ANN member predicts every clip as NHS and still obtains 99.79% accuracy, but its BA is exactly 50%. Precision is also intrinsically low for any detector on such data: with hotspot prevalence π,

(2)

so for ICCAD-5 (π = 0.21%) even R = 95.1% and S = 98.5% yield only P = 11.7%. We therefore interpret precision and F1 relative to prevalence and rank models primarily by BA, recall and specificity.


## IV. PROPOSED METHODOLOGY



### A. Baseline: Lightweight CNN


The common baseline is the lightweight CNN of [13] as adopted in [10]: two “basic blocks” separated by 5×5 max-pooling, followed by flatten, dropout (0.3) and a sigmoid output unit. A basic block stacks three 3×3 convolutions with 12 filters (ELU, ELU, linear), batch normalisation, ELU and 2×2 max-pooling; the model has 12,873 parameters and is trained with the Nadam optimiser [10]. The same trained CNN is also the first ensemble member, so the baseline-versus-ensemble comparison isolates the effect of combination (the two rows are numerically identical in our results).


### B. Heterogeneous Ensemble Members


Four learners with different inductive biases form the ensemble (Fig. 1): the CNN, which learns spatial features end-to-end; an SVM (maximum-margin boundary); a Random Forest (bagged decision trees, low variance); and an ANN (dense network). Each member m outputs a hotspot probability pm(x) for a layout clip x. Each classical member receives the 64-dimensional feature embedding extracted from the CNN's penultimate dense layer. The SVM and ANN operate on standardized versions of this embedding, while the Random Forest uses the 64-dimensional embedding directly.


*Fig. 1. Proposed heterogeneous ensemble. All members see the same benchmark partition; three combiners are compared with the CNN alone.*



### C. Combination Strategies


Majority vote flags a clip as HS when at least half of the members do (hard decisions). Weighted vote averages member probabilities with non-negative weights summing to one and thresholds the result:

(3)

with a clip flagged as HS when s(x) ≥ τ. Stacking [15] instead trains a meta-learner g on the member outputs,

(4)

so that it can learn which member to trust. To avoid leakage, g must be fitted on outputs for data not used to train the members. For stacking, the member models are trained on the training split and their probability outputs on the held-out 15% validation split are used to train the logistic-regression meta-learner. Thus, the meta-learner is fitted only on data not used to train the individual members; no out-of-fold predictions are used.


## V. EXPERIMENTAL SETUP


Protocol. Each of the five benchmarks is trained and tested separately on its official training and test partitions. All models, including the baseline, use the same preprocessing and the same partitions; the test partition is used only for the final scores. Every model is scored with the metrics of Section III-B, and averages over benchmarks are unweighted (macro) means. Table II documents the configuration.


#### TABLE II


Experimental configuration

Component

Setting

Data

Official ICCAD-12 train/test partitions; five benchmarks handled separately (no merging).

CNN input

Input: 48×48 grayscale, pixel values scaled to [0,1]; Nadam optimizer, 15 epochs, batch size 32, binary cross-entropy, class weighting enabled

Baseline CNN

Architecture of [13], [10] (12,873 params.), Nadam;

SVM

RBF kernel, class weights = balanced; trained on 64-D CNN embeddings, standardized using StandardScaler

Random Forest

300 trees, default maximum depth, class weights = balanced; trained on 64-D CNN embeddings

ANN

2 hidden layers: 128 and 64 units, ReLU/standard MLP activations, MLPClassifier, 500 max iterations, early stopping; trained on standardized 64-D CNN embeddings

Majority vote

Hard voting over the four base models using threshold 0.5; ties are resolved as positive (HS)

Weighted vote

Soft voting; weights proportional to each model's validation Balanced Accuracy, normalized to sum to 1; decision threshold 0.5

Stacking

Logistic Regression meta-learner with class-balanced weighting, trained on the validation-set predictions of the four base models

Seeds / runs

Random seed = 42; one run per benchmark

Software / HW

Python 3.12.13; TensorFlow/Keras and scikit-learn; Kaggle notebook environment with NVIDIA Tesla T4


## VI. RESULTS AND ANALYSIS



### A. Baseline versus Ensembles


Table III and Fig. 2 give BA for every model and benchmark. Stacking attains the highest average BA, 92.97%, which is +0.88 pp above the baseline CNN (92.09%) and +4.1 pp above the best classical member (SVM, 88.86%). Majority and weighted voting fall 2.2 and 2.1 pp below the baseline. H1 is therefore supported for stacking only, and only by a small margin. Stacking beats the baseline on ICCAD-1 to -3 (+3.28, +1.87, +1.39 pp) but is lower on ICCAD-4 (−1.54 pp) and ICCAD-5 (−0.62 pp), the two most imbalanced benchmarks, where the baseline CNN is strongest.


#### TABLE III


Balanced accuracy (%) per benchmark

Bench.

CNN

SVM

RF

ANN

Maj.

Wtd.

Stack

B1

82.83

86.60

85.54

83.59

83.77

86.18

86.11

B2

90.79

87.89

87.21

86.09

90.78

88.90

92.66

B3

95.71

97.20

97.05

96.95

97.11

97.18

97.10

B4

93.73

86.34

82.57

79.22

88.06

85.58

92.19

B5

97.42

86.26

88.71

50.00

89.86

92.29

96.80

Avg.

92.09

88.86

88.22

79.17

89.92

90.03

92.97

CNN = baseline. RF = Random Forest; Maj./Wtd. = majority/weighted vote. Best value per row in bold.


*Fig. 2. Balanced accuracy of the baseline CNN and the three ensembles (y-axis truncated at 75%).*


Table IV shows the other metrics averaged over benchmarks. The CNN is a high-recall, lower-specificity detector (R = 92.3%, S = 91.9%), whereas SVM and RF are the opposite (R ≈ 83–84%, S ≈ 93.6%). Voting ensembles inherit the classical members’ conservatism: recall drops to 87.4% (majority) and 86.5% (weighted) while precision and F1 improve. Stacking is the only ensemble that keeps the CNN’s recall (92.7%) and also raises specificity to 93.2%, lifting F1 from 46.7% to 48.6%. Low absolute precision (< 43% for all models) follows from prevalence, Eq. (2).


#### TABLE IV


Average metrics over the five benchmarks (%)

Model

BA

Prec.

Recall

Spec.

F1

CNN (baseline)

92.09

37.72

92.28

91.91

46.75

SVM

88.86

39.63

84.05

93.67

49.94

Random Forest

88.22

42.58

82.86

93.57

51.66

ANN

79.17

37.48

65.76

92.57

43.36

Majority vote

89.92

39.12

87.35

92.48

49.93

Weighted vote

90.03

41.22

86.50

93.55

51.36

Stacking

92.97

37.26

92.69

93.25

48.59

ANN precision on ICCAD-5 is 0/0 (no positive predictions) and is counted as 0. Best value per column in bold.


### B. Per-Benchmark Metrics and Confusion Matrices


Tables V and VI and Fig. 3 give the full per-benchmark results and confusion matrices. The differences between baseline and stacking are of different kinds on different benchmarks. On ICCAD-1 both detect all 226 hotspots and stacking removes 307 of 1,607 false alarms (specificity 65.7% → 72.2%), yet precision stays at 12–15% for every model, so ICCAD-1 is the hardest benchmark for false alarms. On ICCAD-2 stacking finds 19 more of the 498 hotspots (recall 81.9% → 85.7%) for 29 more false alarms. On ICCAD-3 it cuts missed hotspots from 123 to 49 (−60%) but adds 608 false alarms. On ICCAD-4 it misses 6 more hotspots but raises 99 fewer false alarms; on ICCAD-5 it misses one more hotspot but raises 233 fewer false alarms. Pooled over all test sets (Table VI), stacking misses 148 hotspots against 234 for the baseline (−36.8%) with essentially the same number of false alarms (3,500 vs 3,502), which supports H2 for stacking. Weighted voting is also better than the baseline in pooled counts (214 misses, 3,113 false alarms) yet worse in macro-average BA, because pooled counts are dominated by the large ICCAD-2/3 test sets whereas the macro average gives ICCAD-4/5 equal weight.


#### TABLE V


Per-benchmark metrics (%) of baseline and ensembles

Bench.

Model

BA

Prec.

Recall

Spec.

F1

B1

CNN

82.83

12.33

100.00

65.66

21.95

Maj.

83.77

12.95

100.00

67.54

22.93

Wtd.

86.18

14.88

100.00

72.37

25.90

Stack

86.11

14.81

100.00

72.22

25.80

B2

CNN

90.79

73.78

81.93

99.65

77.64

Maj.

90.78

73.38

81.93

99.64

77.42

Wtd.

88.90

75.39

78.11

99.69

76.73

Stack

92.66

71.05

85.74

99.58

77.71

B3

CNN

95.71

67.10

93.20

98.22

78.03

Maj.

97.11

53.68

97.51

96.72

69.25

Wtd.

97.18

56.47

97.29

97.07

71.46

Stack

97.10

55.09

97.29

96.91

70.35

B4

CNN

93.73

28.34

88.70

98.76

42.95

Maj.

88.06

37.26

76.84

99.28

50.18

Wtd.

85.58

40.45

71.75

99.41

51.73

Stack

92.19

33.63

85.31

99.07

48.24

B5

CNN

97.42

7.05

97.56

97.27

13.16

Maj.

89.86

18.33

80.49

99.24

29.86

Wtd.

92.29

18.92

85.37

99.22

30.97

Stack

96.80

11.71

95.12

98.48

20.86

Avg.

CNN

92.09

37.72

92.28

91.91

46.75

Maj.

89.92

39.12

87.35

92.48

49.93

Wtd.

90.03

41.22

86.50

93.55

51.36

Stack

92.97

37.26

92.69

93.25

48.59


#### TABLE VI


Confusion-matrix counts (HS = positive class)

Bench.

Model

TP

FP

FN

TN

B1

CNN

226

1,607

0

3,072

Maj.

226

1,519

0

3,160

Wtd.

226

1,293

0

3,386

Stack

226

1,300

0

3,379

B2

CNN

408

145

90

41,153

Maj.

408

148

90

41,150

Wtd.

389

127

109

41,171

Stack

427

174

71

41,124

B3

CNN

1,685

826

123

45,507

Maj.

1,763

1,521

45

44,812

Wtd.

1,759

1,356

49

44,977

Stack

1,759

1,434

49

44,899

B4

CNN

157

397

20

31,493

Maj.

136

229

41

31,661

Wtd.

127

187

50

31,703

Stack

151

298

26

31,592

B5

CNN

40

527

1

18,800

Maj.

33

147

8

19,180

Wtd.

35

150

6

19,177

Stack

39

294

2

19,033

Pooled

CNN

2,516

3,502

234

140,025

Maj.

2,566

3,564

184

139,963

Wtd.

2,536

3,113

214

140,414

Stack

2,602

3,500

148

140,027


*Fig. 3. Confusion matrices (rows: actual, columns: predicted) of the baseline CNN and the stacking ensemble. Colour shows the row-normalised fraction; counts and percentages are annotated.*



### C. Which Errors Does Combination Reduce?



*Fig. 4 places every model at its average operating point. The CNN sits at high recall but a false-positive rate of 8.09%; SVM and RF sit at lower recall but a lower false-positive rate; the ANN has by far the lowest recall (65.8%) because it detects no hotspot at all on ICCAD-5 and only 58.8% on ICCAD-4. Fixed-rule ensembles land between the members. Majority voting, for example, loses hotspots that the CNN alone catches: on ICCAD-4/5 its recall is 76.8%/80.5% against the CNN’s 88.7%/97.6%, because three weaker members outvote the one strong member. Stacking is the only combiner to move up and to the left of the baseline (higher recall and a lower false-positive rate, 6.75%), because a learned meta-learner can down-weight unreliable members. Per benchmark, stacking mainly removes misses on ICCAD-2 and -3 and false alarms on ICCAD-1, -4 and -5. Errors shared by all members are not removed: on ICCAD-1 every model has recall ≥ 98.7% but specificity of only 65.7–73.2%, so the false alarms there are largely common-mode. A plausible—but untested—reason is ICCAD-1’s small training set (439 clips).*



*Fig. 4. Average operating points (mean over five benchmarks). Stacking is the only ensemble that improves on the CNN in both recall and false-positive rate.*



### D. Ablation: Leave-One-Out Members


To quantify each member’s contribution, one member at a time is removed from the weighted-vote ensemble (full ensemble: 90.03% average BA). Table VII and Fig. 5 show the change. Removing the CNN causes the largest drop (−2.74 pp on average, −10.83 pp on ICCAD-5), so the CNN is the most valuable member, and its contribution is concentrated on the imbalanced ICCAD-4/5. Removing the SVM costs 0.51 pp. Removing the RF (+0.41 pp) or the ANN (+0.90 pp) improves the ensemble; the ANN’s gain comes almost entirely from ICCAD-4 and -5, where it is the weakest member. The three classical members alone (CNN removed) reach 87.29%, which is below the best of them (SVM, 88.86%): combining models is not automatically beneficial, and the CNN is what makes the ensemble competitive. The ablation was run on the weighted-vote ensemble only; the stacking ensemble was not ablated.


#### TABLE VII


Leave-one-out ablation, weighted-vote ensemble: BA (%)

Bench.

Full

− CNN

− SVM

− RF

− ANN

B1

86.18

86.27

85.35

85.31

86.28

B2

88.90

87.90

88.60

89.09

88.69

B3

97.18

97.15

97.18

97.24

97.24

B4

85.58

83.65

84.25

85.84

87.80

B5

92.29

81.46

92.20

94.70

94.66

Avg.

90.03

87.29

89.52

90.44

90.93


*Fig. 5. Change in BA (pp) relative to the full weighted-vote ensemble when a member is removed. Negative = the member helps.*



### E. Comparison with the Reference Work and Validity


The reference work reports 95.3% average BA for the CNN and 97.9% on ICCAD-2 [10], against 92.09% and 90.79% here. These numbers are not directly comparable: [10] reports “validation” BA without stating the partition, whereas ours are computed on test partitions whose sizes match the official ones (Table I). Both agree that the CNN is a strong single model. We did not measure inference time; an ensemble costs at least the sum of its members, so the 6.3 ms per clip reported for the CNN [10] is a lower bound.

Statistical uncertainty. Each benchmark was evaluated once. Treating recall and specificity as binomial proportions, the standard error of a single benchmark’s BA is 0.2–1.7 pp for stacking (largest on ICCAD-5, which has 41 hotspots) and the standard error of the five-benchmark average is about 0.4–0.5 pp for both the CNN and stacking. The +0.88 pp average gain, and the ICCAD-4/5 deficits, are therefore comparable to the sampling noise even before training randomness, so they should be read as trends rather than significant differences. Only the pooled reduction in missed hotspots and the ICCAD-1 and ICCAD-3 gains appear to exceed this range clearly.


## VII. DISCUSSION, LIMITATIONS AND FUTURE WORK


Meaning of the results. Heterogeneity helps only when the combiner can exploit it. A fixed vote treats a 12,873-parameter CNN and three weaker members as equals and lets the majority overrule the strongest, whereas stacking preserves the CNN’s sensitivity and borrows the classical models’ lower false-alarm rate. For CAD use, where a missed hotspot is far costlier than a false alarm, the 37% pooled reduction in misses is the most relevant outcome; the price is running four models instead of one.

Limitations. (i) Single run per benchmark, with no confidence intervals or paired significance tests. (ii) The ablation covers weighted voting only. (iii) Ensemble latency, memory and parameter count were not measured. (iv) The ANN collapsed on ICCAD-5, indicating unstable training under extreme imbalance, and no dedicated imbalance-handling was studied for it. (v) Only within-benchmark performance was tested, not cross-benchmark generalisation, and we did not compare against state-of-the-art detectors such as [7]. (vi) The ICCAD-1 test-set count differs from [10] (Table I).

Future work. Promising directions are: repeated runs with paired tests (e.g., bootstrap or McNemar); cost-sensitive stacking or threshold tuning to target a recall level; pruning weak members such as the ANN; probability calibration before combination; a two-stage cascade in which the cheap CNN screens and the ensemble re-examines only uncertain clips; more diverse deep members; and measured latency–accuracy trade-offs.


## VIII. CONCLUSION


We compared a CNN baseline with majority-vote, weighted-vote and stacking ensembles of CNN, SVM, Random Forest and ANN on all five ICCAD-12 benchmarks. Stacking gave the best average balanced accuracy (92.97% vs 92.09%) and cut pooled missed hotspots by 37% at equal false alarms, while simple voting did not beat the CNN. The CNN is the indispensable member, the ANN is a liability, and the benefit is benchmark-dependent and modest relative to single-run uncertainty. A learned combiner, rather than a fixed vote, is thus the appropriate way to combine heterogeneous models for hotspot detection; confirming the gain requires repeated runs and latency measurements.

AUTHOR CONTRIBUTIONS


#### TABLE VIII


Work allocation

Task

Member(s)

Literature survey and problem definition

Akhil

ICCAD-12 data and class-imbalance analysis

Akhil

Baseline CNN implementation and evaluation

Tejasvini

SVM, Random Forest and ANN members

Tejasvini

Voting and stacking ensembles

Subham

Ablation and error analysis, report and documentation

Subham

CODE AVAILABILITY

Notebooks, results (CSV) and the README with execution instructions: [FILL: group repository URL]. The baseline CNN follows the architecture of [13] as released with [10].

ACKNOWLEDGMENT

The authors thank Dr. G. Lakshmi Priya for the problem statement and guidance.

REFERENCES

[1]H. Yang, Y. Lin, B. Yu, and E. F. Y. Young, “Lithography hotspot detection: From shallow to deep learning,” in Proc. 30th IEEE Int. Syst.-on-Chip Conf. (SOCC), Munich, Germany, 2017, pp. 233–238.

[2]W.-Y. Wen, J.-C. Li, S.-Y. Lin, J.-Y. Chen, and S.-C. Chang, “A fuzzy-matching model with grid reduction for lithography hotspot detection,” IEEE Trans. Comput.-Aided Design Integr. Circuits Syst., vol. 33, no. 11, pp. 1671–1680, Nov. 2014.

[3]Y.-T. Yu, G.-H. Lin, I. H.-R. Jiang, and C. Chiang, “Machine-learning-based hotspot detection using topological classification and critical feature extraction,” IEEE Trans. Comput.-Aided Design Integr. Circuits Syst., vol. 34, no. 3, pp. 460–470, Mar. 2015.

[4]T. Matsunawa, J.-R. Gao, B. Yu, and D. Z. Pan, “A new lithography hotspot detection framework based on AdaBoost classifier and simplified feature extraction,” in Proc. SPIE, vol. 9427, 2015, Art. no. 94270S.

[5]H. Zhang, B. Yu, and E. F. Y. Young, “Enabling online learning in lithography hotspot detection with information-theoretic feature optimization,” in Proc. IEEE/ACM Int. Conf. Comput.-Aided Design (ICCAD), 2016, Art. no. 47.

[6]H. Yang, L. Luo, J. Su, C. Lin, and B. Yu, “Imbalance aware lithography hotspot detection: A deep learning approach,” J. Micro/Nanolithogr. MEMS MOEMS, vol. 16, no. 3, Art. no. 033504, 2017.

[7]H. Yang, J. Su, Y. Zou, Y. Ma, B. Yu, and E. F. Y. Young, “Layout hotspot detection with feature tensor generation and deep biased learning,” IEEE Trans. Comput.-Aided Design Integr. Circuits Syst., vol. 38, no. 6, pp. 1175–1187, Jun. 2019.

[8]J. A. Torres, “ICCAD-2012 CAD contest in fuzzy pattern matching for physical verification and benchmark suite,” in Proc. IEEE/ACM Int. Conf. Comput.-Aided Design (ICCAD), 2012, pp. 349–350.

[9]K. H. Brodersen, C. S. Ong, K. E. Stephan, and J. M. Buhmann, “The balanced accuracy and its posterior distribution,” in Proc. 20th Int. Conf. Pattern Recognit. (ICPR), 2010, pp. 3121–3124.

[10]A. Verma, K. A. Rao, and D. S. Hegde, “Lithography hotspot detection using deep learning,” EE769 course project report, IIT Bombay. [Online]. Available: https://github.com/Intelectron6/Lithography-Hotspot-Detection (accessed Sep. 20, 2026).

[11]K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[12]L. Liao, S. Li, Y. Che, W. Shi, and X. Wang, “Lithography hotspot detection method based on transfer learning using pre-trained deep convolutional neural network,” Appl. Sci., vol. 12, no. 4, Art. no. 2192, 2022.

[13]V. Borisov and J. Scheible, “Lithography hotspots detection using deep learning,” in Proc. 15th Int. Conf. Synthesis, Modeling, Anal. Simul. Methods Appl. Circuit Design (SMACD), Prague, Czech Republic, 2018, pp. 145–148.

[14]T. G. Dietterich, “Ensemble methods in machine learning,” in Multiple Classifier Systems (LNCS 1857). Berlin, Germany: Springer, 2000, pp. 1–15.

[15]D. H. Wolpert, “Stacked generalization,” Neural Netw., vol. 5, no. 2, pp. 241–259, 1992.

[16]O. Sagi and L. Rokach, “Ensemble learning: A survey,” WIREs Data Mining Knowl. Discovery, vol. 8, no. 4, Art. no. e1249, 2018.