# Calibration under class imbalance

**Do differentiable, train-time calibration losses survive class imbalance, and do they calibrate the full probability vector or only the predicted class?**

A small, self-contained study. Multi-class classification, two domains (hospital readmission, CIFAR-100-LT), five training losses, four post-hoc corrections, three seeds.

---

## Why this question

Two literatures exist side by side and rarely meet.

The first says that modern neural networks are overconfident, and that the expected calibration error (ECE) can be reduced either after training (temperature scaling) or during it (focal loss, label smoothing, MDCA, soft-binned ECE). Almost all of this work is evaluated on balanced CIFAR, and almost all of it reports the ECE of the **predicted class only**.

The second says that when classes are imbalanced, the posterior probabilities a classifier learns are **systematically biased**, and that the bias is analytically correctable. Dal Pozzolo, Caelen, Johnson and Bontempi showed this for undersampling in 2015. The bias is not a nuisance: it always inflates the estimated risk, and more data does not remove it.

Nobody has properly crossed the two. That is what this repository does.

A single number makes the stake concrete. Take a true risk of 5 %. Undersample the majority class at rate β = 0.1, which is routine practice on imbalanced data. The classifier now learns:

```
p_s = p / (p + β(1 − p)) = 0.05 / (0.05 + 0.1 × 0.95) = 0.345
```

A **5 % risk is learned as a 35 % risk**. Applied to hospital readmission, that is the difference between discharging a patient and enrolling them in a follow-up programme. The Dal Pozzolo correction recovers 0.05 exactly. The question this repository asks is whether the newer train-time calibration losses do too, and whether they still work when there are 100 classes instead of 2.

---

## The three claims under test

**H1. ECE hides the problem under imbalance.** The ECE only looks at the predicted class, and under imbalance the predicted class is almost always a frequent one. A model can therefore have an excellent ECE while its probabilities for rare classes are badly wrong. The Static Calibration Error (SCE, Nixon et al. 2019), which averages the calibration error over all K classes one-vs-rest, should reveal what the ECE conceals.

**H2. Train-time losses degrade under imbalance.** MDCA constrains the *batch-average* predicted probability of each class to match its *batch-average* empirical frequency. Under a long-tailed distribution with a batch size of 128 and 100 classes, most classes appear zero or one time per batch. The constraint becomes extremely noisy exactly where it is most needed. This is a specific, testable prediction.

**H3. Post-hoc and train-time corrections do not compose.** If a loss has already flattened the model's confidence during training, fitting a temperature on top may overcorrect (T < 1, sharpening rather than flattening). Early runs already show this for focal loss, which comes out *under*confident.

---

## Data

| Dataset | Task | Classes | Imbalance | Source |
|---|---|---|---|---|
| **Diabetes 130-US Hospitals** | 30-day readmission | 3 | ~54 / 35 / 11 % (natural) | [UCI ML Repository, id 296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — CC BY 4.0 |
| **CIFAR-100-LT** | image classification | 100 | exponential, factor 100 (induced) | [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html), long-tailed protocol of Cui et al. (2019) |

The hospital data is the point of the exercise. 101 766 inpatient encounters, ten years, 130 hospitals. The target has three levels: no readmission, readmission after 30 days, readmission within 30 days. The last one, the clinically urgent one, is the rare one (about 11 %). This is a setting where the predicted probability *is* the decision: a hospital with a limited follow-up budget needs to know whether the 12 % it is shown means 12 %.

CIFAR-100-LT is there for comparability. It is the canonical benchmark of the calibration literature, so the numbers can be read against Guo et al. (2017), Mukhoti et al. (2020) and Hebbalaguppe et al. (2022). With 100 classes, the gap between top-label and full-vector calibration becomes measurable rather than theoretical.

The test set is kept **balanced** on CIFAR-100-LT, deliberately. Calibration should be measured on the population you will deploy on, not on the skewed population you happened to train on. That distinction is the whole subject of the 2015 paper.

---

## Method

**Losses** (`src/losses.py`)

| Loss | Mechanism | Differentiable? |
|---|---|---|
| Cross-entropy | baseline | yes, but nothing in its gradient rewards calibration |
| Focal loss (γ=3) | down-weights easy examples, implicitly regularises entropy | yes, calibrates as a side effect |
| Label smoothing (α=0.05) | caps the target logit | yes, but flattens the non-predicted classes |
| **CE + MDCA** | batch-level match between mean confidence and mean frequency, per class | yes, no binning at all |
| **CE + soft-binned ECE** | replaces hard bin membership with a Gaussian kernel | yes, at the cost of a temperature hyperparameter |

The last two are the ones the PhD call is really about. MDCA avoids binning entirely but only enforces an average constraint, far weaker than true calibration. Soft binning stays close to the ECE but its gradient vanishes as the softening temperature goes to zero — which is precisely the trade-off worth studying.

**Post-hoc corrections** (`src/posthoc.py`)

- Temperature scaling — one parameter, preserves accuracy exactly, cannot fix per-class miscalibration.
- Vector scaling — one scale and one bias per class, can fix per-class miscalibration, may change accuracy.
- **Prior-shift correction** — the multi-class generalisation (Saerens et al. 2002) of the Dal Pozzolo–Caelen undersampling correction. Reweights by the ratio of true to training priors.

**Metrics** (`src/metrics.py`) — ECE (15 equal-width bins), adaptive ECE (equal-mass bins), MCE, **SCE / classwise ECE**, Brier score, NLL, and the signed confidence-minus-accuracy gap.

**Protocol.** Strict three-way split. The validation set fits the post-hoc methods and nothing else; the test set is touched once. Fitting a temperature on the test set and then reporting test ECE is a common error in public repositories, and it produces flattering numbers that mean nothing.

---

## Running it

```bash
pip install -r requirements.txt

python run_tabular.py --dataset diabetes --seeds 3 --epochs 30
python run_cifar.py   --imbalance 100 --seeds 3 --epochs 200   # needs a GPU
```

`run_tabular.py` runs on CPU in a few minutes. `run_cifar.py` needs a GPU; it fits comfortably in a free Colab session.

---

## Results

*(to be filled in from `results/summary_*.csv` once the runs are complete — the table below is the shape it takes)*

| Loss | Correction | Accuracy | Gap | ECE | SCE | Brier |
|---|---|---|---|---|---|---|
| CE | none | | | | | |
| CE | temperature | | | | | |
| Focal | none | | | | | |
| CE + MDCA | none | | | | | |
| CE + soft-ECE | none | | | | | |

Two figures carry the argument:

- `figures/reliability.png` — reliability diagrams, with the count histogram underneath. Without the histogram you will happily over-interpret a bin containing twelve observations out of fifty thousand.
- `figures/sce_par_classe.png` — per-class calibration error against training-set class frequency. **If this scatter slopes upward as frequency falls, H1 is confirmed and the ECE is lying by omission.**

---

## Known limitations

Stated up front, because a study that hides them is not worth reading.

- Two datasets is not a benchmark. The findings are indicative, not conclusive.
- Three seeds gives a rough standard deviation, not a confidence interval.
- The soft-binning temperature and the MDCA weight β were set to literature defaults, not tuned. A proper study would sweep β and report the accuracy–calibration frontier, which is arguably the more interesting object.
- The prior-shift correction assumes the covariate distribution is unchanged and only the label prior shifts. On CIFAR-100-LT that holds by construction. On the hospital data it is an assumption, not a fact.
- ResNet-32 is small. Calibration behaviour is known to depend on capacity (Guo et al.), so these results should not be extrapolated to large models.

---

## References

- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.
- Dal Pozzolo, **Caelen**, Johnson, Bontempi (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI.
- Saerens, Latinne, Decaestecker (2002). *Adjusting the Outputs of a Classifier to New a Priori Probabilities.* Neural Computation.
- Nixon, Dusenberry, Zhang, Jerfel, Tran (2019). *Measuring Calibration in Deep Learning.* CVPR Workshops.
- Mukhoti, Kulharia, Sanyal, Golodetz, Torr, Dokania (2020). *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS.
- Hebbalaguppe, Prakash, Madan, Arora (2022). *A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration.* CVPR.
- Karandikar et al. (2021). *Soft Calibration Objectives for Neural Networks.* NeurIPS.
- Cui, Jia, Lin, Song, Belongie (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
- Strack, DeShazo, Gennings, Olmo, Ventura, Cios, Clore (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.

---

Mahmoud Sidibe — DTS Laboratory, Cheikh Anta Diop University, Dakar.
MIT licence.
