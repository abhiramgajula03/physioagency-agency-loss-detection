# PhysioAgency — Physiological Detection of Agency Loss under Algorithmic Task Allocation

A multi-model machine learning pipeline that detects and quantifies **embodied agency loss** — the psychological cost of working under constant algorithmic oversight — using high-frequency physiological data from Empatica E4 wearable sensors.

Built as part of an Innovative Management research assignment applying physiological computing to a supply-chain-relevant problem: as algorithmic task allocation systems increasingly manage human workers, they erode autonomy in ways that traditional self-report surveys can't capture in real time. This project asks whether wearable biosignals can detect that erosion directly — and answers yes, with cross-participant generalization confirmed under rigorous evaluation.

📄 Full report: [`REPORT.pdf`](./REPORT.pdf)

---

## Key Results

Evaluated under **Leave-One-Participant-Out Cross-Validation (LOPO-CV)** — the appropriate standard for small-N physiological studies, since it tests generalization to a genuinely unseen individual rather than allowing within-participant leakage across train/test splits.

| Model | Accuracy | F1 | ROC-AUC | 95% CI (Bootstrap) |
|---|---|---|---|---|
| **ANN** | **0.86** | **0.84** | **0.89** | [0.79, 0.92] |
| Random Forest | 0.81 | 0.79 | 0.85 | [0.74, 0.87] |
| LSTM | 0.72 | 0.70 | 0.75 | [0.64, 0.80] |
| TCN | 0.68 | 0.65 | 0.71 | [0.60, 0.76] |

ANN significantly outperforms both TCN (Wilcoxon p = 0.012) and LSTM (p = 0.031); Bayesian posterior estimate confirms P(θ > 0.75) = 0.97 for ANN accuracy, ruling out small-sample artifacts as the source of the result.

**Signal attribution (RQ2):** EDA mean (r = +0.61) and RMSSD (r = −0.58) are confirmed as the strongest physiological markers of agency loss, both significant at p < 0.001.

**Survival analysis (RQ3):** Kaplan-Meier and Cox Proportional Hazards models confirm that physiological stress markers (EDA HR = 1.84, p < 0.05) precede visible behavioral disengagement across exam sessions — i.e., the physiological signal is an early warning indicator, not just a concurrent one.

---

## What This Project Does

1. **Signal processing** — Decomposes raw EDA into tonic (SCL) and phasic (SCR) components via NeuroKit2; extracts HRV features (RMSSD, SDNN, pNN50) from IBI sequences over 60-second sliding windows (50% overlap).
2. **Leakage-free labeling** — Constructs an Agency Loss Index (ALI) from physiological components *not* used as model input features, eliminating the feature-label leakage that inflated an earlier iteration's accuracy to a suspicious 1.00.
3. **Multi-model classification** — Trains and compares Random Forest, ANN, LSTM, and TCN architectures for binary agency-loss classification.
4. **Causal identification** — Applies an Instrumental Variable (2SLS) strategy using exam session order as an instrument, to move beyond correlational claims.
5. **Anomaly detection & regime clustering** — A deep autoencoder + K-Means pipeline identifies three physiological stress regimes: *Autonomous*, *Pressured*, and *Disengaged/Overloaded*.
6. **Survival analysis** — Kaplan-Meier and Cox PH models forecast disengagement/burnout risk from early physiological signals.
7. **Robustness checks** — Bootstrap confidence intervals (1,000 iterations) and Bayesian Beta-Binomial posterior estimation validate that results hold up despite the small participant pool (N = 10).

---

## Tech Stack

- **Signal processing:** NeuroKit2, SciPy, NumPy, Pandas
- **Deep learning:** TensorFlow / Keras (LSTM, Conv1D/TCN, Dense, BatchNorm, Dropout)
- **Classical ML:** scikit-learn (Random Forest, LeaveOneGroupOut, KMeans, PCA)
- **Causal inference:** `linearmodels` (IV / 2SLS)
- **Survival analysis:** Lifelines (KaplanMeierFitter, CoxPHFitter)
- **Bayesian inference:** PyMC3
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Environment:** Google Colab

---

## Dataset

[Wearable Exam Stress Dataset](https://doi.org/10.13026/kvkb-aj90) (Amin et al., 2022), publicly available via PhysioNet. 10 participants wore Empatica E4 wristbands across three escalating-pressure exam sessions (Midterm 1, Midterm 2, Final), recording EDA, BVP, HR, IBI, TEMP, and ACC.

> The dataset is not included in this repo — download it directly from PhysioNet using the link above and place it in the expected data directory before running the notebook.

**Important scope note:** This dataset is an academically controlled proxy for algorithmic task pressure, not a direct replication of supply chain operations. Results should be read as proof-of-concept evidence for the *physiological detectability* of agency loss, pending real-world replication in warehouse/logistics settings — see the report's Limitations and Boundary Conditions sections for the full discussion.

---

## Repository Structure

```
.
├── project_code.ipynb   # Full pipeline: preprocessing → modeling → evaluation → causal/survival analysis
├── REPORT.pdf            # Full write-up: theory, methodology, results, limitations
└── README.md
```

---

## Theoretical Framework

The pipeline is grounded in a three-stage causal chain, integrating Self-Determination Theory (Deci & Ryan, 2000) and Conservation of Resources Theory (Hobfoll, 1989):

**Algorithmic Task Allocation** → constrains autonomy → **Agency Loss** → activates sympathetic nervous system → **Physiological Stress** (↑EDA, ↓HRV) → accumulates into → **Performance Degradation & Attrition**

This project targets Stage 1 → Stage 2 detection and Stage 2 → Stage 3 prediction, using wearable signals as the measurable interface between algorithmic pressure and its human cost.

---

## Limitations

- **N = 10 participants** — limits statistical power, particularly for survival analysis; addressed via bootstrap/Bayesian robustness checks, but larger cross-dataset validation (WESAD, SWELL-KW, CogWear) is a clear next step.
- **Proxy dataset** — academic exam stress, not direct supply-chain operational data.
- **Population/task-type boundary conditions** — findings may not transfer directly to physically active warehouse contexts without recalibration (see report Section 7 for full discussion).

---

## Author

**Gajula Abhiram** — B.Tech Information Technology, IIIT Allahabad
[LinkedIn](https://www.linkedin.com/in/gajula-abhiram-0a9a462b1) · [GitHub](https://github.com/abhiramgajula03)
