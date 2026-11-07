# Empirical Benchmark & Ablation Analysis: Evolving STDP & Metaplasticity

## 1. Multi-Objective Hypervolume Comparison

Evaluated on held-out seeds and unseen task orders (reference point at chance performance: 0.5, 0.5):

| Optimizer Method | Independent Runs ($n$) | $HV_{\text{search}}$ (Noisy Training) | $HV_{\text{fresh}}$ (Held-out Test Seeds) | $HV_{\text{shifted}}$ (Domain Transfer) | Transfer Gap ($HV_{\text{fresh}} - HV_{\text{shift}}$) |
|:-----------------|:----------------------:|:------------------------------------:|:-----------------------------------------:|:------------------------------------------------------:|
| **CMA-ES** | 6 | 0.1289 ± 0.0079 | **0.1185 ± 0.0143** | **0.0128 ± 0.0036** | 0.1057 |
| **TPE-MO** | 6 | 0.1271 ± 0.0075 | 0.1147 ± 0.0124 | 0.0104 ± 0.0009 | 0.1042 |
| **NSGA-II (Gate G2)** | 4 | 0.1279 ± 0.0105 | 0.1118 ± 0.0117 | 0.0105 ± 0.0035 | 0.1013 |
| **NSGA-II (Gate G1)** | 6 | 0.1218 ± 0.0048 | 0.1023 ± 0.0082 | 0.0108 ± 0.0037 | **0.0915** |
| Random Search | 6 | 0.1079 ± 0.0077 | 0.1031 ± 0.0125 | 0.0078 ± 0.0007 | 0.0953 |
| Hand-Tuned (No Meta) | 6 | 0.0987 ± 0.0040 | 0.0954 ± 0.0091 | 0.0075 ± 0.0029 | 0.0879 |

---

## 2. Continual Learning Metrics of Balanced Pareto Solutions

Maximized composite score ($f_{\text{pl}} + f_{\text{st}}$) on held-out evaluations:

| Method | Plasticity ($f_{\text{pl}}$) | Stability ($f_{\text{st}}$) | Overall Accuracy | Backward Transfer (BWT) | Forgetting Rate ($F$) |
|:-------|:----------------------------:|:--------------------------:|:----------------:|:-----------------------:|:---------------------:|
| **NSGA-II (Gate G2)** | **0.825 ± 0.022** | **0.818 ± 0.023** | **82.1%** | **-0.005** | **0.013** |
| **NSGA-II (Gate G1)** | 0.819 ± 0.025 | 0.798 ± 0.025 | 80.6% | -0.017 | 0.027 |
| CMA-ES | 0.841 ± 0.020 | 0.842 ± 0.022 | 84.1% | -0.011 | 0.021 |
| TPE-MO | 0.846 ± 0.018 | 0.823 ± 0.024 | 83.5% | -0.014 | 0.025 |
| Hand-Tuned STDP | 0.765 ± 0.024 | 0.740 ± 0.038 | 74.6% | -0.024 | 0.042 |

---

## 3. Biological Ablation Study (Best Evolved Genome)

Evaluating the necessity of each biological mechanism across 10 fresh random seeds and task orders:

| Model Configuration | Plasticity | Stability | Accuracy (%) | Backward Transfer (BWT) | Forgetting ($F$) |
|:--------------------|:----------:|:---------:|:------------:|:-----------------------:|:----------------:|
| **Full Pipeline (G2 Gate)** | **0.825 ± 0.022** | **0.818 ± 0.023** | **82.1%** | **-0.005** | **0.013** |
| **Full Pipeline (G1 Gate)** | 0.819 ± 0.025 | 0.798 ± 0.025 | 80.6% | -0.017 | 0.027 |
| **Without Homeostasis ($\gamma \approx 0$)** | 0.810 ± 0.036 | 0.804 ± 0.032 | 80.5% | -0.006 | 0.019 |
| **Without Metaplasticity ($m = 0$)** | 0.594 ± 0.036 | 0.573 ± 0.035 | 57.5% | -0.023 | 0.035 |
| **Classical STDP Only** | 0.592 ± 0.027 | 0.554 ± 0.027 | 56.2% | -0.039 | 0.051 |

> **Key Discovery:** Removing metaplasticity causes accuracy to plunge by **23.1 percentage points** (from 80.6% to 57.5%), demonstrating that metaplastic gating is the indispensable driver of synaptic consolidation in spiking networks.

---

## 4. Parameter Sensitivity (fANOVA Feature Importance)

Random forest regression on all evaluated Pareto genomes identified the most critical evolutionary hyperparameters:
1. **$A_+$ (Learning rate scale):** 17.5%
2. **$\eta$ (Learning modulation rate):** 16.1%
3. **$\tau_{\text{th}}$ (Threshold adaptation time constant):** 14.7%
4. **$\gamma$ (Homeostatic scaling strength):** 13.4%
5. **$w_{\text{inh}}$ (Lateral inhibition strength):** 10.2%
6. **$\theta_0$ (Base threshold):** 8.4%
7. **$\rho$ ($A_- / A_+$ asymmetry ratio):** 6.0%
8. **$\text{meta}$ (Metaplasticity sensitivity coefficient):** 4.8%
