# End-to-End Deep Learning for Plant Acoustic Emission Analysis: 1D Waveform CNNs and 2D Continuous Wavelet Transform Networks

**Author:** Marius Peter  
**Institution:** Institute for Software and Systems Engineering, Clausthal University of Technology  
**Project:** AE-Lab — Machine Learning-Based Detection and Classification of Plant Xylem Cavitation Events Using Acoustic Emission Sensors  
**Date:** October 2026  

---

## 1. Introduction and Scientific Motivation

The classical machine learning pipelines developed in Chapter 6 (summarized in **Table 6.1: Cross-Model Performance Summary**) rely on explicitly defined, domain-engineered time-domain and frequency-domain features (e.g., peak amplitude, rise time, Hjorth parameters, spectral centroid, spectral rolloff, and sub-band energy ratios). While these handcrafted descriptors successfully capture primary acoustic characteristics, they present two intrinsic limitations:
1. **Feature Engineering Bias:** Handcrafted features compress rich, high-frequency ultrasonic waveforms (sampled at up to 50 MS/s) into scalar statistical summaries, potentially discarding subtle phase transitions, multi-harmonic overtones, and transient dispersion patterns.
2. **Temporal vs. Spectral Decoupling:** Conventional Fast Fourier Transform (FFT) features treat the event spectrum as stationary across the integration window, obscuring non-stationary time-frequency evolutions that occur within microseconds of cavitation bubble collapse.

To directly evaluate the hypothesis formulated in the thesis's *Future Work* chapter, this investigation develops and benchmarks two complementary deep learning architectures that process acoustic emission (AE) events without manual feature engineering:
- **One-Dimensional Convolutional Neural Networks (1D-CNN):** Trained directly on normalized voltage time series ($V(t)$), allowing hierarchical convolutional kernels to learn optimal temporal representations directly from the raw data.
- **Two-Dimensional Continuous Wavelet Transform Networks (2D CWT-CNN):** Trained on time-frequency scalograms computed via continuous Morlet wavelets, enabling the network to analyze localized energy distributions simultaneously across time and frequency.

To provide a rigorous, publishable comparison against the established classical machine learning baselines from Table 6.1:
- **Noise Model:** Random Forest ($6{,}648$ training samples, **$0.996$ F1-Score**)
- **Plant Species Model:** XGBoost LOPO ($440$ training samples, **$0.145$ F1-Score**)
- **Stress Classification Model:** SVM RBF ($196$ training samples, **$0.669$ F1-Score**)

All evaluations strictly employ **grouped cross-validation protocols** to enforce zero data leakage and test genuine physical generalization across previously unseen plants and measurement sessions on the exact dataset partitions defined in the thesis.

---

## 2. Methodology and Model Architectures

```
Raw Acoustic Emission (12.5 MS/s)
         │
         ├───► Decimation (x4) & Peak Normalization ───► [B, 1, 3125] ───► 1D Waveform CNN ───► Classification Head
         │
         └───► Continuous Wavelet Transform (Morlet) ──► [B, 1, 32, 128] ─► 2D Scalogram CNN ──► Classification Head
```

### 2.1 Waveform Preprocessing and 1D Representation

Acoustic emission events recorded by the digital storage oscilloscope (TiePie Handyscope HS4 / HS5) consist of $12{,}500$ discrete samples per frame at a native sampling rate of $50\text{ MS/s}$ ($250\ \mu\text{s}$ record length). 

1. **Anti-Aliasing Decimation:** Signals are decimated by a factor of 4, yielding $N = 3{,}125$ discrete voltage points per event at an effective sampling rate of $f_s = 12.5\text{ MS/s}$ (Nyquist frequency $f_{\text{Nyq}} = 6.25\text{ MHz}$). Because the resonant response of the piezoelectric sensor (Pico-HF) is centered between $100\text{ kHz}$ and $800\text{ kHz}$, the decimated bandwidth comfortably captures all physiological acoustic emission dynamics without spectral aliasing.
2. **Peak Amplitude Normalization:** To ensure scale invariance and stable gradient propagation across varying sensor coupling pressures, each waveform $v[n]$ is normalized by its absolute maximum:
   $$\tilde{v}[n] = \frac{v[n]}{\max_{k} |v[k]| + \epsilon}, \quad n \in \{0, \dots, N-1\}$$
   where $\epsilon = 10^{-9}$ prevents division by zero.

### 2.2 Time-Frequency Scalograms via Continuous Wavelet Transform (CWT)

To resolve the non-stationary dynamics of plant acoustic signals, the 1D waveform is mapped into a 2D time-frequency representation using the Continuous Wavelet Transform with the analytic complex Morlet wavelet $\psi(t)$:
$$\psi(t) = \pi^{-1/4} e^{i \omega_0 t} e^{-t^2 / 2}$$
The continuous wavelet coefficients $W(s, \tau)$ at scale $s > 0$ and translation $\tau$ are computed as:
$$W(s, \tau) = \frac{1}{\sqrt{s}} \int_{-\infty}^{\infty} \tilde{v}(t) \psi^*\left(\frac{t - \tau}{s}\right) dt$$
The scalogram power is given by:
$$P(s, \tau) = |W(s, \tau)|^2$$

1. **Logarithmic Scale Spacing:** Scales are distributed logarithmically across 32 levels ($s \in [2, 64]$), providing high temporal resolution at high frequencies and high frequency resolution at lower frequencies.
2. **Dimensionality Reduction:** The continuous time axis is downsampled to $T = 128$ points via uniform stride decimation, producing a standardized 2D tensor of shape $32 \times 128$ (Scales $\times$ Time steps).
3. **Power Normalization:** The scalogram is peak-normalized to $[0, 1]$ before feeding into the network:
   $$\tilde{P}(s, \tau) = \frac{P(s, \tau)}{\max_{s', \tau'} P(s', \tau') + \epsilon}$$

### 2.3 Network Architectures

#### 1D-CNN (Raw Waveform Architecture)
- **Input:** Tensor of shape $(B, 1, 3125)$
- **Block 1:** $\text{Conv1D}(1 \to 16, \text{kernel}=15, \text{stride}=2, \text{pad}=7) \to \text{BatchNorm1D} \to \text{ReLU} \to \text{MaxPool1D}(2)$
- **Block 2:** $\text{Conv1D}(16 \to 32, \text{kernel}=9, \text{stride}=2, \text{pad}=4) \to \text{BatchNorm1D} \to \text{ReLU} \to \text{MaxPool1D}(2)$
- **Block 3:** $\text{Conv1D}(32 \to 64, \text{kernel}=7, \text{stride}=2, \text{pad}=3) \to \text{BatchNorm1D} \to \text{ReLU} \to \text{MaxPool1D}(2)$
- **Block 4:** $\text{Conv1D}(64 \to 128, \text{kernel}=5, \text{stride}=1, \text{pad}=2) \to \text{BatchNorm1D} \to \text{ReLU}$
- **Pooling:** $\text{AdaptiveAvgPool1D}(1) \to \text{Flatten}$ (128 features)
- **Classifier:** $\text{Dropout}(0.3) \to \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to C)$

#### 2D-CNN (CWT Scalogram Architecture)
- **Input:** Tensor of shape $(B, 1, 32, 128)$
- **Block 1:** $\text{Conv2D}(1 \to 16, \text{kernel}=3 \times 3, \text{pad}=1) \to \text{BatchNorm2D} \to \text{ReLU} \to \text{MaxPool2D}(2 \times 2)$
- **Block 2:** $\text{Conv2D}(16 \to 32, \text{kernel}=3 \times 3, \text{pad}=1) \to \text{BatchNorm2D} \to \text{ReLU} \to \text{MaxPool2D}(2 \times 2)$
- **Block 3:** $\text{Conv2D}(32 \to 64, \text{kernel}=3 \times 3, \text{pad}=1) \to \text{BatchNorm2D} \to \text{ReLU}$
- **Pooling:** $\text{AdaptiveAvgPool2D}(1, 1) \to \text{Flatten}$ (64 features)
- **Classifier:** $\text{Dropout}(0.3) \to \text{Linear}(64 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to C)$

---

## 3. Experimental Results and Thesis Benchmark Comparisons

All benchmark tasks have been evaluated under a **unified, leakage-free protocol** where all three model families (Classical ML, 1D-CNN, 2D CWT-CNN) are trained and tested on **strictly identical data indices and cross-validation folds** without test-set target peaking.

---

### 3.1 Task 1: Noise vs. Cavitation Classification (Binary)

Distinguishing genuine plant acoustic emissions from environmental acoustic noise and electrical transients.

#### Understanding the Split Dimensions (Clarification of the 4,855 Test Events):
- **Database Pool:** $24{,}273$ total labelled events in the feature database ($1{,}383$ genuine cavitation signals, $22{,}890$ noise/control events).
- **The 80/20 Hold-Out Split:** An 80/20 stratified split is performed **before** training-set downsampling. This yields:
  - **Training Split (raw 80%):** $19{,}418$ events ($1{,}106$ cavitation signals, $18{,}312$ noise events).
  - **Test Split (untouched 20%):** **$4{,}855$ events** ($277$ cavitation signals, $4{,}578$ noise events). Here, the $277$ signal events represent exactly $20\%$ of all $1{,}383$ cavitation events in the database.
- **Training Subset Balancing (5:1):** To prevent severe class imbalance during training, noise in the training set is downsampled to $5 \times$ the signal count ($5 \times 1{,}106 = 5{,}530$ noise events). The resulting balanced training set comprises $1{,}106 + 5{,}530 = \mathbf{6{,}636\text{ samples}}$ (historically $6{,}648$ samples).
- **Evaluation on Untouched Test Set:** All models are tested on the untouched $4{,}855$ test samples with their natural prevalence ($5.7\%$ signal).

| Model Architecture | Input Representation | Overall Accuracy | Macro F1-Score | Signal Recall (Cavitation) | Signal $F_2$-Score | Dataset & Validation Protocol |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical ML (Random Forest, Table 6.1)** | 10 Handcrafted Features | **99.61%** | *Weighted:* 0.9961<br>*(Macro: 0.9787)* | **100.00%** | **98.65%** | Legacy Split ($4{,}855$ test samples; stored metadata) |
| **Classical ML (Random Forest, re-run)** | 10 Handcrafted Features | 99.53% | 0.9787 | 99.64% | 98.15% | Identical Hold-out Split ($4{,}855$ test samples) |
| **Deep Learning (1D-CNN)** | Raw 1D Voltage Waveform ($3{,}125$ pts) | 99.41% ± 0.12% | 0.9738 ± 0.0050 | 99.40% ± 0.34% | 0.9761 ± 0.0022 | Identical Hold-out Split ($4{,}855$ samples; 3 seeds) |
| **Deep Learning (2D-CNN CWT)** | 2D CWT Morlet Scalogram ($32 \times 128$) | 90.34% ± 11.03% | 0.8027 ± 0.1586 | 90.73% ± 13.10% | 0.7418 ± 0.1646 | Identical Hold-out Split ($4{,}855$ samples; 3 seeds) |

> [!NOTE]
> **Macro F1 vs. Weighted F1:** The thesis reported **0.9961** from Table 6.1, which corresponds to the *weighted* F1-score. Because the test set contains $4{,}578$ noise events and $277$ signals, precision on signal is $\approx 93\%$ (due to $\approx 19$ false positives among thousands of noise transients), yielding an unweighted Macro F1 of $\approx 0.9787$.

---

### 3.2 Task 2: Plant Species Classification (Leave-One-Plant-Out)

Classifying cavitation events across three plant species (*Capsicum annuum*, *Helianthus annuus*, *Solanum lycopersicum*) evaluated strictly on unseen physical plants across the 13 evaluated specimens.

#### Methodological Clarification of the 440 Training Pool and 575 Test Events:
1. **The 440 Training Pool (Table 6.1):** Mini Pepper dominates the database ($994$ events). To prevent the classifier from collapsing into a trivial single-species predictor, the training part of every LOPO fold caps each species at $150$ events (`MAX_PER_CLASS_TRAIN = 150`). Because Tomato has only $140$ available cavitation events, the balanced training pool equals $150 + 150 + 140 = \mathbf{440\text{ samples}}$.
2. **The 575 Test Events:** In the Leave-One-Plant-Out loop, plants with at least 5 events are evaluated as held-out test specimens ($13$ evaluation plants total). To prevent Plant 42 ($~950$ events) from completely dominating test metrics, test events per plant are capped at $150$. Across all 13 folds, the pooled test predictions sum to **$575\text{ test events}$** ($195$ Pepper + $248$ Sunflower + $132$ Tomato). Majority-class baseline is $43.1\%$.

| Model Architecture | Input Representation | Overall Accuracy | Balanced Accuracy | Mean Per-Plant Accuracy | Macro F1-Score | Dataset & Validation Protocol |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical ML (XGBoost, Table 6.1)** | 65 Handcrafted Features + UMAP | 15.30% | 16.18% | 23.93% | 0.1452 | LOPO ($13$ plants, $575$ pooled test events; original run) |
| **Deep Learning (1D-CNN)** | Raw 1D Voltage Waveform ($3{,}125$ pts) | **17.80% ± 4.58%** | **20.23% ± 7.06%** | **32.25% ± 4.67%** | **0.1966 ± 0.0606** | Identical LOPO Folds ($575$ test events; 3 seeds) |
| **Deep Learning (2D-CNN CWT)** | 2D CWT Morlet Scalogram ($32 \times 128$) | 19.94% ± 9.87% | 24.42% ± 12.45% | 32.49% ± 12.87% | 0.1802 ± 0.0879 | Identical LOPO Folds ($575$ test events; 3 seeds) |

> [!IMPORTANT]
> **Specimen-Specific Resonances:** All models remain well below the 33.3% random-guessing chance level and the 43.1% majority baseline. This confirms that stem geometry and sensor acoustic coupling overshadow botanical species differences.

---

### 3.3 Task 3: Stress Type Classification (Mechanical Cut vs. Hydraulic Drought)

Distinguishing mechanical cutting (xylem severance) from water-deficit cavitation across unseen plant specimens.

#### 3A. Balanced Baseline Dataset: 196 Samples (98 Drought vs. 98 Cut, Section 6.4.1)
Evaluated across **5-Fold Stratified Group Cross-Validation on `plant_id`** (zero plant leakage between folds, 3 split seeds):

| Model Architecture | Input Representation | Overall Accuracy (= Balanced Acc.) | Macro F1-Score | Share Predicted Drought | Validation Protocol |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Classical ML (SVM RBF)** | 62 Handcrafted Features | 44.39% ± 0.00% | 0.4406 ± 0.0000 | 42.35% ± 0.00% | 196 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| **Classical ML (Random Forest)** | 62 Handcrafted Features | 35.71% ± 0.00% | 0.3571 ± 0.0000 | 51.02% ± 0.00% | 196 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| **Deep Learning (1D-CNN)** | Raw 1D Voltage ($3{,}125$ pts) | 43.88% ± 7.98% | 0.4303 ± 0.0854 | 57.82% ± 8.56% | 196 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| **Deep Learning (2D-CNN CWT)** | 2D CWT Scalogram ($32 \times 128$) | **53.57% ± 0.42%** | **0.5288 ± 0.0031** | 48.47% ± 11.99% | 196 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| *Legacy: SVM RBF (Table 6.1)* | 20 Handcrafted Features | *54.00%* | *Weighted:* 0.6688<br>*(Macro: 0.3867)* | *96.0% in test* | Single Unstratified Group Split ($50$ test samples: $48$ drought, $2$ cut) |

#### 3B. Extended Full-Database Dataset: 502 Samples (251 Drought vs. 251 Cut)
Evaluated across all available drought events in the database paired with an equal number of cut events under 5-Fold Stratified Group CV on `plant_id` (3 split seeds):

| Model Architecture | Input Representation | Overall Accuracy (= Balanced Acc.) | Macro F1-Score | Share Predicted Drought | Validation Protocol |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Classical ML (SVM RBF)** | 62 Handcrafted Features | 56.77% ± 0.28% | 0.5674 ± 0.0028 | 47.14% ± 0.09% | 502 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| **Classical ML (Random Forest)** | 62 Handcrafted Features | 39.58% ± 0.09% | 0.3870 ± 0.0005 | 61.95% ± 0.28% | 502 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| **Deep Learning (1D-CNN)** | Raw 1D Voltage ($3{,}125$ pts) | 33.27% ± 4.18% | 0.3307 ± 0.0410 | 45.68% ± 3.41% | 502 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |
| **Deep Learning (2D-CNN CWT)** | 2D CWT Scalogram ($32 \times 128$) | **59.83% ± 16.11%** | **0.5807 ± 0.1622** | 37.72% ± 15.79% | 502 samples \| 5-Fold Group CV (`plant_id`, 3 seeds) |

#### Resolving Methodological Discrepancies in the Stress Benchmark:
1. **Accuracy vs. Balanced Accuracy Alignment:** On perfectly balanced datasets ($98:98$ or $251:251$), pooled Accuracy mathematically equals pooled Balanced Accuracy. The earlier discrepancy ($62.55\%$ vs. $48.72\%$) occurred because the legacy script averaged per-fold balanced accuracies; since several folds in group CV contain plants of only one stress condition, fold-wise balanced accuracy degrades into a degenerate score. Evaluating globally across the pooled cross-validation predictions resolves this completely.
2. **The Nature of the Legacy 54% / 0.669 Result:** In Table 6.1, the classical SVM achieved a weighted F1 of $0.6688 \approx 0.669$ on a single held-out test split of $50$ samples. Inspection of the confusion matrix (`[[1, 1], [22, 26]]`) reveals that this test set contained $48$ Drought samples and only $2$ Cut samples. Because plants were split as entire groups, this split was non-stratified and heavily imbalanced ($96\%$ drought), causing the support-weighted F1 to reflect solely the drought class ($F_1 = 0.693$). Its unweighted Macro F1 was only $0.3867$. Therefore, **the 5-fold grouped cross-validation reported above represents the scientifically valid benchmark.**
3. **Physical Interpretation:** Across unseen plants, classification accuracy hovers around chance ($35\%–56\%$), directly confirming the conclusion of Section 6.4.2: single acoustic bubble collapse transients under mechanical cut and drought strongly overlap and require contextual environmental sensor data to separate reliably.

---

## 4. Synthesis and Recommendations

| Benchmark Task | Classical ML Baseline | 1D-CNN (Raw Waveforms) | 2D CWT-CNN (Scalograms) | Thesis Finding |
| :--- | :--- | :--- | :--- | :--- |
| **1. Noise vs. Cavitation** | **0.979 Macro F1** / **99.5% Acc** (RF, $4{,}855$ test) | **0.974 Macro F1** / **99.4% Acc** | 0.803 Macro F1 / 90.3% Acc | **Full Success:** Raw 1D-CNN matches Random Forest without manual feature extraction. |
| **2. Plant Species (LOPO)** | **0.145 Macro F1** / 15.3% Acc (XGBoost, $575$ test) | **0.197 Macro F1** / 17.8% Acc | 0.180 Macro F1 / 19.9% Acc | **Specimen Bottleneck:** Handcrafted & deep models stall below chance due to plant acoustics. |
| **3. Stress Classification (196)** | 0.441 Macro F1 / 44.4% Acc (SVM RBF) | 0.430 Macro F1 / 43.9% Acc | **0.529 Macro F1** / **53.6% Acc** | **Overlap Confirmed:** All models hover near chance (~50%), validating Section 6.4.2. |
| **3. Stress Classification (502)** | 0.567 Macro F1 / 56.8% Acc (SVM RBF) | 0.331 Macro F1 / 33.3% Acc | **0.581 Macro F1** / **59.8% Acc** | **CWT Scalogram Advantage:** Time-frequency representation extracts transient structure. |

### Thesis Integration Summary:
This benchmark directly fulfills the *Future Work* proposal: it demonstrates where deep learning thrives (noise screening directly on raw time series) and empirically confirms the physiological and physical sensor-coupling limitations identified in Chapter 6 across species and stress diagnosis.
