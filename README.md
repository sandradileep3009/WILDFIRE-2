# AI-Powered Drone-Based Wildfire Monitoring and Dynamic Fire Spread Prediction

## Overview

This project investigates **wildfire spread prediction from sequential aerial imagery** using deep learning.

Most wildfire detection systems focus on identifying whether fire is present in an image. This project goes a step further by modeling the **temporal evolution of the fire region** and predicting its spatial extent at the next timestep.

The core research question is:

> **Given a sequence of aerial observations of an evolving wildfire, can a temporal deep-learning model predict the fire region at the next timestep?**

The proposed approach uses a **ConvLSTM-based model** to learn both spatial and temporal patterns in wildfire progression.

---

## Research Pipeline

```text
Aerial Fire Images
        ↓
Dataset Audit & Preprocessing
        ↓
Image–Mask Matching
        ↓
Temporal Sequence Construction
        ↓
ConvLSTM Growth Prediction
        ↓
Predicted Future Fire Mask
        ↓
Quantitative + Qualitative Evaluation
```

---

## Methodology

### 1. Fire Region Representation

Wildfire images are associated with spatial fire masks representing the location of fire within the image.

Instead of predicting only:

```text
Fire / No Fire
```

the growth prediction task operates on:

```text
Fire Region Mask
```

This allows the model to learn changes in the **shape and spatial extent** of the fire.

---

### 2. Temporal Sequence Construction

Individual images are organized according to their temporal/frame order.

For a sequence length of 3:

```text
Frame t-2 ─┐
Frame t-1 ─┼──→ ConvLSTM ──→ Predicted Mask at t+1
Frame t   ─┘
```

The model therefore receives multiple observations of the evolving fire rather than a single image.

The current experimental setup uses:

* **Sequence length:** 3 frames
* **Prediction horizon:** 1 future timestep

---

### 3. ConvLSTM

The primary growth prediction model is a **Convolutional LSTM (ConvLSTM)**.

ConvLSTM is used because it combines:

* Convolutional operations for spatial information
* LSTM recurrence for temporal information

Conceptually:

```text
Spatial information
        +
Temporal information
        ↓
     ConvLSTM
        ↓
Future fire-region prediction
```

The model outputs a probability map representing the predicted future fire region.

---

## Loss Function

The growth model uses a combination of **Binary Cross Entropy and Dice-based loss**.

This is intended to address the highly imbalanced nature of fire-mask segmentation, where background pixels can substantially outnumber fire pixels.

The objective is to encourage both:

* Pixel-level classification accuracy
* Spatial overlap between predicted and ground-truth fire regions

---

## Data Leakage Prevention

Because wildfire observations are temporally related, randomly splitting individual frames can cause temporal leakage.

The project therefore uses **run-level splitting**, grouping temporally related sequences before creating training and evaluation sets.

This prevents nearly identical neighboring frames from appearing across different splits.

---

## Evaluation

The model is evaluated using spatial overlap metrics, primarily:

### Intersection over Union (IoU)

$$
IoU = \frac{|P \cap G|}{|P \cup G|}
$$

where \(P\) is the predicted fire region and \(G\) is the ground-truth region.

### Dice Coefficient

$$
Dice = \frac{2|P \cap G|}{|P| + |G|}
$$

Both metrics evaluate how closely the predicted fire region overlaps with the actual fire region.

The project also performs **qualitative visualization** to examine cases where the model under-predicts, over-predicts, or incorrectly localizes fire growth.

---

## Baseline Comparison

The ConvLSTM approach is evaluated against simpler prediction strategies to determine whether explicitly modeling temporal information provides an advantage.

The evaluation includes comparisons with non-temporal baselines and a persistence-based approach where appropriate.

---

## Project Structure

```text
Wildfire_Growth_Prediction/
│
├── models/
│   ├── convlstm.py
│   └── segmentation.py
│
├── preprocessing/
│   └── ...
│
├── training/
│   ├── losses.py
│   ├── train_fullmask.py
│   └── train_growth.py
│
├── utils/
│   ├── seed.py
│   └── splits.py
│
├── config.py
├── evaluation_growth.py
├── final_evaluation.py
├── plot_research_results.py
├── run_multiseed.py
├── visualize_predictions.py
│
├── README.md
└── requirements.txt
```


## Future Work

Potential extensions include:

* Integration of meteorological and geographic variables
* Longer temporal sequences
* Evaluation on independent wildfire events
* Comparison with additional temporal architectures

---

## Author

**Sandra Dileep**

---
