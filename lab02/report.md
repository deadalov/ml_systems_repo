# Lab 2: Two-Hidden-Layer Neural Network from Scratch (load_digits)

**Student:** Abdulla Kaldybek (ID: 73687)  
**Course:** Introduction to Machine Learning Systems  

---

## 1. Overview & Architecture
This repository contains a complete NumPy implementation of a two-hidden-layer neural network trained on the scikit-learn load_digits dataset (8x8 grayscale images of handwritten digits, N = 1797, D = 64, K = 10).

* **Input Layer:** D = 64 features (normalized to [0, 1] by dividing by 16.0)
* **Hidden Layer 1:** H1 = 64 units, ReLU activation, He initialization
* **Hidden Layer 2:** H2 = 32 units, ReLU activation, He initialization
* **Output Layer:** K = 10 units, numerically stable Softmax activation
* **Loss Function:** Categorical Cross-Entropy Loss
* **Optimization:** Batch Gradient Descent (learning rate = 0.5, 500 epochs)

---

## 2. Numerical Gradient Check (Graded Deliverable #1)
Analytical gradients from backpropagation were verified against centered finite-difference numerical gradients (eps = 1e-5). All parameters achieved relative errors well below the 1e-6 threshold:

| Parameter | Shape | Relative Error | Status |
| :--- | :--- | :--- | :--- |
| W1 | (64, 8) | 9.83e-11 | **PASS** |
| b1 | (1, 8) | 6.88e-11 | **PASS** |
| W2 | (8, 6) | 1.79e-10 | **PASS** |
| b2 | (1, 6) | 7.87e-11 | **PASS** |
| W3 | (6, 10) | 5.06e-10 | **PASS** |
| b3 | (1, 10) | 9.25e-11 | **PASS** |

**Overall Gradient Check:** PASSED

---

## 3. Training Progress & Loss Curves (Graded Deliverable #2)

| Epoch | Train Loss | Validation Loss | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **1 / 500** | 2.5939 | 2.2434 | 15.80% | 17.22% |
| **100 / 500** | 0.1811 | 0.4206 | 93.81% | 85.28% |
| **200 / 500** | 0.0314 | 0.0879 | 99.51% | 97.50% |
| **300 / 500** | 0.0168 | 0.0801 | 99.86% | 97.78% |
| **400 / 500** | 0.0107 | 0.0758 | 99.93% | 97.78% |
| **500 / 500** | 0.0075 | 0.0735 | 100.00% | 97.78% |

### Training & Validation Loss Curve
![Loss Curve](loss_curve.png)

---

## 4. Computational Timing Breakdown (Graded Deliverable #3)
Wall-clock execution time across 500 epochs was profiled using time.perf_counter():

| Stage | Time (s) | Percentage (%) |
| :--- | :--- | :--- |
| **Forward Pass** | 0.1549 s | 44.8% |
| **Loss Computation** | 0.0306 s | 8.9% |
| **Backward Pass** | 0.1561 s | 45.2% |
| **Parameter Update** | 0.0037 s | 1.1% |
| **Total** | **0.3454 s** | **100.0%** |

---

## 5. How to Run
    pip install numpy matplotlib scikit-learn
    python3 lab2.py
