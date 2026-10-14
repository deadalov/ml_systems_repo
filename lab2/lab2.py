import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. Dataset Preparation (load_digits)
# ==========================================
def load_and_prep_data():
    digits = load_digits()
    X = digits.data / 16.0  # Normalize [0, 16] -> [0, 1]
    y = digits.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y)

    return train_test_split(
        X, Y, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )

# ==========================================
# 2. Network Architecture & Initialization
# ==========================================
def init_params(D=64, H1=64, H2=32, K=10, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / D), (D, H1)),
        "b1": np.zeros((1, H1)),
        "W2": rng.normal(0, np.sqrt(2.0 / H1), (H1, H2)),
        "b2": np.zeros((1, H2)),
        "W3": rng.normal(0, np.sqrt(2.0 / H2), (H2, K)),
        "b3": np.zeros((1, K)),
    }

# ==========================================
# 3. Forward Propagation & Cross-Entropy Loss
# ==========================================
def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(0, Z1)  # ReLU

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = np.maximum(0, Z2)  # ReLU

    Z3 = A2 @ params["W3"] + params["b3"]
    # Numerically stable Softmax
    shift_Z3 = Z3 - np.max(Z3, axis=1, keepdims=True)
    exp_scores = np.exp(shift_Z3)
    Y_hat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "Y_hat": Y_hat}
    return Y_hat, cache

def compute_loss(Y_hat, Y):
    N = Y.shape[0]
    return -np.sum(Y * np.log(Y_hat + 1e-12)) / N

# ==========================================
# 4. Backpropagation
# ==========================================
def backward(Y, params, cache):
    N = Y.shape[0]

    dZ3 = (cache["Y_hat"] - Y) / N
    dW3 = cache["A2"].T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

# ==========================================
# 5. Numerical Gradient Checking (Graded #1)
# ==========================================
def gradient_check(X_sample, Y_sample, eps=1e-5):
    print("\n" + "="*50)
    print("1. NUMERICAL GRADIENT CHECK")
    print("="*50)
    params = init_params(D=X_sample.shape[1], H1=8, H2=6, K=10, seed=1)
    _, cache = forward(X_sample, params)
    grads = backward(Y_sample, params, cache)

    all_passed = True
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        num_grad = np.zeros_like(params[name])
        it = np.nditer(params[name], flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = params[name][idx]

            params[name][idx] = orig + eps
            loss_plus = compute_loss(forward(X_sample, params)[0], Y_sample)

            params[name][idx] = orig - eps
            loss_minus = compute_loss(forward(X_sample, params)[0], Y_sample)

            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            params[name][idx] = orig
            it.iternext()

        rel_err = np.linalg.norm(grads[name] - num_grad) / max(
            np.linalg.norm(grads[name]), np.linalg.norm(num_grad), 1e-8
        )
        status = "PASS" if rel_err < 1e-6 else "FAIL"
        if rel_err >= 1e-6:
            all_passed = False
        print(f"Param {name:2s} | Relative Error: {rel_err:.2e} | [{status}]")

    print("-" * 50)
    print(f"Overall Gradient Check: {'PASSED' if all_passed else 'FAILED'}")

# ==========================================
# 6. Prediction & Accuracy (Reddi §3.6)
# ==========================================
def predict(X, params):
    Y_hat, _ = forward(X, params)
    return np.argmax(Y_hat, axis=1)

def accuracy(X, y_true, params):
    preds = predict(X, params)
    return np.mean(preds == y_true) * 100.0

# ==========================================
# 7. Training & Timing Breakdown (Graded #2 & #3)
# ==========================================
def train_model(X_train, Y_train, y_train, X_test, Y_test, y_test, epochs=500, lr=0.5):
    print("\n" + "="*50)
    print("2. MODEL TRAINING & PROGRESS")
    print("="*50)
    params = init_params(D=64, H1=64, H2=32, K=10, seed=42)

    train_losses, val_losses = [], []
    timings = {"forward": 0.0, "loss": 0.0, "backward": 0.0, "update": 0.0}

    for epoch in range(1, epochs + 1):
        # Forward timing
        t0 = time.perf_counter()
        Y_hat, cache = forward(X_train, params)
        timings["forward"] += time.perf_counter() - t0

        # Loss timing
        t0 = time.perf_counter()
        tr_loss = compute_loss(Y_hat, Y_train)
        timings["loss"] += time.perf_counter() - t0

        # Backward timing
        t0 = time.perf_counter()
        grads = backward(Y_train, params, cache)
        timings["backward"] += time.perf_counter() - t0

        # Update timing
        t0 = time.perf_counter()
        for k in params:
            params[k] -= lr * grads[k]
        timings["update"] += time.perf_counter() - t0

        # Record validation loss
        val_Y_hat, _ = forward(X_test, params)
        val_loss = compute_loss(val_Y_hat, Y_test)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if epoch % 100 == 0 or epoch == 1:
            tr_acc = accuracy(X_train, y_train, params)
            val_acc = accuracy(X_test, y_test, params)
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Timing Summary
    total_time = sum(timings.values())
    print("\n" + "="*50)
    print("3. COMPUTATIONAL TIMING BREAKDOWN")
    print("="*50)
    for stage, t_sec in timings.items():
        pct = (t_sec / total_time) * 100
        print(f"{stage.capitalize():10s}: {t_sec:.4f} s ({pct:5.1f}%)")
    print("-" * 50)
    print(f"{'Total':10s}: {total_time:.4f} s (100.0%)")

    # Save Loss Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2, linestyle="--")
    plt.title("Two-Hidden-Layer Neural Network Loss Curve (load_digits)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("\n[+] Saved loss curve plot to: loss_curve.png")

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_and_prep_data()
    gradient_check(X_train[:5], Y_train[:5])
    train_model(X_train, Y_train, y_train, X_test, Y_test, y_test, epochs=500, lr=0.5)
