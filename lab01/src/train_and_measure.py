import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Загрузка и разбиение данных
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

process = psutil.Process(os.getpid())
results = []
accuracies = []

os.makedirs("lab01/results", exist_ok=True)

for name, model in models.items():
    # Warm-up (1 запуск)
    model.fit(X_train, y_train)
    
    # Замер времени обучения (5 запусков) и памяти
    train_times = []
    mem_before_train = process.memory_info().rss
    
    for _ in range(5):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
        
    mem_after_train = process.memory_info().rss
    median_train_time = np.median(train_times)
    peak_train_mem_kb = max(0, (mem_after_train - mem_before_train) / 1024)

    # Точность на тесте
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append({"model": name, "accuracy": f"{acc:.4f}"})

    # Сохранение и замер размера модели
    model_path = f"lab01/results/{name}.joblib"
    joblib.dump(model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    # Замер инференса (100 повторений) и памяти инференса
    single_sample = X_test[0:1]
    _ = model.predict(single_sample)  # warm-up инференса
    
    inf_times = []
    mem_before_inf = process.memory_info().rss
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(single_sample)
        inf_times.append(time.perf_counter() - t0)
    mem_after_inf = process.memory_info().rss
    
    median_inf_latency_ms = np.median(inf_times) * 1000
    peak_inf_mem_kb = max(0, (mem_after_inf - mem_before_inf) / 1024)

    results.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "Train Time (s)": round(median_train_time, 4),
        "Inference Latency (ms)": round(median_inf_latency_ms, 4),
        "Size (Bytes)": size_bytes,
        "Size (KB)": round(size_kb, 2),
        "Train Peak Mem (KB)": round(peak_train_mem_kb, 2),
        "Inference Peak Mem (KB)": round(peak_inf_mem_kb, 2),
    })

# Сохранение accuracy
pd.DataFrame(accuracies).to_csv("lab01/results/baseline_accuracy.csv", index=False)

# Вывод сводной таблицы замеров
df_res = pd.DataFrame(results)
print("\n--- System Cost Measurements ---")
print(df_res.to_markdown(index=False))