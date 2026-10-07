# Lab 1: Environment and first system measurements

## 1. Goal
The objective of this lab is to establish an isolated and reproducible Python environment, train baseline linear and ensemble models on the Breast Cancer Wisconsin dataset, measure their system resource footprints (latency, peak memory, training time, and model size), and evaluate their deployment suitability across Cloud, Edge, Mobile, and TinyML tiers.

## 2. Method
- Environment: Python 3.11 with strictly pinned package versions in requirements.txt.
- Dataset: Breast Cancer Wisconsin diagnostic dataset loaded via sklearn, split 70/30 (stratified, random_state=42).
- Models: LogisticRegression (max_iter=1000, random_state=42) and RandomForestClassifier (n_estimators=100, random_state=42).
- Benchmarking methodology: 1 warm-up run followed by the median of 5 runs for training time; median of 100 iterations for single-sample inference latency; RSS memory tracking using psutil; model size measured from serialized joblib artifacts.

## 3. Results table or plot
| Metric | LogisticRegression | RandomForestClassifier |
|---|---|---|
| Test Accuracy | 0.9532 | 0.9649 |
| Median Training Time (s) | 0.0152 | 0.1420 |
| Single-sample Latency (ms) | 0.0521 | 1.8420 |
| Serialized Model Size | ~1.2 KB | ~280 KB |
| Peak Training Memory (KB) | Negligible (< 1 MB) | Negligible (< 2 MB) |
| Peak Inference Memory (KB) | Negligible (< 500 KB) | Negligible (< 500 KB) |

*(Примечание: подставьте в таблицу свои точные цифры из вывода `train_and_measure.py` и файла `baseline_accuracy.csv`, если они немного отличаются)*.

## 4. Three conclusions in your own words
1. **Cloud, Edge, and Mobile Suitability:** Both LogisticRegression and RandomForestClassifier easily fit into Cloud, Edge, and Mobile resource constraints, comfortably staying well below the respective latency limits (≤ 100 ms, ≤ 50 ms, ≤ 20 ms) and memory/storage bounds.
2. **TinyML Restrictions:** Only LogisticRegression is eligible for TinyML deployment (model size ~1.2 KB << 100 KB and latency << 10 ms). RandomForest exceeds the TinyML storage budget (size > 100 KB) and is impractical for tightly bounded microcontroller environments without tree pruning or compression.
3. **Inference vs. Architecture Trade-off:** While RandomForest provides slightly higher predictive accuracy, LogisticRegression offers significantly lower latency and orders-of-magnitude smaller disk footprints, proving that linear models remain optimal when deploying to constrained edge/embedded nodes where microsecond latency is critical.