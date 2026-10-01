# Methodology

## 1. Data Preprocessing

Missing values were imputed with empty strings to guarantee seamless string operations and prevent Pandas from silently dropping valid entities. We utilized vectorized Pandas string methods to strip whitespaces, standardizing all text to lowercase, and regex to isolate the first continuous block of digits from the address column.

## 2. Candidate Generation (Blocking)

To prevent Out-Of-Memory (OOM) failures caused by the naive Cartesian product of a 1GB dataset, we implemented "Shattered Composite Keys."

* **Key Generation:** We created hyper-specific blocking keys by combining attributes, such as `Country + First Numeric Address Token + First 2 Characters of Name`.
* **Heavy Hitter Capping:** We implemented a strict frequency cap on the blocking keys. Any key appearing more than 50 times in Source 2 or Source 3 was nullified. This eliminated generic clusters (e.g., massive office complexes or shell-company registries) that cause RAM-crashing Cartesian explosions, forcefully bounding our candidate pairs down to a manageable few million.
* **Execution:** Progressive C-level Pandas hash joins were used across three distinct composite keys to safely union the candidate lists.

## 3. Feature Extraction

Because generating features for millions of candidate pairs is highly CPU-bound, we bypassed Pandas entirely during extraction:

* **O(1) Lookups:** The raw text dataset was mapped to native Python dictionaries, allowing instant O(1) text retrieval for candidate pairs without memory-spiking Pandas `.merge()` operations.
* **Multi-Core Parallelization:** We chunked the candidate arrays and used `joblib` to distribute `rapidfuzz` calculations (Ratio, Token Sort, Token Set) across all available physical and logical CPU cores.
* **Memory Downcasting:** String distance arrays were cast directly to `np.float32` inside the multiprocessing loops, and length differences to `np.int8`. This reduced the feature matrix RAM footprint by approximately 75%.

## 4. Model Training & Validation

We formulated the task as a binary classification problem to output calibrated match probabilities.

* **Algorithm:** XGBoost was configured to emulate LightGBM's mathematical structure (`grow_policy='lossguide'`, `max_leaves=63`) for optimal precision on highly imbalanced data.
* **Hardware Acceleration:** Training was offloaded to the NVIDIA GPU (`tree_method='hist'`, `device='cuda'`).
* **VRAM Optimization:** To prevent GPU Out-Of-Memory errors on the massive dense feature matrix, we utilized XGBoost's `QuantileDMatrix` locked to `max_bin=127`. This dynamically binned the data on the CPU, slashing the GPU memory footprint by 80%. Standard `DMatrix` was preserved for inference to guarantee exact bin mapping.
* **Validation:** GroupKFold cross-validation was used (grouped by `source1_entity_id`) to prevent data leakage between folds.

## 5. Threshold Optimization for F_0.5

Because the evaluation metric (F_0.5) penalizes False Positives twice as heavily as False Negatives, default probability thresholding (0.50) is insufficient. We utilized the Out-Of-Fold (OOF) predictions from our cross-validation loop to perform a deterministic sweep of probability thresholds (0.50 to 0.95). The pipeline mathematically scores the F_0.5 metric across these thresholds and strictly applies the highest-scoring cutoff to the blind test set, ensuring Precision is aggressively protected.

## 6. Fault Tolerance & Memory Management

The pipeline integrates Parquet checkpointing. Intermediate multi-million row feature matrices are cached to disk via PyArrow, wrapped in I/O fault-tolerant `try/except` blocks. If execution is interrupted during modeling, the pipeline instantly reloads the checkpoints, saving hours of redundant string calculation. Finally, strict garbage collection (`gc.collect()`) and explicit variable deletion (`del`) were enforced after every pipeline stage to maintain a flat memory profile.
