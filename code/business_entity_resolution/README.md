# Amazon ML Challenge: Business Entity Resolution

This repository contains an end-to-end, memory-optimized Entity Resolution pipeline designed specifically to maximize the F<sub>0.5</sub> score while processing 1GB+ of text data within strict local RAM and GPU VRAM constraints.

## Architecture Highlights

* **Blocking:** Shattered composite keys with strict frequency capping (Heavy Hitter protection) to prevent Cartesian memory explosions.
* **Feature Extraction:** Multi-core string matching using `joblib` and `rapidfuzz`, O(1) dictionary lookups (bypassing Pandas `.merge()`), and strict `float32`/`int8` downcasting for up to an 80% RAM reduction.
* **Modeling:** GPU-accelerated XGBoost utilizing `QuantileDMatrix` (`max_bin=127`) to train on millions of pairs without exceeding local VRAM limits.
* **Resilience:** Automatic Parquet checkpointing to disk to save intermediate feature matrices and bypass redundant calculations on restarts.

## 1. Environment Setup

Ensure Python 3.9+ is installed. Install the pinned dependencies required to reproduce the environment:

```bash
pip install -r requirements.txt
```

## 2. Directory Structure

Place the unzipped raw dataset folders in the root directory, mirroring this exact structure:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## 3. End-to-End Execution

Run the main pipeline script from the project root. This executes the entire pipeline sequentially:

```text
data loading → blocking → multi-core feature extraction → threshold tuning → model training → test inference
```

Run:

```bash
python code/business_entity_resolution/src/pipeline.py
```

## 4. Outputs

The pipeline automatically creates an `output/` directory and generates the exact files required for submission:

* `output/candidate_pairs.tsv`: The candidate pair set generated during blocking.
* `output/matching_results.tsv`: The final predicted matches (formatted for leaderboard submission).
* `output/xgboost_model.json`: The trained XGBoost model booster file.
* `output/*.parquet`: Intermediate feature checkpoints for fault tolerance.
