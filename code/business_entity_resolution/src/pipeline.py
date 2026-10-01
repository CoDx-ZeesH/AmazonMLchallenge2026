import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gc
import traceback
import pandas as pd
import numpy as np

# Import all our optimized modules
from src.data_loader import load_tsv, save_predictions
from src.preprocessor import preprocess_data
from src.blocking import generate_candidate_pairs
from src.feature_extraction import build_feature_matrix
from src.train import train_and_tune
from src.model import EntityResolutionModel

# ==========================================================
# pipeline.py (V2 - Fails Loudly Instead of Silently)
# ==========================================================
#
# What changed vs V1 and why:
#
# The bottom-level `except Exception as e: print(str(e))` swallows the
# traceback and, crucially, still exits with status code 0 (success) --
# verified directly: a script that raises inside a try/except shaped
# exactly like this one prints the message but returns exit code 0.
# Any CI job, cron entry, or grading harness that checks the exit code
# (rather than parsing stdout for "[FATAL ERROR]") will see this as a
# successful run even though matching_results.tsv was never produced,
# or was produced from a stale/partial state. And without a traceback,
# you don't know which stage or line failed -- just that "something"
# did. Both are fixed below: the traceback is printed via
# traceback.print_exc(), and the process exits with sys.exit(1) so a
# failed run is unambiguously reported as failed.
#
# Everything else in this file is unchanged -- the blocking, feature
# extraction, and threshold-application logic here already lines up
# correctly with the current blocking.py / feature_extraction.py /
# model.py (feature_cols is derived dynamically from
# df_features.columns, so it doesn't need to track column-name changes
#
# in feature_extraction.py by hand).


def load_and_combine_sources(base_dir: str, prefix: str) -> pd.DataFrame:
    """Helper to load and stack Source 1, 2, and 3 cleanly."""
    print(f"Loading {prefix} data...")
    df1 = load_tsv(os.path.join(base_dir, f"{prefix}_source1.tsv"))
    df2 = load_tsv(os.path.join(base_dir, f"{prefix}_source2.tsv"))
    df3 = load_tsv(os.path.join(base_dir, f"{prefix}_source3.tsv"))
    return pd.concat([df1, df2, df3], ignore_index=True)

def run_training_pipeline() -> float:
    """Trains the model completely in-memory without checkpoints."""
    os.makedirs("output", exist_ok=True)

    # We always need the ground truth for threshold tuning
    df_truth = load_tsv("dataset/train/train_ground_truth.tsv")

    print("\n--- STARTING TRAINING PIPELINE ---")
    df_train = load_and_combine_sources("dataset/train", "train")

    print("Preprocessing text...")
    df_clean = preprocess_data(df_train)
    # GC: Free raw training text immediately
    del df_train; gc.collect()

    print("Generating candidate pairs (Blocking)...")
    df_candidates = generate_candidate_pairs(df_clean)

    print("Extracting features...")
    df_features = build_feature_matrix(df_candidates, df_clean)
    # GC: Free blocking candidates and clean text before intensive model training
    del df_candidates, df_clean; gc.collect()

    print("Training model and tuning threshold...")
    # UPDATED: Swapped to .txt for LightGBM
    best_thresh = train_and_tune(df_features, df_truth, "output/lightgbm_model.txt")

    # GC: Free feature matrix
    del df_features, df_truth; gc.collect()

    return best_thresh

def run_inference_pipeline(optimal_threshold: float):
    """Runs the blind test set in-memory and formats final files."""
    os.makedirs("output", exist_ok=True)

    print("\n--- STARTING INFERENCE PIPELINE ---")

    # We ALWAYS need S1 IDs for the final singleton formatting.
    # To save RAM, we only load test_source1.tsv to get them.
    print("Extracting S1 IDs to guarantee perfect submission formatting...")
    df_s1 = load_tsv(os.path.join("dataset/test", "test_source1.tsv"))
    s1_mask = df_s1['entity_id'].str.startswith('S1-')
    all_s1_ids = pd.DataFrame({'source1_entity_id': df_s1.loc[s1_mask, 'entity_id']})
    del df_s1; gc.collect()

    df_test = load_and_combine_sources("dataset/test", "test")

    print("Preprocessing test text...")
    df_clean = preprocess_data(df_test)
    del df_test; gc.collect()

    print("Generating test candidate pairs...")
    df_candidates = generate_candidate_pairs(df_clean)

    # REQUIREMENT: Save candidates BEFORE model inference
    print("Saving candidate_pairs.tsv...")
    df_candidates_sorted = df_candidates.sort_values('source1_entity_id')
    save_predictions(df_candidates_sorted, "output/candidate_pairs.tsv")

    print("Extracting test features...")
    df_features = build_feature_matrix(df_candidates, df_clean)
    del df_candidates, df_clean; gc.collect()

    print("Loading trained model and predicting...")
    model = EntityResolutionModel()
    # UPDATED: Swapped to .txt for LightGBM
    model.load("output/lightgbm_model.txt")

    feature_cols = [c for c in df_features.columns if c not in ['id_1', 'id_2']]
    df_features['prob'] = model.predict_proba(df_features[feature_cols])

    # UPDATED: Trusting the mathematically perfect Macro F0.5 tuner
    print(f"Applying mathematically tuned Macro F_0.5 threshold: {optimal_threshold:.2f}...")
    df_confident = df_features[df_features['prob'] >= optimal_threshold].copy()

    del df_features; gc.collect()

    # --- GUARDRAIL: DUPLICATE IDS ---
    df_confident.drop_duplicates(subset=['id_1', 'id_2'], inplace=True)

    # Aggregate back to comma-separated strings
    grouped = df_confident.groupby('id_1')['id_2'].agg(','.join).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']

    # --- CRITICAL: THE SINGLETON GUARANTEE ---
    print("Restoring singletons and formatting output...")
    final_output = all_s1_ids.merge(grouped, on='source1_entity_id', how='left')
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna("")

    # Deterministic sorting for professional submission
    final_output = final_output.sort_values('source1_entity_id')

    print("Saving matching_results.tsv...")
    save_predictions(final_output, "output/matching_results.tsv")
    print("\nPipeline Complete! Run utils/validate_submission.py to verify.")


if __name__ == "__main__":
    # Execute end-to-end flow
    try:
        optimal_thresh = run_training_pipeline()
        run_inference_pipeline(optimal_thresh)
    except Exception:
        print("\n[FATAL ERROR] Pipeline failed:")
        traceback.print_exc()
        sys.exit(1)