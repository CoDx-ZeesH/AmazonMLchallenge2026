import pandas as pd
import numpy as np

# ==========================================
# metrics.py (V2 - Dictionary Bypass)
# ==========================================

def _parse_id_string(id_str: str) -> set:
    """Safely converts a comma-separated string into a Python set."""
    if not isinstance(id_str, str) or not id_str.strip():
        return set()
    return set(id_str.strip().split(','))

def _calculate_row_f05(true_set: set, pred_set: set) -> float:
    """Calculates the F_0.5 score using pre-parsed sets."""
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
        
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
        
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return (1.25 * precision * recall) / ((0.25 * precision) + recall)

def compute_macro_f05(df_truth: pd.DataFrame, df_preds: pd.DataFrame) -> float:
    """
    Calculates Macro-Averaged F_0.5 bypassing pd.merge for O(1) alignment.
    """
    # Direct dictionary mapping bypassing Pandas merge overhead
    truth_dict = dict(zip(df_truth['source1_entity_id'], df_truth['matched_entity_ids'].fillna("")))
    pred_dict = dict(zip(df_preds['source1_entity_id'], df_preds['matched_entity_ids'].fillna("")))
    
    # O(1) lookup and calculation loop
    scores = [
        _calculate_row_f05(
            _parse_id_string(truth_val),
            _parse_id_string(pred_dict.get(s1_id, ""))
        )
        for s1_id, truth_val in truth_dict.items()
    ]
    
    return float(np.mean(scores))