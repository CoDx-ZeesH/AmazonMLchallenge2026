import pandas as pd

def generate_predictions(model, features_df, threshold, s1_id_col='id_1', s2_id_col='id_2'):
    """
    Generates thresholded binary predictions from candidate pair features.
    """
    # 1. Predict match probability
    features_df = features_df.copy()
    feature_cols = [col for col in features_df.columns if col not in [s1_id_col, s2_id_col] and not col.endswith('_1') and not col.endswith('_2')]
    
    probas = model.predict_proba(features_df[feature_cols])
    
    # 2. Apply threshold
    predictions = {}
    for i, proba in enumerate(probas):
        s1_id = features_df.iloc[i][s1_id_col]
        if s1_id not in predictions:
            predictions[s1_id] = []
            
        if proba >= threshold:
            predictions[s1_id].append(features_df.iloc[i][s2_id_col])
            
    # 3. Format as required for submission (comma-separated strings)
    results = [{'source1_entity_id': k, 'matched_entity_ids': ','.join(v)} for k, v in predictions.items()]
    return pd.DataFrame(results)
