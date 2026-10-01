import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.model import EntityResolutionModel


def _parse_truth(df_truth: pd.DataFrame) -> dict:
    """Build {source1_entity_id -> set(matched_entity_ids)} , empty set = singleton."""
    df_truth = df_truth.copy()
    df_truth['matched_entity_ids'] = df_truth['matched_entity_ids'].fillna("")

    true_matches_dict = {}
    for s1, matches_str in zip(df_truth['source1_entity_id'], df_truth['matched_entity_ids']):
        matches_str = str(matches_str).strip()
        true_matches_dict[s1] = (
            {x.strip() for x in matches_str.split(',') if x.strip()}
            if matches_str else set()
        )
    return true_matches_dict


def _macro_f05_at_threshold(val_df: pd.DataFrame, true_sizes: pd.Series, t: float) -> float:
    """
    Vectorized macro F0.5 (leaderboard-style, with the singleton rule) at one threshold.
    val_df must have columns: id_1, y (0/1 true-pair label), prob.
    true_sizes: Series indexed by id_1 -> number of true matches for that id_1 (0 = singleton).
    """
    pred = val_df['prob'].values >= t
    y = val_df['y'].values.astype(bool)

    tp = pd.Series(np.where(y & pred, 1, 0), index=val_df['id_1'].values).groupby(level=0).sum()
    fp = pd.Series(np.where(~y & pred, 1, 0), index=val_df['id_1'].values).groupby(level=0).sum()

    tp = tp.reindex(true_sizes.index, fill_value=0).values
    fp = fp.reindex(true_sizes.index, fill_value=0).values
    fn = np.clip(true_sizes.values - tp, a_min=0, a_max=None)

    is_singleton = true_sizes.values == 0
    scores = np.empty(len(true_sizes))

    # Singleton rule: any predicted match at all (FP > 0) scores 0, else 1
    scores[is_singleton] = np.where(fp[is_singleton] > 0, 0.0, 1.0)

    ns = ~is_singleton
    denom_p = tp[ns] + fp[ns]
    denom_r = tp[ns] + fn[ns]
    prec = np.divide(tp[ns], denom_p, out=np.zeros(ns.sum()), where=denom_p > 0)
    rec = np.divide(tp[ns], denom_r, out=np.zeros(ns.sum()), where=denom_r > 0)
    denom = 0.25 * prec + rec
    f05 = np.divide(1.25 * prec * rec, denom, out=np.zeros(ns.sum()), where=denom > 0)
    scores[ns] = f05

    return float(scores.mean())


def _candidate_thresholds(val_probs: np.ndarray, max_candidates: int = 500) -> np.ndarray:
    """
    Every unique probability is a potential breakpoint where predictions change,
    so testing them all (rather than an arbitrary fixed-step grid) is what
    actually finds the optimal threshold instead of approximating it.
    Subsample via quantiles if there are too many for speed.
    """
    uniq = np.unique(val_probs)
    if len(uniq) <= max_candidates:
        thresholds = uniq
    else:
        qs = np.linspace(0, 1, max_candidates)
        thresholds = np.quantile(uniq, qs)
        thresholds = np.unique(thresholds)
    # also try the midpoints between consecutive values, and a couple of extremes
    thresholds = np.unique(np.concatenate([thresholds, [0.0, 1.0]]))
    return thresholds


def train_and_tune(df_features: pd.DataFrame, df_truth: pd.DataFrame, model_output_path: str) -> float:
    print("\n--- FAST 1-SPLIT MODEL TRAINING & MACRO-F0.5 TUNING ---")

    # 1. Parse ground truth
    print("Mapping ground truth pairs...")
    true_matches_dict = _parse_truth(df_truth)

    print("Labeling feature matrix...")
    pair_tuples = list(zip(df_features['id_1'], df_features['id_2']))
    y = np.array([1 if p[1] in true_matches_dict.get(p[0], set()) else 0 for p in pair_tuples], dtype=np.int8)

    feature_cols = [c for c in df_features.columns if c not in ['id_1', 'id_2']]
    X = df_features[feature_cols]

    # Sanity check: silent NaN/inf in features is a common, silent score-killer
    n_bad = int(np.isnan(X.values).sum() + np.isinf(X.values).sum())
    if n_bad > 0:
        print(f"WARNING: {n_bad} NaN/inf values found in feature matrix before training.")

    # 2. Single 80/20 group split (grouped by id_1 so an entity never straddles train/val)
    print("Splitting train/validation data...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=df_features['id_1']))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    id1_val = df_features['id_1'].iloc[val_idx].values
    id2_val = df_features['id_2'].iloc[val_idx].values

    print(f"Train pairs: {len(train_idx):,} | Validation pairs: {len(val_idx):,}")

    # 3. Train model
    model = EntityResolutionModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # 4. Threshold search
    print("\nEvaluating MACRO F_0.5 score across thresholds (Protecting Singletons)...")
    val_probs = model.predict_proba(X_val)
    if val_probs.ndim == 2:  # guard against sklearn-style [:, [P(neg), P(pos)]] output
        val_probs = val_probs[:, 1]
    val_probs = np.asarray(val_probs).ravel()

    val_df = pd.DataFrame({'id_1': id1_val, 'y': y_val, 'prob': val_probs})

    unique_val_s1 = np.unique(id1_val)
    true_sizes = pd.Series(
        {s1: len(true_matches_dict.get(s1, set())) for s1 in unique_val_s1}
    ).reindex(unique_val_s1)

    thresholds = _candidate_thresholds(val_probs)
    print(f"Testing {len(thresholds)} candidate thresholds (data-driven, not a fixed grid)...")

    best_thresh = 0.50
    best_f05 = -1.0
    for t in thresholds:
        macro_f05 = _macro_f05_at_threshold(val_df, true_sizes, t)
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = t

    print(f"\n[Optimal Threshold Selected]: {best_thresh:.4f} (Macro F_0.5: {best_f05:.4f})")

    # Optional: sanity-check threshold stability via bootstrap resampling of the val set.
    # A threshold picked from one small/unlucky split can overfit; this shows how much
    # the chosen threshold moves if the val set were slightly different.
    n_boot = 20
    boot_threshs = []
    rng = np.random.default_rng(42)
    n = len(val_df)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_val_df = val_df.iloc[sample_idx]
        boot_s1 = np.unique(boot_val_df['id_1'].values)
        boot_true_sizes = true_sizes.reindex(boot_s1)
        b_best_t, b_best_f = 0.50, -1.0
        for t in thresholds:
            f = _macro_f05_at_threshold(boot_val_df, boot_true_sizes, t)
            if f > b_best_f:
                b_best_f, b_best_t = f, t
        boot_threshs.append(b_best_t)
    print(
        f"Bootstrap threshold stability (n={n_boot}): "
        f"mean={np.mean(boot_threshs):.4f}, std={np.std(boot_threshs):.4f}, "
        f"range=[{np.min(boot_threshs):.4f}, {np.max(boot_threshs):.4f}]"
    )
    if np.std(boot_threshs) > 0.05:
        print("NOTE: threshold is fairly unstable across resamples — consider a larger val "
            "split or averaging thresholds across multiple GroupShuffleSplit seeds.")

    model.save(model_output_path)
    return best_thresh