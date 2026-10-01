import pandas as pd
import numpy as np

# ==========================================================
# blocking.py (V5 - Token-Aware, Precision-Tunable Blocking)
# ==========================================================
#
# What changed vs V4 and why:
#
# 1. TOKEN-BASED KEYS (k4, k5)
#    V4 only used positional prefixes (first N chars of the name /
#    address). That misses any pair where words are reordered or an
#    extra word is inserted ("Burger King" vs "King Burger Grill").
#    k4/k5 explode name/address into individual tokens and match on
#    ANY shared informative token (inverted-index blocking), which is
#    the standard fix for word-order and word-insertion variance.
#
# 2. STOPWORD / SUFFIX STRIPPING
#    Legal suffixes ("inc", "llc", "corp", "ltd", "co", "group") and
#    generic words are removed before building prefix AND token keys.
#    Without this, "X Corp" and "Y Corp" can collide on a k1 prefix
#    that is really just "corp", and token keys get flooded with
#    near-universal tokens.
#
# 3. SOFTER HEAVY-HITTER CONTROL
#    V4 fully deletes a key value once it appears >150 times in S23.
#    That is an all-or-nothing recall loss for every S1 record that
#    would have matched through it. V5 keeps the same idea but makes
#    the threshold a parameter, applies it consistently to every key
#    (prefix + token), and lets you separately cap the *result*
#    (max candidates kept per source1 record) so one noisy key can't
#    blow up a single record's block even if it survives the
#    frequency filter.
#
# 4. BUILT-IN QUALITY EVALUATOR
#    evaluate_blocking() computes Pairs Completeness (recall of true
#    matches captured) and Reduction Ratio (how much smaller than the
#    full cartesian product your candidate set is) against a labeled
#    ground-truth pair list, if you have one (even a partial/sampled
#    one). This is the standard blocking-quality pair of metrics and
#    lets you tune thresholds instead of guessing.
#
# Schema assumption (same as V4): df_clean has columns
#   entity_id, business_name_clean, business_address_clean, country_clean
# entity_id values are prefixed "S1-" for source 1, anything else is
# treated as source 2/3. If you also have postal_code / phone columns,
# pass their names in `extra_key_cols` and they'll be folded into an
# extra exact-match key automatically.

STOPWORDS = {
    "inc", "llc", "corp", "corporation", "co", "company", "ltd",
    "limited", "group", "the", "and", "of", "restaurant", "store",
    "shop", "llp", "plc", "gmbh", "sa", "srl",
}


def _strip_stopwords(series: pd.Series) -> pd.Series:
    """Remove generic/legal-suffix tokens before keying, join back with spaces."""
    def _clean(s):
        if not s:
            return s
        toks = [t for t in s.split() if t not in STOPWORDS]
        return " ".join(toks) if toks else s  # never fully empty out a name
    return series.apply(_clean)


def _cap_heavy_hitters(df: pd.DataFrame, key_col: str, threshold: int) -> pd.DataFrame:
    """Nullify any key value that appears more than `threshold` times (generic/noisy key)."""
    counts = df[key_col].value_counts()
    heavy = counts[counts > threshold].index
    if len(heavy):
        df.loc[df[key_col].isin(heavy), key_col] = pd.NA
    return df


def _build_token_frame(df: pd.DataFrame, text_col: str, id_col: str,
                        min_token_len: int, heavy_threshold: int) -> pd.DataFrame:
    """
    Melt a text column into one row per (entity_id, token) for inverted-index
    blocking: two records are candidates if they share ANY surviving token.
    """
    tokens = df[text_col].str.split()
    exploded = pd.DataFrame({
        id_col: df[id_col].repeat(tokens.str.len().fillna(0).astype(int)),
        "token": [t for sub in tokens.dropna() for t in sub],
    })
    exploded = exploded[exploded["token"].str.len() >= min_token_len]
    exploded = _cap_heavy_hitters(exploded, "token", heavy_threshold)
    return exploded.dropna(subset=["token"])


def generate_candidate_pairs(
    df_clean: pd.DataFrame,
    heavy_hitter_threshold: int = 150,
    token_heavy_hitter_threshold: int = 200,
    max_candidates_per_s1_record: int = 500,
    name_prefix_len: int = 5,
    min_key_len: int = 5,
    min_token_len: int = 3,
    extra_key_cols: list | None = None,
) -> pd.DataFrame:
    """
    Generates candidate pairs using a mix of positional composite keys
    (fast, precise for near-identical prefixes) and token-based inverted
    index keys (recall-boosting for reordered/partial matches), with
    heavy-hitter capping applied at both the key level and, as a backstop,
    the per-record result level.
    """
    s1_mask = df_clean["entity_id"].str.startswith("S1-")
    df_s1 = df_clean[s1_mask].copy()
    df_s23 = df_clean[~s1_mask].copy()

    df_s1.fillna("", inplace=True)
    df_s23.fillna("", inplace=True)

    # --- Stopword-stripped working columns (used for keys only, never for output) ---
    df_s1["_name_core"] = _strip_stopwords(df_s1["business_name_clean"].str.lower())
    df_s23["_name_core"] = _strip_stopwords(df_s23["business_name_clean"].str.lower())

    name_no_space_s1 = df_s1["_name_core"].str.replace(r"\s+", "", regex=True)
    name_no_space_s23 = df_s23["_name_core"].str.replace(r"\s+", "", regex=True)

    df_s1["first_num"] = df_s1["business_address_clean"].str.extract(r"(\d+)")[0].fillna("")
    df_s23["first_num"] = df_s23["business_address_clean"].str.extract(r"(\d+)")[0].fillna("")

    df_s1["last_word"] = df_s1["_name_core"].str.split().str[-1].fillna("")
    df_s23["last_word"] = df_s23["_name_core"].str.split().str[-1].fillna("")

    # --- Composite prefix keys (same intent as V4, on stopword-stripped names) ---
    df_s1["k1"] = df_s1["country_clean"] + "_" + name_no_space_s1.str[:name_prefix_len]
    df_s23["k1"] = df_s23["country_clean"] + "_" + name_no_space_s23.str[:name_prefix_len]

    df_s1["k2"] = df_s1["country_clean"] + "_" + df_s1["first_num"] + "_" + name_no_space_s1.str[:2]
    df_s23["k2"] = df_s23["country_clean"] + "_" + df_s23["first_num"] + "_" + name_no_space_s23.str[:2]

    df_s1["k3"] = df_s1["country_clean"] + "_" + df_s1["last_word"] + "_" + df_s1["business_address_clean"].str[:2]
    df_s23["k3"] = df_s23["country_clean"] + "_" + df_s23["last_word"] + "_" + df_s23["business_address_clean"].str[:2]

    # nullify degenerate keys
    df_s1.loc[df_s1["first_num"] == "", "k2"] = pd.NA
    df_s23.loc[df_s23["first_num"] == "", "k2"] = pd.NA
    df_s1.loc[df_s1["last_word"].str.len() < 3, "k3"] = pd.NA
    df_s23.loc[df_s23["last_word"].str.len() < 3, "k3"] = pd.NA

    for k in ["k1", "k2", "k3"]:
        df_s1.loc[df_s1[k].str.len() < min_key_len, k] = pd.NA
        df_s23.loc[df_s23[k].str.len() < min_key_len, k] = pd.NA
        df_s23 = _cap_heavy_hitters(df_s23, k, heavy_hitter_threshold)

    # --- Optional exact-match key from extra columns (postal code, phone, etc.) ---
    extra_key_cols = extra_key_cols or []
    has_extra_key = False
    for col in extra_key_cols:
        if col in df_s1.columns and col in df_s23.columns:
            df_s1["k_extra"] = df_s1.get("k_extra", "") + "|" + df_s1[col].astype(str)
            df_s23["k_extra"] = df_s23.get("k_extra", "") + "|" + df_s23[col].astype(str)
            has_extra_key = True
    if has_extra_key:
        df_s1.loc[df_s1["k_extra"] == "|" * len(extra_key_cols), "k_extra"] = pd.NA
        df_s23.loc[df_s23["k_extra"] == "|" * len(extra_key_cols), "k_extra"] = pd.NA
        df_s23 = _cap_heavy_hitters(df_s23, "k_extra", heavy_hitter_threshold)

    # --- Progressive hash joins on prefix keys ---
    cols_to_keep = ["entity_id_s1", "entity_id_s23"]
    join_keys = ["k1", "k2", "k3"] + (["k_extra"] if has_extra_key else [])

    match_frames = []
    for k in join_keys:
        m = df_s1[["entity_id", k]].dropna().merge(
            df_s23[["entity_id", k]].dropna(), on=k, suffixes=("_s1", "_s23")
        )[cols_to_keep]
        match_frames.append(m)

    # --- Token-based inverted-index keys (name + address) ---
    tok_s1_name = _build_token_frame(df_s1, "_name_core", "entity_id", min_token_len, token_heavy_hitter_threshold)
    tok_s23_name = _build_token_frame(df_s23, "_name_core", "entity_id", min_token_len, token_heavy_hitter_threshold)
    m_tok_name = tok_s1_name.merge(tok_s23_name, on="token", suffixes=("_s1", "_s23"))[cols_to_keep]
    match_frames.append(m_tok_name)

    tok_s1_addr = _build_token_frame(df_s1, "business_address_clean", "entity_id", min_token_len, token_heavy_hitter_threshold)
    tok_s23_addr = _build_token_frame(df_s23, "business_address_clean", "entity_id", min_token_len, token_heavy_hitter_threshold)
    m_tok_addr = tok_s1_addr.merge(tok_s23_addr, on="token", suffixes=("_s1", "_s23"))[cols_to_keep]
    match_frames.append(m_tok_addr)

    # --- Union and dedupe ---
    all_matches = pd.concat(match_frames, ignore_index=True).drop_duplicates()

    del match_frames, df_s1, df_s23, name_no_space_s1, name_no_space_s23

    # --- Per-record backstop cap: even a key that survives frequency filtering
    #     shouldn't be allowed to give one S1 record thousands of candidates ---
    if max_candidates_per_s1_record is not None:
        sizes = all_matches.groupby("entity_id_s1").size()
        oversized = sizes[sizes > max_candidates_per_s1_record].index
        if len(oversized):
            capped = all_matches[all_matches["entity_id_s1"].isin(oversized)].groupby(
                "entity_id_s1", group_keys=False
            ).head(max_candidates_per_s1_record)
            all_matches = pd.concat([
                all_matches[~all_matches["entity_id_s1"].isin(oversized)],
                capped,
            ], ignore_index=True)

    grouped = all_matches.groupby("entity_id_s1")["entity_id_s23"].agg(",".join).reset_index()
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]

    final_candidates = pd.DataFrame({"source1_entity_id": df_clean.loc[s1_mask, "entity_id"]})
    final_candidates = final_candidates.merge(grouped, on="source1_entity_id", how="left")
    final_candidates["candidate_entity_ids"] = final_candidates["candidate_entity_ids"].fillna("")

    return final_candidates


def evaluate_blocking(candidates: pd.DataFrame, ground_truth_pairs: set) -> dict:
    """
    candidates: output of generate_candidate_pairs (source1_entity_id, candidate_entity_ids)
    ground_truth_pairs: set of (source1_entity_id, matching_entity_id) tuples known to be true matches

    Returns Pairs Completeness (recall of true matches that made it into some block)
    and the total candidate pair count, so you can trade off against Reduction Ratio
    for your own full-population size.
    """
    found = set()
    total_pairs = 0
    lookup = dict(zip(candidates["source1_entity_id"], candidates["candidate_entity_ids"]))
    for s1_id, cand_str in lookup.items():
        if not cand_str:
            continue
        cand_ids = cand_str.split(",")
        total_pairs += len(cand_ids)
        for cid in cand_ids:
            if (s1_id, cid) in ground_truth_pairs:
                found.add((s1_id, cid))

    pairs_completeness = len(found) / len(ground_truth_pairs) if ground_truth_pairs else float("nan")
    return {
        "pairs_completeness": pairs_completeness,
        "true_matches_found": len(found),
        "true_matches_total": len(ground_truth_pairs),
        "total_candidate_pairs": total_pairs,
    }