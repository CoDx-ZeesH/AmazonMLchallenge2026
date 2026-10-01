import pandas as pd
import numpy as np
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# ======================================================================
# feature_extraction.py (V2 - Precision-Oriented, Missing-Value-Aware)
# ======================================================================
#
# What changed vs your version and why (aimed at raising F0.5, i.e.
# precision weighted 2x recall):
#
# 1. MISSING-VALUE SEMANTICS
#    Your original code returns 0.0 for any feature when a name or
#    address string is empty. That is indistinguishable from "these
#    two strings are completely dissimilar" to whatever model you feed
#    this into. A pair with a genuinely blank address next to a
#    near-perfect name match now looks identical, feature-wise, to a
#    pair with two totally different addresses. Tree-based models
#    (XGBoost/LightGBM/CatBoost, which is what you'd normally use here)
#    handle NaN natively and can learn "ignore this feature when
#    missing" -- but only if you actually give them NaN instead of a
#    manufactured 0. This alone tends to reduce false positives coming
#    from blank-field pairs, which directly helps precision.
#    -> name_missing / addr_missing indicator columns, and the
#       underlying similarity features are NaN (not 0) when missing.
#
# 2. IDF-WEIGHTED RARE-TOKEN OVERLAP (name_rare_token_jaccard)
#    token_set_ratio treats "Restaurant" and "Sunrise" as equally
#    important. Two businesses that both contain the word
#    "restaurant" but nothing else in common will still score
#    moderately on token_set_ratio -- exactly the kind of
#    generic-word collision that produces false positives and hurts
#    precision. Weighting token overlap by inverse document frequency
#    (rare shared words count far more than common ones) gives the
#    classifier a feature that specifically targets that failure mode.
#
# 3. CHARACTER N-GRAM JACCARD (name_char_bigram_jaccard)
#    A different error signature than edit-distance/token metrics --
#    catches near-duplicates with word-boundary differences
#    ("SunriseBakery" vs "Sunrise Bakery") that ratio/token_sort can
#    both misjudge depending on tokenization quirks. Cheap extra
#    signal, low correlation with the rapidfuzz features you already
#    have.
#
# 4. INITIALS / ABBREVIATION FEATURE (name_initials_ratio)
#    Catches "JP Morgan Chase" vs "J.P. Morgan Chase Bank" style pairs
#    where token order and length differ a lot but each token's first
#    letter lines up -- a common real-world business-name pattern
#    none of your existing features target directly.
#
# 5. SCALE-INVARIANT LENGTH FEATURES
#    Absolute len_diff conflates "12 vs 13 chars" (probably still a
#    match) with "2 vs 50 chars" at the same raw diff value further up
#    the range. Added name_len_ratio / addr_len_ratio
#    (shorter/longer, in [0,1]) alongside the raw diffs -- scale
#    invariance tends to matter more once your true positives span a
#    wide range of name lengths.
#
# 6. NUMERIC ADDRESS PROXIMITY (addr_num_diff)
#    Your addr_num_match was strict equality only ("12" == "12"). A
#    small numeric difference is informative and different from
#    "completely different building number" -- e.g. "12" vs "120"
#    should look almost as bad as "12" vs "999", but "12" vs "14" is a
#    much softer disagreement (could be a typo/OCR issue in the source
#    data). Added a normalized numeric distance alongside the exact
#    match flag.
#
# 7. OPTIONAL COUNTRY / EXTRA-FIELD EXACT MATCH
#    If df_clean has country_clean (or you pass other exact-match
#    columns via extra_exact_cols), an exact-match flag is one of the
#    cheapest, highest-precision features you can add -- it costs
#    almost nothing and gives the model a hard veto signal.
#
# 8. SINGLE COMBINED PARALLEL PASS
#    Your version dispatches 8 separate Parallel(...) calls, each of
#    which re-pickles and re-sends the same n1/n2/a1/a2 chunks to
#    worker processes. That's 8x the inter-process serialization
#    overhead for the same data. This version computes every
#    string-similarity feature for a chunk in one worker call, so the
#    chunks are only shipped to workers once.
#
# Everything downstream in your pipeline that reads name_ratio,
# name_token_sort, name_token_set, addr_ratio, addr_token_set,
# name_len_diff, addr_len_diff, addr_num_match, name_partial,
# name_jaro, addr_partial still gets those exact columns -- this is
# additive, nothing was renamed or removed.

STOPWORDS = {
    "inc", "llc", "corp", "corporation", "co", "company", "ltd",
    "limited", "group", "the", "and", "of", "restaurant", "store",
    "shop", "llp", "plc", "gmbh", "sa", "srl",
}


def _tokenize(s: str):
    return s.split() if s else []


def _build_idf(name_dict: dict, stopwords: set) -> dict:
    """
    Document frequency -> IDF weight per token across every business name
    in df_clean. Rare, distinctive words (e.g. 'sunrise') get a high
    weight; common/generic words (e.g. 'restaurant') get pushed toward
    the stopword floor. Computed once, passed read-only into workers.
    """
    doc_freq = Counter()
    n_docs = 0
    for name in name_dict.values():
        n_docs += 1
        toks = {t for t in _tokenize(name.lower()) if t not in stopwords}
        for t in toks:
            doc_freq[t] += 1
    if n_docs == 0:
        return {}
    return {t: float(np.log((n_docs + 1) / (c + 1)) + 1.0) for t, c in doc_freq.items()}


def chunk_list(lst, n_chunks):
    """Splits a list into equal-sized chunks for multi-core distribution."""
    size = max(1, len(lst) // n_chunks)
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _chunk_all_features(n1_chunk, n2_chunk, a1_chunk, a2_chunk, idf):
    """
    Computes every string-similarity feature for one chunk in a single
    pass (one loop over the chunk, one round-trip to the worker process)
    instead of a separate Parallel dispatch per feature.
    """
    size = len(n1_chunk)
    f = {
        "name_ratio": np.full(size, np.nan, dtype=np.float32),
        "name_token_sort": np.full(size, np.nan, dtype=np.float32),
        "name_token_set": np.full(size, np.nan, dtype=np.float32),
        "name_partial": np.full(size, np.nan, dtype=np.float32),
        "name_jaro": np.full(size, np.nan, dtype=np.float32),
        "name_char_bigram_jaccard": np.full(size, np.nan, dtype=np.float32),
        "name_rare_token_jaccard": np.full(size, np.nan, dtype=np.float32),
        "name_initials_ratio": np.full(size, np.nan, dtype=np.float32),
        "addr_ratio": np.full(size, np.nan, dtype=np.float32),
        "addr_token_sort": np.full(size, np.nan, dtype=np.float32),
        "addr_token_set": np.full(size, np.nan, dtype=np.float32),
        "addr_partial": np.full(size, np.nan, dtype=np.float32),
        "addr_jaro": np.full(size, np.nan, dtype=np.float32),
        "name_missing": np.zeros(size, dtype=np.int8),
        "addr_missing": np.zeros(size, dtype=np.int8),
    }

    for i in range(size):
        a, b = n1_chunk[i], n2_chunk[i]
        ca, cb = a1_chunk[i], a2_chunk[i]

        if not a or not b:
            f["name_missing"][i] = 1
        else:
            f["name_ratio"][i] = fuzz.ratio(a, b)
            f["name_token_sort"][i] = fuzz.token_sort_ratio(a, b)
            f["name_token_set"][i] = fuzz.token_set_ratio(a, b)
            f["name_partial"][i] = fuzz.partial_ratio(a, b)
            f["name_jaro"][i] = JaroWinkler.normalized_similarity(a, b) * 100.0

            bg_a = {a[j:j + 2] for j in range(len(a) - 1)} or {a}
            bg_b = {b[j:j + 2] for j in range(len(b) - 1)} or {b}
            union_bg = bg_a | bg_b
            f["name_char_bigram_jaccard"][i] = (
                100.0 * len(bg_a & bg_b) / len(union_bg) if union_bg else 0.0
            )

            toks_a = set(_tokenize(a))
            toks_b = set(_tokenize(b))
            union_t = toks_a | toks_b
            if union_t:
                inter_w = sum(idf.get(t, 1.0) for t in (toks_a & toks_b))
                union_w = sum(idf.get(t, 1.0) for t in union_t)
                f["name_rare_token_jaccard"][i] = 100.0 * inter_w / union_w if union_w else 0.0

            init_a = "".join(t[0] for t in _tokenize(a) if t)
            init_b = "".join(t[0] for t in _tokenize(b) if t)
            f["name_initials_ratio"][i] = fuzz.ratio(init_a, init_b) if init_a and init_b else 0.0

        if not ca or not cb:
            f["addr_missing"][i] = 1
        else:
            f["addr_ratio"][i] = fuzz.ratio(ca, cb)
            f["addr_token_sort"][i] = fuzz.token_sort_ratio(ca, cb)
            f["addr_token_set"][i] = fuzz.token_set_ratio(ca, cb)
            f["addr_partial"][i] = fuzz.partial_ratio(ca, cb)
            f["addr_jaro"][i] = JaroWinkler.normalized_similarity(ca, cb) * 100.0

    return f


def build_feature_matrix(
    df_candidates: pd.DataFrame,
    df_clean: pd.DataFrame,
    extra_exact_cols: list | None = None,
) -> pd.DataFrame:
    """
    Transforms candidates into features via O(1) dictionary lookups and a
    single combined multi-core pass over string-similarity features.

    extra_exact_cols: optional list of df_clean column names (e.g.
    ['country_clean', 'postal_code_clean']) to add as cheap, high-precision
    exact-match indicator features.
    """
    # --- 1. EXPLODE CANDIDATES ---
    df_pairs = df_candidates[["source1_entity_id", "candidate_entity_ids"]].copy()
    df_pairs["candidate_entity_ids"] = df_pairs["candidate_entity_ids"].str.split(",")
    df_pairs = df_pairs.explode("candidate_entity_ids")
    df_pairs = df_pairs[df_pairs["candidate_entity_ids"] != ""]

    id_1_list = df_pairs["source1_entity_id"].tolist()
    id_2_list = df_pairs["candidate_entity_ids"].tolist()
    del df_pairs

    total_pairs = len(id_1_list)
    if total_pairs == 0:
        return pd.DataFrame(columns=["id_1", "id_2"])

    # --- 2. O(1) DICTIONARY LOOKUPS ---
    print("    Building instant lookup dictionaries...")
    df_clean_indexed = df_clean.set_index("entity_id")
    name_dict = df_clean_indexed["business_name_clean"].fillna("").str.lower().to_dict()
    addr_dict = df_clean_indexed["business_address_clean"].fillna("").to_dict()
    first_nums = df_clean_indexed["business_address_clean"].str.extract(r"(\d+)")[0].fillna("").to_dict()

    extra_exact_cols = extra_exact_cols or []
    extra_dicts = {}
    for col in extra_exact_cols:
        if col in df_clean_indexed.columns:
            extra_dicts[col] = df_clean_indexed[col].fillna("").to_dict()
    del df_clean_indexed

    idf = _build_idf(name_dict, STOPWORDS)

    n1 = [name_dict.get(x, "") for x in id_1_list]
    n2 = [name_dict.get(x, "") for x in id_2_list]
    a1 = [addr_dict.get(x, "") for x in id_1_list]
    a2 = [addr_dict.get(x, "") for x in id_2_list]
    num1 = [first_nums.get(x, "") for x in id_1_list]
    num2 = [first_nums.get(x, "") for x in id_2_list]

    CORES = multiprocessing.cpu_count()
    CHUNKS = max(1, CORES * 2)
    print(f"    Extracting features for {total_pairs:,} pairs utilizing {CORES} CPU cores...")

    n1_chunks = chunk_list(n1, CHUNKS)
    n2_chunks = chunk_list(n2, CHUNKS)
    a1_chunks = chunk_list(a1, CHUNKS)
    a2_chunks = chunk_list(a2, CHUNKS)

    # --- 3. SINGLE COMBINED PARALLEL PASS (all string features at once) ---
    chunk_results = Parallel(n_jobs=-1)(
        delayed(_chunk_all_features)(c_n1, c_n2, c_a1, c_a2, idf)
        for c_n1, c_n2, c_a1, c_a2 in tqdm(
            zip(n1_chunks, n2_chunks, a1_chunks, a2_chunks),
            total=len(n1_chunks), desc="Feature chunks"
        )
    )
    feature_keys = chunk_results[0].keys()
    features = {k: np.concatenate([r[k] for r in chunk_results]) for k in feature_keys}
    del chunk_results

    # --- 4. CHEAP VECTORIZABLE MATH FEATURES (no need for multiprocessing) ---
    name_len_diff = np.array([abs(len(x) - len(y)) for x, y in zip(n1, n2)], dtype=np.int32)
    addr_len_diff = np.array([abs(len(x) - len(y)) for x, y in zip(a1, a2)], dtype=np.int32)
    name_len_ratio = np.array(
        [min(len(x), len(y)) / max(len(x), len(y)) if max(len(x), len(y)) else np.nan
         for x, y in zip(n1, n2)], dtype=np.float32
    )
    addr_len_ratio = np.array(
        [min(len(x), len(y)) / max(len(x), len(y)) if max(len(x), len(y)) else np.nan
         for x, y in zip(a1, a2)], dtype=np.float32
    )

    addr_num_match = np.array(
        [1 if x and y and x == y else 0 for x, y in zip(num1, num2)], dtype=np.int8
    )

    def _num_diff(x, y):
        if not x or not y:
            return np.nan
        try:
            return abs(int(x) - int(y)) / max(int(x), int(y), 1)
        except ValueError:
            return np.nan

    addr_num_diff = np.array([_num_diff(x, y) for x, y in zip(num1, num2)], dtype=np.float32)

    # --- 5. ASSEMBLE FINAL DATAFRAME ---
    print("    Constructing final feature matrix...")
    df_features = pd.DataFrame({
        "id_1": id_1_list,
        "id_2": id_2_list,
        "name_ratio": features["name_ratio"],
        "name_token_sort": features["name_token_sort"],
        "name_token_set": features["name_token_set"],
        "name_partial": features["name_partial"],
        "name_jaro": features["name_jaro"],
        "name_char_bigram_jaccard": features["name_char_bigram_jaccard"],
        "name_rare_token_jaccard": features["name_rare_token_jaccard"],
        "name_initials_ratio": features["name_initials_ratio"],
        "addr_ratio": features["addr_ratio"],
        "addr_token_sort": features["addr_token_sort"],
        "addr_token_set": features["addr_token_set"],
        "addr_partial": features["addr_partial"],
        "addr_jaro": features["addr_jaro"],
        "name_missing": features["name_missing"],
        "addr_missing": features["addr_missing"],
        "name_len_diff": name_len_diff,
        "addr_len_diff": addr_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_len_ratio": addr_len_ratio,
        "addr_num_match": addr_num_match,
        "addr_num_diff": addr_num_diff,
    })

    for col, d in extra_dicts.items():
        v1 = [d.get(x, "") for x in id_1_list]
        v2 = [d.get(x, "") for x in id_2_list]
        df_features[f"{col}_match"] = np.array(
            [1 if x and y and x == y else 0 for x, y in zip(v1, v2)], dtype=np.int8
        )

    return df_features