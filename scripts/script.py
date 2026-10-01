#!/usr/bin/env python3
"""Step 1 audit for the Amazon ML Challenge: Business Entity Resolution.

Read-only audit. Processes TSVs in chunks and stores frequency/distinct-value
state in a temporary SQLite database to avoid retaining large value sets in RAM.

Outputs:
  reports/step1_audit.txt
  reports/step1_field_stats.csv
  reports/step1_match_stats.csv
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

import pandas as pd

FIELDS = ["business_name", "business_address", "country"]
SOURCES = ["source1", "source2", "source3"]
SPLITS = ["train", "test"]
TOP_N = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Amazon ER challenge data without modifying the pipeline.")
    p.add_argument("--data-dir", default="dataset", help="Dataset root containing train/ and test/")
    p.add_argument("--chunk-size", type=int, default=100_000, help="TSV rows per chunk")
    p.add_argument("--output-dir", default="reports", help="Directory for generated reports")
    return p.parse_args()


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"_+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    PRAGMA journal_mode=OFF;
    PRAGMA synchronous=OFF;
    PRAGMA temp_store=FILE;
    CREATE TABLE IF NOT EXISTS value_freq (
        split TEXT NOT NULL,
        src TEXT NOT NULL,
        field TEXT NOT NULL,
        kind TEXT NOT NULL,
        val TEXT NOT NULL,
        cnt INTEGER NOT NULL,
        PRIMARY KEY(split, src, field, kind, val)
    );
    CREATE INDEX IF NOT EXISTS idx_vf_lookup
      ON value_freq(field, kind, val, split, src);

    CREATE TABLE IF NOT EXISTS file_stats (
        split TEXT NOT NULL,
        src TEXT NOT NULL,
        rows INTEGER NOT NULL,
        duplicate_rows INTEGER NOT NULL,
        duplicate_entity_ids INTEGER NOT NULL,
        PRIMARY KEY(split, src)
    );
    """)


def upsert_counts(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    conn.executemany(
        """INSERT INTO value_freq(split,src,field,kind,val,cnt)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(split,src,field,kind,val)
           DO UPDATE SET cnt=cnt+excluded.cnt""",
        rows,
    )


def load_one_file(path: Path, chunk_size: int, split: str, src: str, conn: sqlite3.Connection) -> dict:
    required = {"entity_id", *FIELDS}
    row_count = 0
    nulls = Counter()
    duplicate_rows = 0
    # Exact row hashes are kept on disk, not in RAM.
    row_hash_db = f"rowhash_{split}_{src}"
    conn.execute(f"CREATE TABLE IF NOT EXISTS {row_hash_db} (h INTEGER PRIMARY KEY)")
    entity_seen_db = f"entity_{split}_{src}"
    conn.execute(f"CREATE TABLE IF NOT EXISTS {entity_seen_db} (id TEXT PRIMARY KEY)")

    schemas = {}
    lengths = {f: {"n": 0, "sum": 0, "min": None, "max": 0} for f in FIELDS}

    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], chunksize=chunk_size)
    for chunk in reader:
        if row_count == 0:
            schemas = {c: str(t) for c, t in chunk.dtypes.items()}
            missing = required.difference(chunk.columns)
            if missing:
                raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
        row_count += len(chunk)

        # Disk-backed duplicate checks.
        core = chunk[["entity_id", *FIELDS]].fillna("")
        hashes = pd.util.hash_pandas_object(core, index=False).astype("int64").tolist()
        for h in hashes:
            cur = conn.execute(f"INSERT OR IGNORE INTO {row_hash_db}(h) VALUES (?)", (int(h),))
            if cur.rowcount == 0:
                duplicate_rows += 1

        ids = [(str(x),) for x in chunk["entity_id"].fillna("").tolist() if str(x) != ""]
        conn.executemany(f"INSERT OR IGNORE INTO {entity_seen_db}(id) VALUES (?)", ids)

        db_rows = []
        for field in FIELDS:
            vals = chunk[field].fillna("").astype(str)
            missing_mask = vals.str.strip().eq("")
            nulls[field] += int(missing_mask.sum())
            nonempty = vals[~missing_mask]
            if nonempty.empty:
                continue

            lens = nonempty.str.len()
            st = lengths[field]
            st["n"] += len(lens)
            st["sum"] += int(lens.sum())
            lo, hi = int(lens.min()), int(lens.max())
            st["min"] = lo if st["min"] is None else min(st["min"], lo)
            st["max"] = max(st["max"], hi)

            raw_counts = nonempty.value_counts()
            db_rows.extend((split, src, field, "raw", str(k), int(v)) for k, v in raw_counts.items())
            normalized = nonempty.map(norm_text)
            normalized = normalized[normalized.ne("")]
            norm_counts = normalized.value_counts()
            db_rows.extend((split, src, field, "norm", str(k), int(v)) for k, v in norm_counts.items())

        upsert_counts(conn, db_rows)
        conn.commit()

    dup_entity_ids = int(conn.execute(
        f"SELECT COUNT(*) FROM (SELECT id FROM {entity_seen_db}) x "
        # entity_seen stores DISTINCT IDs, so compare with total file rows below.
    ).fetchone()[0])
    # More direct duplicate-ID count by counting source rows vs distinct entity IDs.
    distinct_ids = conn.execute(f"SELECT COUNT(*) FROM {entity_seen_db}").fetchone()[0]
    dup_entity_ids = max(0, row_count - int(distinct_ids))

    conn.execute(
        """INSERT OR REPLACE INTO file_stats(split,src,rows,duplicate_rows,duplicate_entity_ids)
           VALUES (?,?,?,?,?)""",
        (split, src, row_count, duplicate_rows, dup_entity_ids),
    )
    conn.commit()
    return {
        "path": str(path),
        "rows": row_count,
        "schemas": schemas,
        "nulls": dict(nulls),
        "lengths": lengths,
        "duplicate_rows": duplicate_rows,
        "duplicate_entity_ids": dup_entity_ids,
    }


def fetch_freq(conn: sqlite3.Connection, split: str, src: str, field: str, kind: str) -> list[tuple[str, int]]:
    return [
        (r[0], int(r[1]))
        for r in conn.execute(
            "SELECT val,cnt FROM value_freq WHERE split=? AND src=? AND field=? AND kind=? ORDER BY cnt DESC LIMIT ?",
            (split, src, field, kind, TOP_N),
        )
    ]


def counts_summary(conn: sqlite3.Connection, split: str, src: str, field: str, kind: str) -> dict:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(cnt),0), COALESCE(MAX(cnt),0) FROM value_freq WHERE split=? AND src=? AND field=? AND kind=?",
        (split, src, field, kind),
    ).fetchone()
    return {"unique": int(row[0]), "nonempty": int(row[1]), "max_count": int(row[2])}


def overlap_count(conn: sqlite3.Connection, split_a: str, src_a: str, split_b: str, src_b: str, field: str, kind: str) -> int:
    q = """
    SELECT COUNT(*) FROM value_freq a
    JOIN value_freq b ON a.field=b.field AND a.kind=b.kind AND a.val=b.val
    WHERE a.split=? AND a.src=? AND b.split=? AND b.src=? AND a.field=? AND a.kind=?
    """
    return int(conn.execute(q, (split_a, src_a, split_b, src_b, field, kind)).fetchone()[0])


def s1_row_presence_rate(conn: sqlite3.Connection, split: str, other_split: str, other_src: str, field: str, kind: str) -> float:
    """Percent of S1 non-empty rows whose value occurs in another source."""
    # Count S1 value frequencies whose value exists in target source, then divide by S1 nonempty rows.
    row = conn.execute(
        """
        SELECT COALESCE(SUM(a.cnt),0)
        FROM value_freq a
        JOIN value_freq b ON a.field=b.field AND a.kind=b.kind AND a.val=b.val
        WHERE a.split=? AND a.src='source1' AND a.field=? AND a.kind=?
          AND b.split=? AND b.src=? AND b.kind=? AND b.field=?
        """,
        (split, field, kind, other_split, other_src, kind, field),
    ).fetchone()
    denom = counts_summary(conn, split, "source1", field, kind)["nonempty"]
    return 100.0 * float(row[0]) / denom if denom else 0.0


def ground_truth_stats(path: Path, chunk_size: int) -> dict:
    counts: list[int] = []
    s2_links = 0
    s3_links = 0
    empty = 0
    total = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
        total += len(chunk)
        for raw in chunk["matched_entity_ids"]:
            text = "" if raw is None else str(raw).strip()
            ids = [] if not text else [x.strip() for x in text.split(",") if x.strip()]
            n = len(ids)
            counts.append(n)
            s2_links += sum(x.startswith("S2-") for x in ids)
            s3_links += sum(x.startswith("S3-") for x in ids)
            empty += n == 0
    dist = Counter(counts)
    return {
        "s1_entities": total,
        "zero_matches": empty,
        "one_match": dist.get(1, 0),
        "two_plus": sum(v for k, v in dist.items() if k >= 2),
        "s2_links": s2_links,
        "s3_links": s3_links,
        "avg_matches": sum(counts) / total if total else 0.0,
        "median_matches": median(counts) if counts else 0,
        "max_matches": max(counts) if counts else 0,
        "distribution": dict(sorted(dist.items())),
    }


def pct(n: float, d: float) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def field_record(conn: sqlite3.Connection, split: str, src: str, field: str, file_stats: dict) -> dict:
    raw = counts_summary(conn, split, src, field, "raw")
    norm = counts_summary(conn, split, src, field, "norm")
    top_raw = fetch_freq(conn, split, src, field, "raw")
    top_norm = fetch_freq(conn, split, src, field, "norm")

    # Cross-source rates are defined from Source 1 rows for comparability.
    raw_s12 = raw_s13 = norm_s12 = norm_s13 = ""
    if src == "source1":
        raw_s12 = round(s1_row_presence_rate(conn, split, split, "source2", field, "raw"), 4)
        raw_s13 = round(s1_row_presence_rate(conn, split, split, "source3", field, "raw"), 4)
        norm_s12 = round(s1_row_presence_rate(conn, split, split, "source2", field, "norm"), 4)
        norm_s13 = round(s1_row_presence_rate(conn, split, split, "source3", field, "norm"), 4)

    return {
        "split": split,
        "source": src,
        "field": field,
        "rows": file_stats["rows"],
        "missing_pct": round(pct(file_stats["nulls"].get(field, 0), file_stats["rows"]), 4),
        "unique_count": raw["unique"],
        "unique_ratio_pct": round(pct(raw["unique"], max(1, file_stats["rows"] - file_stats["nulls"].get(field, 0))), 4),
        "raw_exact_rate_s1_s2_pct": raw_s12,
        "raw_exact_rate_s1_s3_pct": raw_s13,
        "norm_exact_rate_s1_s2_pct": norm_s12,
        "norm_exact_rate_s1_s3_pct": norm_s13,
        "raw_top1_frequency_pct": round(pct(top_raw[0][1] if top_raw else 0, raw["nonempty"]), 4),
        "norm_top1_frequency_pct": round(pct(top_norm[0][1] if top_norm else 0, norm["nonempty"]), 4),
        "nonempty_min_len": file_stats["lengths"][field]["min"] or 0,
        "nonempty_mean_len": round(file_stats["lengths"][field]["sum"] / max(1, file_stats["lengths"][field]["n"]), 3),
        "nonempty_max_len": file_stats["lengths"][field]["max"],
        "top_raw_value": top_raw[0][0] if top_raw else "",
        "top_normalized_value": top_norm[0][0] if top_norm else "",
    }


def train_test_shift(conn: sqlite3.Connection, field: str) -> dict:
    tr = counts_summary(conn, "train", "source1", field, "norm")
    te = counts_summary(conn, "test", "source1", field, "norm")
    overlap = overlap_count(conn, "train", "source1", "test", "source1", field, "norm")
    return {
        "field": field,
        "train_unique": tr["unique"],
        "test_unique": te["unique"],
        "unique_overlap": overlap,
        "train_unique_seen_in_test_pct": round(pct(overlap, tr["unique"]), 4),
        "test_unique_seen_in_train_pct": round(pct(overlap, te["unique"]), 4),
        "train_top1_frequency_pct": round(pct(fetch_freq(conn,"train","source1",field,"norm")[0][1] if fetch_freq(conn,"train","source1",field,"norm") else 0, tr["nonempty"]),4),
        "test_top1_frequency_pct": round(pct(fetch_freq(conn,"test","source1",field,"norm")[0][1] if fetch_freq(conn,"test","source1",field,"norm") else 0, te["nonempty"]),4),
    }


def top5(stats_by_file: dict, gt: dict, conn: sqlite3.Connection) -> list[str]:
    findings = []
    findings.append(
        f"{pct(gt['zero_matches'], gt['s1_entities']):.2f}% of training S1 entities are singletons (0 true matches), so false-positive control is important."
    )
    overlaps = []
    for field in FIELDS:
        for other in ["source2", "source3"]:
            n = overlap_count(conn, "train", "source1", "train", other, field, "norm")
            overlaps.append((n, field, other))
    n, field, other = max(overlaps)
    findings.append(f"Largest normalized unique-value overlap for train S1 is `{field}` with {other}: {n:,} shared values; this is a candidate blocking anchor, subject to row-level recall checks.")

    heavy = []
    for field in FIELDS:
        freq = fetch_freq(conn, "train", "source1", field, "norm")
        if freq:
            _, ncnt = freq[0]
            nonempty = counts_summary(conn, "train", "source1", field, "norm")["nonempty"]
            heavy.append((ncnt / max(1, nonempty), field, freq[0][0]))
    share, hfield, _ = max(heavy)
    findings.append(f"Heaviest normalized S1 field is `{hfield}` at {share*100:.2f}% of non-empty rows for its top value; blocking should frequency-cap common keys.")
    findings.append(f"Ground-truth multiplicity is average={gt['avg_matches']:.3f}, median={gt['median_matches']}, max={gt['max_matches']:,}; the pipeline must support legitimate one-to-many links.")

    train_c = set()
    test_c = set()
    # Country cardinality is small in practice; keep this tiny in RAM.
    for source in SOURCES:
        for val, _ in fetch_freq(conn, "train", source, "country", "raw"):
            train_c.add(val)
        for val, _ in fetch_freq(conn, "test", source, "country", "raw"):
            test_c.add(val)
    new_c = sorted(test_c - train_c)
    if new_c:
        findings.append(f"Test contains country labels absent from the training sample: {', '.join(new_c)}; treat country as open-set text, not a fixed category list.")
    else:
        findings.append("No new country label appears in the top observed values; keep country handling open-set anyway.")
    return findings


def write_reports(out_dir: Path, stats_by_file: dict, gt: dict, conn: sqlite3.Connection) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    field_rows = []
    for key, fs in stats_by_file.items():
        split, src = key.split("/")
        for field in FIELDS:
            field_rows.append(field_record(conn, split, src, field, fs))
    pd.DataFrame(field_rows).to_csv(out_dir / "step1_field_stats.csv", index=False)

    match_rows = [
        {"metric": "s1_entities", "value": gt["s1_entities"]},
        {"metric": "zero_matches", "value": gt["zero_matches"]},
        {"metric": "zero_match_pct", "value": round(pct(gt["zero_matches"], gt["s1_entities"]), 6)},
        {"metric": "one_match", "value": gt["one_match"]},
        {"metric": "one_match_pct", "value": round(pct(gt["one_match"], gt["s1_entities"]), 6)},
        {"metric": "two_plus", "value": gt["two_plus"]},
        {"metric": "two_plus_pct", "value": round(pct(gt["two_plus"], gt["s1_entities"]), 6)},
        {"metric": "total_s2_links", "value": gt["s2_links"]},
        {"metric": "total_s3_links", "value": gt["s3_links"]},
        {"metric": "average_matches_per_s1", "value": round(gt["avg_matches"], 6)},
        {"metric": "median_matches_per_s1", "value": gt["median_matches"]},
        {"metric": "max_matches_per_s1", "value": gt["max_matches"]},
    ]
    match_rows += [{"metric": f"s1_match_count_{k}", "value": v} for k, v in gt["distribution"].items()]
    pd.DataFrame(match_rows).to_csv(out_dir / "step1_match_stats.csv", index=False)

    lines = [
        "AMAZON ML CHALLENGE — STEP 1 DATA AUDIT",
        "=" * 70,
        "Purpose: read-only audit before blocking/modeling changes.",
        "",
        "1) DATASET SUMMARY",
        "-" * 30,
    ]
    for key, fs in stats_by_file.items():
        lines.append(f"{key}: {fs['rows']:,} rows | {fs['path']}")

    lines += ["", "2) SCHEMA / DTYPE", "-" * 30]
    for key, fs in stats_by_file.items():
        lines.append(f"[{key}]")
        for c, t in fs["schemas"].items():
            lines.append(f"  {c}: {t}")

    lines += ["", "3) FIELD QUALITY", "-" * 30]
    for key, fs in stats_by_file.items():
        split, src = key.split("/")
        lines.append(f"[{key}]")
        for field in FIELDS:
            rec = field_record(conn, split, src, field, fs)
            lines.append(
                f"  {field}: missing={rec['missing_pct']:.2f}% | unique={rec['unique_count']:,} | "
                f"unique_ratio={rec['unique_ratio_pct']:.2f}% | len(min/mean/max)="
                f"{rec['nonempty_min_len']}/{rec['nonempty_mean_len']:.1f}/{rec['nonempty_max_len']}"
            )
            if src == "source1":
                lines.append(
                    f"    raw exact row presence: S2={rec['raw_exact_rate_s1_s2_pct']:.2f}% | S3={rec['raw_exact_rate_s1_s3_pct']:.2f}%"
                )
                lines.append(
                    f"    norm exact row presence: S2={rec['norm_exact_rate_s1_s2_pct']:.2f}% | S3={rec['norm_exact_rate_s1_s3_pct']:.2f}%"
                )
            lines.append("    top raw values:")
            for val, n in fetch_freq(conn, split, src, field, "raw"):
                lines.append(f"      {n:>10,}  {val!r}")
            lines.append("    top normalized values:")
            for val, n in fetch_freq(conn, split, src, field, "norm"):
                lines.append(f"      {n:>10,}  {val!r}")

    lines += ["", "4) COUNTRY DISTRIBUTIONS", "-" * 30]
    for key, fs in stats_by_file.items():
        split, src = key.split("/")
        lines.append(f"[{key}]")
        for country, n in fetch_freq(conn, split, src, "country", "raw"):
            lines.append(f"  {country!r}: {n:,} ({pct(n, fs['rows']):.2f}%)")

    lines += ["", "5) DUPLICATES", "-" * 30]
    for key, fs in stats_by_file.items():
        lines.append(f"{key}: duplicate required-field rows={fs['duplicate_rows']:,}; duplicated entity_id rows={fs['duplicate_entity_ids']:,}")

    lines += ["", "6) GROUND TRUTH", "-" * 30,
              f"S1 entities: {gt['s1_entities']:,}",
              f"Zero matches: {gt['zero_matches']:,} ({pct(gt['zero_matches'],gt['s1_entities']):.2f}%)",
              f"Exactly one match: {gt['one_match']:,} ({pct(gt['one_match'],gt['s1_entities']):.2f}%)",
              f"Two+ matches: {gt['two_plus']:,} ({pct(gt['two_plus'],gt['s1_entities']):.2f}%)",
              f"Total S2 links: {gt['s2_links']:,}",
              f"Total S3 links: {gt['s3_links']:,}",
              f"Average matches/S1: {gt['avg_matches']:.4f}",
              f"Median matches/S1: {gt['median_matches']}",
              f"Maximum matches/S1: {gt['max_matches']:,}",
              "Match-count distribution:"]
    lines += [f"  {k}: {v:,}" for k, v in gt["distribution"].items()]

    lines += ["", "7) NORMALIZED EXACT UNIQUE-VALUE OVERLAP", "-" * 30]
    for field in FIELDS:
        lines.append(f"{field}:")
        for a, b in [("source1","source2"),("source1","source3"),("source2","source3")]:
            n = overlap_count(conn, "train", a, "train", b, field, "norm")
            lines.append(f"  train {a} ↔ {b}: {n:,} shared normalized values")

    lines += ["", "8) TRAIN vs TEST S1 NORMALIZED DISTRIBUTION", "-" * 30]
    for field in FIELDS:
        shift = train_test_shift(conn, field)
        lines.append(
            f"{field}: train_unique={shift['train_unique']:,}, test_unique={shift['test_unique']:,}, "
            f"shared={shift['unique_overlap']:,}, train_seen_in_test={shift['train_unique_seen_in_test_pct']:.2f}%, "
            f"test_seen_in_train={shift['test_unique_seen_in_train_pct']:.2f}%"
        )

    lines += ["", "9) SUSPICIOUS / RARE PATTERNS", "-" * 30]
    suspicious = []
    for key, fs in stats_by_file.items():
        for field in FIELDS:
            rec = field_record(conn, *key.split("/"), field, fs)
            if rec["missing_pct"] >= 30:
                suspicious.append(f"{key}: {field} has {rec['missing_pct']:.1f}% missing values.")
            if rec["norm_top1_frequency_pct"] >= 10:
                suspicious.append(f"{key}: {field} has a heavy-hitter normalized value at {rec['norm_top1_frequency_pct']:.1f}% of non-empty rows; frequency-cap it during blocking.")
            if rec["nonempty_max_len"] > max(200, 8 * rec["nonempty_mean_len"]):
                suspicious.append(f"{key}: {field} has a long-tail length distribution (max={rec['nonempty_max_len']}, mean={rec['nonempty_mean_len']:.1f}).")
        if fs["duplicate_rows"]:
            suspicious.append(f"{key}: {fs['duplicate_rows']:,} duplicate required-field rows detected.")
        if fs["duplicate_entity_ids"]:
            suspicious.append(f"{key}: {fs['duplicate_entity_ids']:,} duplicate entity_id rows detected.")
    lines += [f"- {x}" for x in suspicious] if suspicious else ["No threshold-based suspicious patterns detected."]

    lines += ["", "10) TOP 5 DISCOVERIES FOR STEP 2", "-" * 30]
    lines += [f"{i}. {x}" for i, x in enumerate(top5(stats_by_file, gt, conn), 1)]
    (out_dir / "step1_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    data = Path(args.data_dir)
    out = Path(args.output_dir)
    train = data / "train"
    test = data / "test"
    expected = {
        "train/source1": train / "train_source1.tsv",
        "train/source2": train / "train_source2.tsv",
        "train/source3": train / "train_source3.tsv",
        "test/source1": test / "test_source1.tsv",
        "test/source2": test / "test_source2.tsv",
        "test/source3": test / "test_source3.tsv",
        "train/ground_truth": train / "train_ground_truth.tsv",
    }
    missing = [f"{k}: {p}" for k, p in expected.items() if not p.exists()]
    if missing:
        print("Missing expected files:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="amazon_er_audit_") as tmp:
        conn = sqlite3.connect(Path(tmp) / "audit.sqlite")
        init_db(conn)
        stats_by_file = {}
        try:
            for key in ["train/source1","train/source2","train/source3","test/source1","test/source2","test/source3"]:
                split, src = key.split("/")
                print(f"[audit] {key} -> {expected[key]}")
                stats_by_file[key] = load_one_file(expected[key], args.chunk_size, split, src, conn)
            print("[audit] parsing ground truth...")
            gt = ground_truth_stats(expected["train/ground_truth"], args.chunk_size)
            write_reports(out, stats_by_file, gt, conn)
        finally:
            conn.close()

    print("\nSTEP 1 AUDIT COMPLETE")
    print(f"  {out / 'step1_audit.txt'}")
    print(f"  {out / 'step1_field_stats.csv'}")
    print(f"  {out / 'step1_match_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
