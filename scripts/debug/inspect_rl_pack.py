from pathlib import Path
import pandas as pd

PACK = Path(
    "data/rl_observation_pack.parquet"
)

df = pd.read_parquet(PACK)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
for c in df.columns:
    print(c)

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== HEAD ===")
print(df.head(5))

print("\n=== TAIL ===")
print(df.tail(5))

# timestamp
if "timestamp" in df.columns:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    print("\n=== TIMESTAMP RANGE ===")
    print("min:", ts.min())
    print("max:", ts.max())
    print("n_unique:", ts.nunique())
    print("n_na:", ts.isna().sum())

    dups = ts.duplicated().sum()
    print("duplicated timestamps:", dups)

# segments
if "segment" in df.columns:
    print("\n=== SEGMENT COUNTS ===")
    print(df["segment"].value_counts(dropna=False))

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        tmp = df.copy()
        tmp["_ts"] = ts
        print("\n=== SEGMENT TIME RANGES ===")
        print(
            tmp.groupby("segment")["_ts"]
            .agg(["min", "max", "count"])
            .sort_values("min")
        )

# split ids
for col in ["_source_split_id", "split_id"]:
    if col in df.columns:
        print(f"\n=== {col} COUNTS ===")
        print(df[col].value_counts(dropna=False).sort_index())

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            tmp = df.copy()
            tmp["_ts"] = ts
            print(f"\n=== {col} TIME RANGES ===")
            print(
                tmp.groupby(col)["_ts"]
                .agg(["min", "max", "count"])
                .sort_values("min")
            )

# run / model provenance
for col in [
    "_source_upstream_run_id",
    "_source_upstream_model_id",
    "_source_train_end",
    "_source_run_dir",
]:
    if col in df.columns:
        print(f"\n=== {col} ===")
        vals = df[col].dropna().astype(str)
        print("n_unique:", vals.nunique())
        print(vals.value_counts().head(20))

# regime family columns
families = {
    "p_filter": [c for c in df.columns if c.startswith("p_filter_")],
    "dp_filter": [c for c in df.columns if c.startswith("dp_filter_")],
    "p_smooth": [c for c in df.columns if c.startswith("p_smooth_")],
    "gamma_filter": [c for c in df.columns if c.startswith("gamma_filter_")],
    "alpha_filter": [c for c in df.columns if c.startswith("alpha_filter_")],
    "A_t": [c for c in df.columns if c.startswith("A_t_")],
}
print("\n=== FEATURE FAMILIES ===")
for k, v in families.items():
    print(k, len(v), v[:10])

# quick integrity checks
print("\n=== INTEGRITY CHECKS ===")
for col in ["p_max_filter", "p_margin_filter", "effective_num_states_filter", "expected_state_filter"]:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        print(col, {
            "na": int(s.isna().sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        })

p_cols = [c for c in df.columns if c.startswith("p_filter_")]
if p_cols:
    p = df[p_cols].apply(pd.to_numeric, errors="coerce")
    row_sum = p.sum(axis=1)
    print("\n=== P_FILTER SUM CHECK ===")
    print({
        "mean_sum": float(row_sum.mean()),
        "min_sum": float(row_sum.min()),
        "max_sum": float(row_sum.max()),
        "rows_far_from_1": int(((row_sum - 1.0).abs() > 1e-4).sum()),
    })