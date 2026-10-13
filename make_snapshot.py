"""
Create a random customer-level snapshot of a large transaction parquet file.

Randomly selects customers and keeps ALL of their transaction rows, so every selected customer is complete.
Memory-safe for very large files: the full table is never loaded into pandas. Only three columns are scanned
to build the customer list, then only the selected customers' rows are read.

Examples
--------
    # about 100,000 rows (whole customers), stratified by M0 month and target, saved as CSV
    python make_snapshot.py --path collections_data.parquet --out snapshot.csv --rows 100000

    # a fixed number of customers instead (better for running the full notebook)
    python make_snapshot.py --path collections_data.parquet --out snapshot_20k.csv --customers 20000

The output format follows the file extension: .csv writes CSV, .parquet writes parquet.
From a notebook:   %run make_snapshot.py --path collections_data.parquet --out snapshot.csv --rows 100000

Reading the CSV back with the original dtypes (ids as text, dates parsed, compact numeric types):
    from make_snapshot import load_snapshot_csv
    df = load_snapshot_csv("snapshot.csv")      # use in place of pd.read_parquet(PARQUET_PATH)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ID, M0, TARGET = "cifwithcheckdigit", "m0_monthenddate", "target_value"


# ----------------------------------------------------------------------------------------------
def _decode(col):
    """Dictionary-encoded (categorical) column -> plain strings. Other types are returned unchanged."""
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


def _value_type(t):
    return t.value_type if pa.types.is_dictionary(t) else t


def customer_table(dataset):
    """One row per customer: row count, M0 month and target. Scans only 3 columns, aggregates in Arrow."""
    t = dataset.to_table(columns=[ID, M0, TARGET])
    g = t.group_by(ID).aggregate([(ID, "count"), (M0, "min"), (TARGET, "min")])
    g = g.set_column(g.schema.get_field_index(ID), ID, _decode(g[ID]))      # ~120k keys only
    cust = g.to_pandas().rename(columns={f"{ID}_count": "n_rows", f"{M0}_min": "m0", f"{TARGET}_min": "target"})
    cust["month"] = pd.to_datetime(cust["m0"]).dt.to_period("M").astype(str)
    cust["stratum"] = cust["month"] + "|" + cust["target"].astype(str)
    return cust[[ID, "n_rows", "m0", "month", "target", "stratum"]]


def draw_sample(cust, rows=None, customers=None, stratify=True, seed=42, tol=0.05, max_tries=200):
    """
    Random customers. `rows` targets a total row count (whole customers, so it is approximate: the draw is
    repeated with successive seeds until within `tol`). `customers` targets a number of customers.
    Stratified sampling keeps the M0-month x target mix of the population.
    """
    if (rows is None) == (customers is None):
        raise ValueError("Specify exactly one of rows= or customers=")
    frac = rows / cust["n_rows"].sum() if rows is not None else customers / len(cust)
    if frac >= 1:
        raise ValueError("Requested sample is as large as the population")
    best = None
    for k in range(max_tries):
        rs = seed + k
        if stratify:
            chosen = cust.groupby("stratum", group_keys=False).sample(frac=frac, random_state=rs)
        else:
            chosen = cust.sample(frac=frac, random_state=rs)
        dev = abs(chosen["n_rows"].sum() - rows) / rows if rows is not None else abs(len(chosen) - customers) / customers
        if best is None or dev < best[0]:
            best = (dev, k, chosen)
        if dev <= tol:
            break
    return best[2].copy(), best[1], best[0]


def extract_rows(dataset, chosen):
    ids = pa.array(chosen[ID].astype(str).tolist(), type=_value_type(dataset.schema.field(ID).type))
    tbl = dataset.to_table(filter=ds.field(ID).isin(ids))
    return tbl.set_column(tbl.schema.get_field_index(ID), ID, _decode(tbl[ID]))     # plain string ids from here on


def check_complete(tbl, chosen):
    """Every selected customer must have exactly the same number of rows as in the full file."""
    got = tbl.group_by(ID).aggregate([(ID, "count")]).to_pandas().set_index(ID)[f"{ID}_count"]
    want = chosen.set_index(ID)["n_rows"]
    bad = (got.reindex(want.index) != want)
    assert not bad.any(), f"{int(bad.sum())} customers are incomplete in the snapshot"
    assert len(got) == len(want), "snapshot contains customers that were not selected"


def pseudonymise(tbl, seed):
    """Replace customer ids with random codes (C000001...). Returns table and the id map (keep the map private)."""
    uniq = tbl[ID].unique()
    order = np.random.default_rng(seed).permutation(len(uniq))
    labels = np.array([f"C{i:06d}" for i in range(len(uniq))])[order]
    pos = pc.index_in(tbl[ID], value_set=uniq).to_numpy(zero_copy_only=False)
    new = pa.array(labels[pos])
    idx = tbl.schema.get_field_index(ID)
    mapping = pd.DataFrame({"original_id": uniq.to_pylist(), "snapshot_id": labels})
    return tbl.set_column(idx, ID, new), mapping


def compare(cust, chosen):
    """Population vs snapshot: is the sample representative?"""
    def summary(d):
        w = d["n_rows"]
        return pd.Series({"customers": len(d), "rows": int(w.sum()), "target_rate": d["target"].mean(),
                          "rows/cust mean": w.mean(), "rows/cust median": w.median(), "rows/cust p95": w.quantile(.95)})
    out = pd.DataFrame({"population": summary(cust), "snapshot": summary(chosen)})
    months = pd.DataFrame({"population": cust["month"].value_counts(normalize=True),
                           "snapshot": chosen["month"].value_counts(normalize=True)}).sort_index()
    return out, months


def write_snapshot(tbl, out_path):
    """Write the snapshot as CSV or parquet depending on the file extension."""
    if out_path.lower().endswith(".csv"):
        # CSV has no categorical type: write dictionary-encoded columns as plain text
        for i, f in enumerate(tbl.schema):
            if pa.types.is_dictionary(f.type):
                tbl = tbl.set_column(i, f.name, tbl[f.name].cast(f.type.value_type))
        pacsv.write_csv(tbl, out_path)
    else:
        pq.write_table(tbl, out_path, compression="zstd")


def load_snapshot_csv(path):
    """Read a snapshot CSV back with the dtypes of the original parquet file."""
    df = pd.read_csv(
        path,
        dtype={ID: "str", "due_count": "uint8", "payment_count": "uint8", TARGET: "int8",
               "m0_arrearsamount": "float32", "m0_total_due_with_arrears": "float32", "total_paid": "float32",
               "amount": "float32", "balance": "float32"},
        parse_dates=[M0, "payment_date", "transactiondate"],
    )
    df[M0] = df[M0].astype("datetime64[s]")
    df["payment_date"] = df["payment_date"].astype("datetime64[s]")
    df["transactiondate"] = df["transactiondate"].astype("datetime64[ns]")
    for c in ["collectionsmethodname", "trantypedesc", "spendgroup", "categoryname"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def sibling(path, suffix):
    """snapshot.csv -> snapshot_id_map.csv / snapshot_manifest.json (works for any extension)."""
    return os.path.splitext(path)[0] + suffix


def write_snapshot(tbl, path):
    """Write as CSV or parquet depending on the file extension."""
    if path.lower().endswith(".csv"):
        # CSV has no categorical type: write category columns as plain text
        for i, f in enumerate(tbl.schema):
            if pa.types.is_dictionary(f.type):
                tbl = tbl.set_column(i, f.name, tbl[f.name].cast(pa.string()))
        pacsv.write_csv(tbl, path)
    else:
        pq.write_table(tbl, path, compression="zstd")


SNAPSHOT_DTYPES = {
    ID: "string", "m0_arrearsamount": "float32", "due_count": "uint8", "payment_count": "uint8",
    "m0_total_due_with_arrears": "float32", "total_paid": "float32", TARGET: "int8",
    "collectionsmethodname": "category", "trantypedesc": "category", "amount": "float32", "balance": "float32",
    "spendgroup": "category", "categoryname": "category",
}
SNAPSHOT_DATES = [M0, "payment_date", "transactiondate"]


def read_snapshot(path):
    """Read a snapshot written by this script and restore the dtypes of the original parquet file."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    df = pd.read_csv(path, dtype={k: v for k, v in SNAPSHOT_DTYPES.items()}, parse_dates=SNAPSHOT_DATES)
    df[ID] = df[ID].astype(str)
    return df


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="collections_data.parquet", help="source parquet file")
    ap.add_argument("--out", default="snapshot.csv", help="output file; .csv or .parquet")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rows", type=int, default=None, help="target number of transaction rows (whole customers)")
    g.add_argument("--customers", type=int, default=None, help="target number of customers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.05, help="allowed relative deviation from --rows")
    ap.add_argument("--no-stratify", action="store_true", help="plain random sample instead of month x target strata")
    ap.add_argument("--keep-ids", action="store_true", help="do NOT pseudonymise customer ids")
    args, _ = ap.parse_known_args()
    if args.rows is None and args.customers is None:
        args.rows = 100_000

    t0 = time.time()
    dataset = ds.dataset(args.path)
    print(f"Scanning customer list from {args.path} ...")
    cust = customer_table(dataset)
    print(f"Population: {len(cust):,} customers, {cust['n_rows'].sum():,} rows ({time.time() - t0:,.0f}s)")

    chosen, attempt, dev = draw_sample(cust, rows=args.rows, customers=args.customers,
                                       stratify=not args.no_stratify, seed=args.seed, tol=args.tol)
    print(f"Selected {len(chosen):,} customers with {chosen['n_rows'].sum():,} rows "
          f"(draw #{attempt + 1}, deviation from target {dev:.1%})")

    tbl = extract_rows(dataset, chosen)
    check_complete(tbl, chosen)
    print(f"Extracted {tbl.num_rows:,} rows; completeness check passed ({time.time() - t0:,.0f}s)")

    summary, months = compare(cust, chosen)
    print("\nRepresentativeness check:\n", summary.round(3).to_string())
    print("\nM0 month mix:\n", months.round(3).to_string())

    if not args.keep_ids:
        tbl, mapping = pseudonymise(tbl, args.seed)
        map_path = os.path.splitext(args.out)[0] + "_id_map.csv"
        mapping.to_csv(map_path, index=False)
        print(f"\nCustomer ids pseudonymised. Keep {map_path} inside your secure environment; do not share it.")

    write_snapshot(tbl, args.out)
    manifest = {"source": args.path, "seed": args.seed, "draw_attempt": attempt, "stratified": not args.no_stratify,
                "customers": int(len(chosen)), "rows": int(tbl.num_rows), "ids_pseudonymised": not args.keep_ids}
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.out} ({tbl.num_rows:,} rows, {os.path.getsize(args.out) / 1e6:,.1f} MB). Total time {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
