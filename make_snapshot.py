"""
Create a random customer-level snapshot of a large transaction parquet file.

Randomly selects customers and keeps ALL of their transaction rows (all observations), so every selected
customer is complete. Observations with exclude_flag == 1 are dropped unless --keep-excluded is given.
Memory-safe: only a few columns are scanned to build the customer list, then only the selected customers'
rows are read.

All output goes into a folder (default: ./snapshot):
    snapshot/<out>                  the snapshot (.parquet or .csv, follows the extension)
    snapshot/<name>_id_map.csv      original -> pseudonymised customer ids   (keep private)
    snapshot/<name>_obs_id_map.csv  original -> pseudonymised obs ids        (keep private)
    snapshot/<name>_manifest.json   settings used to build the snapshot

Examples
--------
    python make_snapshot.py --customers 5000 --out snapshot.parquet
    python make_snapshot.py --customers 20000 --out snapshot_20k.parquet --outdir snapshot
    python make_snapshot.py --rows 2000000 --out snapshot.csv

From a notebook:   %run make_snapshot.py --customers 5000 --out snapshot.parquet

Reading back (restores dtypes for CSV):
    from make_snapshot import read_snapshot
    df = read_snapshot("snapshot/snapshot.parquet")
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ID, OBS, M0, TARGET = "cifwithcheckdigit", "obs_id", "m0_monthenddate", "target_value"
EXCLUDE = ds.field("exclude_flag") == 0


# ----------------------------------------------------------------------------------------------
def _decode(col):
    """Dictionary-encoded (categorical) column -> plain values. Other types are returned unchanged."""
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


def _value_type(t):
    return t.value_type if pa.types.is_dictionary(t) else t


def customer_table(dataset, filt=EXCLUDE):
    """
    One row per customer: rows, observations, bad observations, first M0 month.
    Aggregates to observation level first (a client can have several observations with different targets),
    then to client level. Stratum = first M0 month x ever-bad.
    """
    t = dataset.to_table(columns=[ID, OBS, M0, TARGET], filter=filt)
    t = t.set_column(t.schema.get_field_index(ID), ID, _decode(t[ID]))
    t = t.set_column(t.schema.get_field_index(OBS), OBS, _decode(t[OBS]))
    obs = (t.group_by([ID, OBS])
             .aggregate([(OBS, "count"), (M0, "min"), (TARGET, "max")])
             .to_pandas()
             .rename(columns={f"{OBS}_count": "n_rows", f"{M0}_min": "m0", f"{TARGET}_max": "target"}))
    obs[ID] = obs[ID].astype(str)
    cust = obs.groupby(ID).agg(n_rows=("n_rows", "sum"), n_obs=(OBS, "size"),
                               bad_obs=("target", "sum"), m0=("m0", "min")).reset_index()
    cust["ever_bad"] = (cust["bad_obs"] > 0).astype(int)
    cust["month"] = pd.to_datetime(cust["m0"]).dt.to_period("M").astype(str)
    cust["stratum"] = cust["month"] + "|" + cust["ever_bad"].astype(str)
    return cust


def draw_sample(cust, rows=None, customers=None, stratify=True, seed=42, tol=0.05, max_tries=200):
    """
    Random customers. `rows` targets a total row count (whole customers, so approximate: the draw is
    repeated with successive seeds until within `tol`). `customers` targets a number of customers.
    Stratified sampling keeps the first-M0-month x ever-bad mix of the population.
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
        dev = (abs(chosen["n_rows"].sum() - rows) / rows if rows is not None
               else abs(len(chosen) - customers) / customers)
        if best is None or dev < best[0]:
            best = (dev, k, chosen)
        if dev <= tol:
            break
    return best[2].copy(), best[1], best[0]


def extract_rows(dataset, chosen, filt=EXCLUDE):
    id_type = _value_type(dataset.schema.field(ID).type)
    ids = pa.array(chosen[ID].astype(str).tolist()).cast(id_type)      # str -> int32 (or stays str)
    f = ds.field(ID).isin(ids) if filt is None else (ds.field(ID).isin(ids) & filt)
    tbl = dataset.to_table(filter=f)
    return tbl.set_column(tbl.schema.get_field_index(ID), ID, _decode(tbl[ID]))


def check_complete(tbl, chosen):
    """Every selected customer must have exactly the same number of rows as in the (filtered) full file."""
    got = tbl.group_by(ID).aggregate([(ID, "count")]).to_pandas().set_index(ID)[f"{ID}_count"]
    got.index = got.index.astype(str)
    want = chosen.set_index(ID)["n_rows"]
    want.index = want.index.astype(str)
    bad = got.reindex(want.index) != want
    assert not bad.any(), f"{int(bad.sum())} customers are incomplete in the snapshot"
    assert len(got) == len(want), "snapshot contains customers that were not selected"


def pseudonymise(tbl, col, prefix, seed):
    """Replace a column's values with random codes (prefix0000001...). Returns table and the map (keep private)."""
    data = _decode(tbl[col])
    uniq = pc.unique(data)
    order = np.random.default_rng(seed).permutation(len(uniq))
    labels = np.array([f"{prefix}{i:07d}" for i in range(len(uniq))])[order]
    pos = pc.index_in(data, value_set=uniq).to_numpy(zero_copy_only=False)
    mapping = pd.DataFrame({"original_id": uniq.to_pylist(), "snapshot_id": labels})
    return tbl.set_column(tbl.schema.get_field_index(col), col, pa.array(labels[pos])), mapping


def compare(cust, chosen):
    """Population vs snapshot: is the sample representative?"""
    def summary(d):
        w = d["n_rows"]
        return pd.Series({"customers": len(d), "observations": int(d["n_obs"].sum()), "rows": int(w.sum()),
                          "obs_bad_rate": d["bad_obs"].sum() / d["n_obs"].sum(),
                          "obs/cust mean": d["n_obs"].mean(), "rows/cust mean": w.mean(),
                          "rows/cust median": w.median(), "rows/cust p95": w.quantile(.95)})
    out = pd.DataFrame({"population": summary(cust), "snapshot": summary(chosen)})
    months = pd.DataFrame({"population": cust["month"].value_counts(normalize=True),
                           "snapshot": chosen["month"].value_counts(normalize=True)}).sort_index()
    return out, months


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
    ID: "string", OBS: "string", "obs_seq": "uint16",
    "m0_arrearsamount": "float32", "crccreditlimit": "float32", "due_count": "uint8", "payment_count": "uint8",
    "m0_total_due_with_arrears": "float32", "total_paid": "float32", TARGET: "int8",
    "target_rule": "category", "new_to_arrears_flag": "int8", "product_group": "category",
    "m0_start_cd_group": "category", "days_from_payment": "float32", "tx_period": "category",
    "collectionsmethodname": "category", "trantypedesc": "category", "amount": "float32", "balance": "float32",
    "spendgroup": "category", "categoryname": "category", "exclude_flag": "int8",
}
SNAPSHOT_DATES = [M0, "payment_date", "transactiondate"]


def read_snapshot(path):
    """Read a snapshot written by this script and restore the dtypes of the original parquet file."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {k: v for k, v in SNAPSHOT_DTYPES.items() if k in header}
    dates = [c for c in SNAPSHOT_DATES if c in header]
    return pd.read_csv(path, dtype=dtypes, parse_dates=dates)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="Collectionsdata_updated.parquet", help="source parquet file")
    ap.add_argument("--outdir", default="snapshot", help="folder for all output files")
    ap.add_argument("--out", default="snapshot.parquet", help="snapshot file name; .csv or .parquet")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rows", type=int, default=None, help="target number of transaction rows (whole customers)")
    g.add_argument("--customers", type=int, default=None, help="target number of customers (default 5000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.05, help="allowed relative deviation from --rows")
    ap.add_argument("--no-stratify", action="store_true", help="plain random sample instead of month x ever-bad strata")
    ap.add_argument("--keep-ids", action="store_true", help="do NOT pseudonymise customer and observation ids")
    ap.add_argument("--keep-excluded", action="store_true", help="keep observations with exclude_flag == 1")
    args, _ = ap.parse_known_args()
    if args.rows is None and args.customers is None:
        args.customers = 5_000

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, os.path.basename(args.out))
    stem = os.path.splitext(out_path)[0]
    filt = None if args.keep_excluded else EXCLUDE

    t0 = time.time()
    dataset = ds.dataset(args.path)
    print(f"Scanning customer list from {args.path} ...")
    cust = customer_table(dataset, filt)
    print(f"Population: {len(cust):,} customers, {cust['n_obs'].sum():,} observations, "
          f"{cust['n_rows'].sum():,} rows ({time.time() - t0:,.0f}s)")

    chosen, attempt, dev = draw_sample(cust, rows=args.rows, customers=args.customers,
                                       stratify=not args.no_stratify, seed=args.seed, tol=args.tol)
    print(f"Selected {len(chosen):,} customers with {chosen['n_rows'].sum():,} rows "
          f"(draw #{attempt + 1}, deviation from target {dev:.1%})")

    tbl = extract_rows(dataset, chosen, filt)
    check_complete(tbl, chosen)
    print(f"Extracted {tbl.num_rows:,} rows; completeness check passed ({time.time() - t0:,.0f}s)")

    summary, months = compare(cust, chosen)
    print("\nRepresentativeness check:\n", summary.round(3).to_string())
    print("\nFirst-M0 month mix:\n", months.round(3).to_string())

    if not args.keep_ids:
        tbl, cmap = pseudonymise(tbl, ID, "C", args.seed)
        tbl, omap = pseudonymise(tbl, OBS, "O", args.seed + 1)
        cmap.to_csv(stem + "_id_map.csv", index=False)
        omap.to_csv(stem + "_obs_id_map.csv", index=False)
        print(f"\nIds pseudonymised. Keep the *_id_map.csv files in {args.outdir}/ private; do not share them.")

    write_snapshot(tbl, out_path)
    manifest = {"source": args.path, "seed": args.seed, "draw_attempt": attempt, "stratified": not args.no_stratify,
                "customers": int(len(chosen)), "observations": int(chosen["n_obs"].sum()), "rows": int(tbl.num_rows),
                "ids_pseudonymised": not args.keep_ids, "excluded_dropped": not args.keep_excluded}
    with open(stem + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path} ({tbl.num_rows:,} rows, {os.path.getsize(out_path) / 1e6:,.1f} MB). "
          f"Total time {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()