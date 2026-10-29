"""
clean_data.py (v2, observation-aware) - read the collections data and return two cleaned tables.

    clients : one row per OBSERVATION (customer + obs_id) - arrears / account information + log features
    txns    : one row per distinct transaction WITHIN an observation - plus log features and days_before_m0

Why "observation"? The extract now carries obs_id / obs_seq, so one customer can appear in several observation
windows, and the same transaction can legitimately appear in more than one of them (overlapping windows).
Duplicates are therefore only removed WITHIN an observation, never across observations. If the file has no
obs_id the key falls back to (customer, M0), which reproduces the earlier behaviour.

Which columns belong to which table is INFERRED from the data: a column that is constant within every sampled
observation is observation-level (-> clients); one that varies is transaction-level (-> txns). New columns such as
product_group or tx_period are therefore placed automatically. The split is printed (and stored in
report["levels"]) so you can check it, and can be overridden with client_cols= / txn_cols=.

Collapsing the repeated observation-level columns into one row per observation is NOT a data-quality step; it
just removes their repetition across transaction rows. The data-quality step is the exact-duplicate removal in
`txns` (plus the optional conflict check, check_conflicts=True).

Usage in a notebook
-------------------
    from clean_data import load_and_clean
    clients, txns, report = load_and_clean("snapshot/snapshot.parquet")

Usage from the command line (writes clients_clean.parquet and transactions_clean.parquet)
-------------------------------------------------------------------------------------------
    python clean_data.py --path collections_data.parquet --out-dir cleaned
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

# ----------------------------------------------------------------------------------------------
ID, OBS = "cifwithcheckdigit", "obs_id"
M0, TARGET, PAY_DATE = "m0_monthenddate", "target_value", "payment_date"
TXN_DATE, TXN_TYPE, AMT, BAL = "transactiondate", "trantypedesc", "amount", "balance"
SPEND, CAT, METHOD = "spendgroup", "categoryname", "collectionsmethodname"

# Used only when level inference is switched off (infer_levels_=False)
DEFAULT_TXN_COLS = [TXN_DATE, TXN_TYPE, AMT, BAL, SPEND, CAT, "days_from_payment", "tx_period"]

MONEY_CLIENT = ["m0_arrearsamount", "m0_total_due_with_arrears", "total_paid", "crccreditlimit"]  # logged if present
MONEY_TXN = [AMT, BAL]
DATE_COLS = [M0, PAY_DATE, TXN_DATE]

# Describe the outcome period / label (after M0). Fine for analysis, never model inputs.
OUTCOME_COLS = ["payment_count", "total_paid", TARGET, "target_rule"]

CSV_DTYPES = {ID: "string", "m0_arrearsamount": "float32", "due_count": "uint8", "payment_count": "uint8",
              "m0_total_due_with_arrears": "float32", "total_paid": "float32", TARGET: "int8",
              METHOD: "category", TXN_TYPE: "category", AMT: "float32", BAL: "float32",
              SPEND: "category", CAT: "category"}


# ----------------------------------------------------------------------------------------------
"""
PUT INTO SEPARATE FEATURE ENGINEERING FILE
def signed_log1p(x):
    sign(x) * log(1 + |x|): compresses heavy tails and is defined for zero and negative values.
    For non-negative values it equals log1p(x).
    return np.sign(x) * np.log1p(np.abs(x))
"""


def _to_datetime(s):
    """Dates may arrive as timestamps, date32 (python date objects), text or categoricals: normalise to datetime64."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(str)
    return pd.to_datetime(s, errors="coerce")


def _decode(col):
    """Dictionary-encoded column -> plain values."""
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


def _value_type(t):
    return t.value_type if pa.types.is_dictionary(t) else t


def _ids_array(ids, typ):
    """Customer ids as an arrow array of the column's own type (ids may be strings or integers in the file)."""
    typ = _value_type(typ)
    vals = [int(i) for i in ids] if pa.types.is_integer(typ) else [str(i) for i in ids]
    return pa.array(vals, type=typ)


def _categories(col):
    """Distinct values of a (possibly dictionary-encoded) arrow column."""
    if pa.types.is_dictionary(col.type):
        return sorted(col.unify_dictionaries().chunk(0).dictionary.to_pylist())
    return sorted(pc.unique(col).to_pylist())


def _group_first_row(t, key):
    """Row number of the first occurrence of each key (arrow group_by; decodes dictionary keys if needed)."""
    t = t.append_column("__row", pa.array(np.arange(t.num_rows, dtype=np.int64)))
    try:
        g = t.group_by(key).aggregate([("__row", "min")])
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        t2 = pa.Table.from_arrays([_decode(t[c]) for c in key] + [t["__row"]], names=key + ["__row"])
        g = t2.group_by(key).aggregate([("__row", "min")])
    return g["__row_min"]


# ----------------------------------------------------------------------------------------------
def _open(path):
    """Return (column_names, get) where get(cols, ids=None) -> pyarrow.Table (optionally only some customers)."""
    if path.lower().endswith(".parquet"):
        dataset = ds.dataset(path)
        names = dataset.schema.names
        vt = _value_type(dataset.schema.field(ID).type)

        def get(cols, ids=None):
            flt = ds.field(ID).isin(_ids_array(ids, vt)) if ids is not None else None
            return dataset.to_table(columns=cols, filter=flt)
        return names, get

    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    df = pd.read_csv(path, dtype={k: v for k, v in CSV_DTYPES.items() if k in hdr},
                     parse_dates=[c for c in DATE_COLS if c in hdr])
    df[ID] = df[ID].astype(str)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    del df

    def get(cols, ids=None):
        t = tbl.select(cols)
        if ids is not None:
            t = t.filter(pc.is_in(t[ID], value_set=_ids_array(ids, t[ID].type)))
        return t
    return hdr, get


def infer_levels(names, get, key, n_customers=2000, seed=0):
    """
    Split non-key columns into observation-level (constant within every sampled observation) and
    transaction-level (varies within an observation). Also flags integer row counters (consecutive 0..n or 1..n
    inside every observation, e.g. a per-observation sequence number): those are NOT used to decide whether two
    rows are duplicates, otherwise no duplicate could ever match.
    """
    others = [c for c in names if c not in key]
    ids_tbl = get([ID])
    ids = pd.Series(_decode(pc.unique(ids_tbl[ID])).to_pylist())
    del ids_tbl
    sample = ids.sample(min(n_customers, len(ids)), random_state=seed).tolist()
    sub = get(key + others, ids=sample).to_pandas()
    sub[ID] = sub[ID].astype(str)
    g = sub.groupby(key, observed=True, dropna=False)
    varies = {c: bool((g[c].nunique(dropna=False) > 1).any()) for c in others}
    client_cols = [c for c in others if not varies[c]]
    txn_cols = [c for c in others if varies[c]]

    row_counters = []
    for c in txn_cols:
        if pd.api.types.is_integer_dtype(sub[c].dtype):
            a = g[c].agg(["min", "max", "nunique", "size"])
            a = a[a["size"] > 1]
            if len(a) and ((a["nunique"] == a["size"]) & ((a["max"] - a["min"] + 1) == a["size"])).all():
                row_counters.append(c)
    return client_cols, txn_cols, row_counters


# ----------------------------------------------------------------------------------------------
def build_clients(get, key, client_cols, report, check_conflicts=False):
    """One row per observation: the first record of each key (the columns are repeated on every transaction row)."""
    cols = key + client_cols
    t = get(cols)
    report["clients_raw_rows"] = t.num_rows
    first = _group_first_row(t, key)
    clients = t.take(first).to_pandas()
    report["clients_after"] = len(clients)

    if check_conflicts:   # optional (slower): is every observation-level column really constant within its observation?
        try:
            distinct = t.group_by(cols).aggregate([])
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
            distinct = pa.Table.from_arrays([_decode(t[c]) for c in cols], names=cols).group_by(cols).aggregate([])
        report["clients_distinct_records"] = distinct.num_rows
        report["clients_with_conflicting_records"] = distinct.num_rows - len(clients)
    del t

    clients[ID] = clients[ID].astype(str)
    for c in (M0, PAY_DATE):
        if c in clients:
            clients[c] = _to_datetime(clients[c])
    clients = clients.sort_values(key).reset_index(drop=True)

    """
    for c in MONEY_CLIENT:
        if c in clients:
            clients[f"{c}_slog"] = signed_log1p(clients[c].astype("float32")).astype("float32")
    """
    if {"m0_arrearsamount", "m0_total_due_with_arrears"} <= set(clients.columns):
        clients["arrears_share_of_due"] = (clients["m0_arrearsamount"] /
                                           clients["m0_total_due_with_arrears"].replace(0, np.nan)).astype("float32")
    if {M0, PAY_DATE} <= set(clients.columns):
        clients["days_payment_before_m0"] = (clients[M0] - clients[PAY_DATE]).dt.days.astype("Int16")
    return clients


def build_transactions(get, key, txn_cols, dedupe_exclude, clients, batch_customers, report,
                       id_as_category=True, verbose=True):
    """Exact duplicates are removed WITHIN an observation (key + all transaction columns). Batched by customer."""
    tcols = key + txn_cols
    dedupe_cols = key + [c for c in txn_cols if c not in dedupe_exclude]
    t = get(tcols)
    report["txn_raw_rows"] = t.num_rows

    cats = {}
    for c in txn_cols:
        ty = t[c].type
        if pa.types.is_dictionary(ty) or pa.types.is_string(ty) or pa.types.is_large_string(ty):
            values = _categories(t[c])
            if len(values) <= 5000:                                  # skip high-cardinality text
                cats[c] = pd.CategoricalDtype(values)
    id_dtype = pd.CategoricalDtype(clients[ID].drop_duplicates().tolist())

    m0_in_key = M0 in key
    m0_map = clients[key + [M0]] if (M0 in clients and not m0_in_key) else None
    check_cols = [c for c in (ID, TXN_DATE, AMT) if c in tcols]
    ids_all = clients[ID].drop_duplicates().to_numpy()               # clients is sorted by customer id

    parts, n_matched, n_null, n_dup, n_no_obs = [], 0, 0, 0, 0
    n_batches = int(np.ceil(len(ids_all) / batch_customers))
    t0 = time.time()
    for b in range(n_batches):
        batch_ids = ids_all[b * batch_customers:(b + 1) * batch_customers]
        col = _decode(t[ID])
        mask = pc.is_in(col, value_set=_ids_array(batch_ids, col.type))
        sub = t.filter(mask).to_pandas()
        sub[ID] = sub[ID].astype(str)
        if TXN_DATE in sub:
            sub[TXN_DATE] = _to_datetime(sub[TXN_DATE])
        n_matched += len(sub)

        n0 = len(sub)
        sub = sub.dropna(subset=check_cols)
        n_null += n0 - len(sub)

        n0 = len(sub)
        sub = sub.drop_duplicates(subset=dedupe_cols)                # key is included => within an observation only
        n_dup += n0 - len(sub)

        """
        for c in MONEY_TXN:
            if c in sub:
                sub[f"{c}_slog"] = signed_log1p(sub[c].astype("float32")).astype("float32")
        """
        if m0_in_key and TXN_DATE in sub:                              # no obs_id: M0 is part of the key
            sub[M0] = _to_datetime(sub[M0])
            sub["days_before_m0"] = (sub[M0] - sub[TXN_DATE]).dt.days.astype("Int16")
        elif m0_map is not None and TXN_DATE in sub:
            sub = sub.merge(m0_map, on=key, how="left")
            n_no_obs += int(sub[M0].isna().sum())
            sub["days_before_m0"] = (sub[M0] - sub[TXN_DATE]).dt.days.astype("Int16")
            sub = sub.drop(columns=[M0])
        for c, dt in cats.items():
            sub[c] = sub[c].astype(dt)
        if id_as_category:
            sub[ID] = sub[ID].astype(id_dtype)
        sort_cols = key + ([TXN_DATE] if TXN_DATE in sub else [])
        parts.append(sub.sort_values(sort_cols).reset_index(drop=True))
        if verbose:
            print(f"  transactions batch {b + 1}/{n_batches}: {len(sub):,} rows kept ({time.time() - t0:,.0f}s)")
        del sub

    txns = pd.concat(parts, ignore_index=True)       # batches are consecutive slices of the sorted client ids
    report["txn_rows_without_client"] = report["txn_raw_rows"] - n_matched
    report["txn_rows_unmatched_observation"] = n_no_obs
    report["txn_rows_dropped_missing_key_fields"] = n_null
    report["txn_exact_duplicates_removed"] = n_dup
    report["txn_after"] = len(txns)
    return txns


def load_and_clean(path, batch_customers=10_000, id_as_category=True, verbose=True,
                   infer_levels_=True, client_cols=None, txn_cols=None, dedupe_exclude=None,
                   check_conflicts=False):
    """
    Read `path` (.parquet or .csv) and return (clients, txns, report).

    clients: one row per observation (customer + obs_id). Raw columns, `<money>_slog` log features,
             arrears_share_of_due, days_payment_before_m0.
    txns:    one row per distinct transaction within an observation. Raw columns, amount_slog, balance_slog,
             days_before_m0. Exact duplicates are removed within each observation only.
    report:  dict of row counts and what was removed; report["levels"] holds the column split that was used.

    dedupe_exclude: transaction-level columns ignored when deciding if two rows are duplicates (default: detected
                    integer row counters such as a per-observation sequence number).
    """
    t0 = time.time()
    report = {}
    names, get = _open(path)
    key = [ID, OBS] if OBS in names else [ID, M0]
    report["key"] = key

    others = [c for c in names if c not in key]
    detected_counters = []
    if client_cols is None and txn_cols is None:
        if infer_levels_:
            client_cols, txn_cols, detected_counters = infer_levels(names, get, key)
        else:
            txn_cols = [c for c in DEFAULT_TXN_COLS if c in others]
            client_cols = [c for c in others if c not in txn_cols]
    elif client_cols is None:
        client_cols = [c for c in others if c not in txn_cols]
    elif txn_cols is None:
        txn_cols = [c for c in others if c not in client_cols]
    if dedupe_exclude is None:
        dedupe_exclude = detected_counters
    report["levels"] = {"client": client_cols, "transaction": txn_cols, "ignored_for_duplicates": list(dedupe_exclude)}
    if verbose:
        print(f"Observation key: {key}")
        print(f"Observation-level columns ({len(client_cols)}): {client_cols}")
        print(f"Transaction-level columns ({len(txn_cols)}): {txn_cols}")
        print(f"Ignored when matching duplicates (row counters): {list(dedupe_exclude)}\nBuilding clients table ...")

    clients = build_clients(get, key, client_cols, report, check_conflicts)
    report["customers"] = clients[ID].nunique()
    report["max_observations_per_customer"] = int(clients.groupby(ID).size().max())
    if verbose:
        print(f"  {report['clients_raw_rows']:,} raw rows -> {report['clients_after']:,} observations "
              f"({report['customers']:,} customers) ({time.time() - t0:,.0f}s)\nBuilding transactions table ...")

    txns = build_transactions(get, key, txn_cols, dedupe_exclude, clients, batch_customers, report,
                              id_as_category, verbose)
    report["clients_without_transactions"] = len(clients) - txns.groupby(key, observed=True).ngroups

    for name, df in (("clients", clients), ("transactions", txns)):     # reported, never dropped automatically
        if "exclude_flag" in df:
            report[f"exclude_flag_in_{name}"] = df["exclude_flag"].value_counts(dropna=False).to_dict()
    if verbose:
        print_report(report)
        print(f"Done in {time.time() - t0:,.0f}s")
    return clients, txns, report


def print_report(r):
    rows = [
        ("CLIENTS (one row per observation)", ""),
        ("  raw rows (observation fields repeated per transaction)", f"{r['clients_raw_rows']:,}"),
        ("  observations after cleaning", f"{r['clients_after']:,}"),
        ("  distinct customers", f"{r['customers']:,}"),
        ("  max observations per customer", f"{r['max_observations_per_customer']:,}"),
    ]
    if "clients_with_conflicting_records" in r:
        rows.append(("  observations with conflicting records (kept first)", f"{r['clients_with_conflicting_records']:,}"))
    rows += [
        ("TRANSACTIONS", ""),
        ("  raw rows", f"{r['txn_raw_rows']:,}"),
        ("  rows with no matching client", f"{r['txn_rows_without_client']:,}"),
        ("  rows with no matching observation", f"{r['txn_rows_unmatched_observation']:,}"),
        ("  dropped (missing id / date / amount)", f"{r['txn_rows_dropped_missing_key_fields']:,}"),
        ("  exact duplicates removed (within observation)", f"{r['txn_exact_duplicates_removed']:,} "
                                      f"({r['txn_exact_duplicates_removed'] / max(r['txn_raw_rows'], 1):.2%} of raw)"),
        ("  transactions after cleaning", f"{r['txn_after']:,}"),
        ("CHECKS", ""),
        ("  observations without any transaction", f"{r['clients_without_transactions']:,}"),
    ]
    print("\n".join(f"{a:<62}{b:>16}" for a, b in rows))
    for k in [k for k in r if k.startswith("exclude_flag_in_")]:
        print(f"{k}: {r[k]}")


def save(clients, txns, out_dir="cleaned"):
    os.makedirs(out_dir, exist_ok=True)
    clients.to_parquet(os.path.join(out_dir, "clients_clean.parquet"), index=False)
    txns.to_parquet(os.path.join(out_dir, "transactions_clean.parquet"), index=False)
    print(f"Saved to {out_dir}/clients_clean.parquet and {out_dir}/transactions_clean.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="collections_data.parquet", help="source file (.parquet or .csv)")
    ap.add_argument("--out-dir", default="cleaned")
    ap.add_argument("--batch-customers", type=int, default=10_000, help="lower this if memory is tight")
    ap.add_argument("--plain-ids", action="store_true", help="keep customer ids as plain strings instead of category")
    ap.add_argument("--no-infer", action="store_true", help="use the default column split instead of inferring it")
    ap.add_argument("--check-conflicts", action="store_true", help="verify observation-level columns are constant (slower)")
    args, _ = ap.parse_known_args()
    c, t, _ = load_and_clean(args.path, args.batch_customers, id_as_category=not args.plain_ids,
                             infer_levels_=not args.no_infer, check_conflicts=args.check_conflicts)
    save(c, t, args.out_dir)