"""
clean_data.py - read the collections data and return two cleaned tables.

    clients : one row per customer  (arrears / account information, plus log features)
    txns    : one row per transaction (plus log features and days_before_m0)

The raw file repeats the customer-level columns on every transaction row, and also contains exact duplicate
transaction rows. Each table is de-duplicated SEPARATELY, so the two tables have very different sizes
(for example ~120k clients vs tens of millions of transactions).

Works on the full 3.5 GB parquet file without loading it all into pandas, and also on CSV snapshots
produced by make_snapshot.py.

Usage in a notebook
-------------------
    from clean_data import load_and_clean
    clients, txns, report = load_and_clean("collections_data.parquet")      # or "snapshot.csv"

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
# Column definitions (from the data dictionary)
ID, M0, TARGET, PAY_DATE = "cifwithcheckdigit", "m0_monthenddate", "target_value", "payment_date"
TXN_DATE, TXN_TYPE, AMT, BAL = "transactiondate", "trantypedesc", "amount", "balance"
SPEND, CAT, METHOD = "spendgroup", "categoryname", "collectionsmethodname"

CLIENT_COLS = [ID, M0, PAY_DATE, "m0_arrearsamount", "due_count", "payment_count",
               "m0_total_due_with_arrears", "total_paid", TARGET, METHOD]
TXN_COLS = [ID, TXN_DATE, TXN_TYPE, AMT, BAL, SPEND, CAT]

MONEY_CLIENT = ["m0_arrearsamount", "m0_total_due_with_arrears", "total_paid"]   # log-transformed in `clients`
MONEY_TXN = [AMT, BAL]                                                          # log-transformed in `txns`
TXN_CATEGORICALS = [TXN_TYPE, SPEND, CAT]

# Columns that describe the outcome period (after M0). Fine for analysis and labelling, but never model inputs.
OUTCOME_COLS = ["payment_count", "total_paid", TARGET]

CSV_DTYPES = {ID: "string", "m0_arrearsamount": "float32", "due_count": "uint8", "payment_count": "uint8",
              "m0_total_due_with_arrears": "float32", "total_paid": "float32", TARGET: "int8",
              METHOD: "category", TXN_TYPE: "category", AMT: "float32", BAL: "float32",
              SPEND: "category", CAT: "category"}


# ----------------------------------------------------------------------------------------------
def signed_log1p(x):
    """sign(x) * log(1 + |x|): compresses heavy tails and is defined for zero and negative values.
    For non-negative values it equals log1p(x)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _to_datetime(s):
    """Dates may arrive as timestamps, date32 (python date objects), text or categoricals: normalise to datetime64."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(str)
    return pd.to_datetime(s, errors="coerce")


def _decode(col):
    """Dictionary-encoded column -> plain values."""
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


def _categories(col):
    """Distinct values of a (possibly dictionary-encoded) arrow column."""
    if pa.types.is_dictionary(col.type):
        return sorted(col.unify_dictionaries().chunk(0).dictionary.to_pylist())
    return sorted(pc.unique(col).to_pylist())


def _table_getter(path):
    """Return f(columns) -> pyarrow.Table, for parquet (lazy column reads) or CSV (read once)."""
    if path.lower().endswith(".parquet"):
        dataset = ds.dataset(path)
        return lambda cols: dataset.to_table(columns=cols)
    df = pd.read_csv(path, dtype=CSV_DTYPES, parse_dates=[M0, PAY_DATE, TXN_DATE])
    df[ID] = df[ID].astype(str)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    del df
    return lambda cols: tbl.select(cols)


# ----------------------------------------------------------------------------------------------
def build_clients(get_table, report):
    """One row per customer. Duplicates are removed by taking the DISTINCT customer-level records."""
    t = get_table(CLIENT_COLS)
    report["clients_raw_rows"] = t.num_rows

    """
    REMOVED FOR NOW BECAUSE DUPLICATES MAY NOT BE A DATA ISSUE

    try:
        distinct = t.group_by(CLIENT_COLS).aggregate([])
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):          # differing dictionaries across chunks
        distinct = pa.Table.from_arrays([_decode(t[c]) for c in CLIENT_COLS], names=CLIENT_COLS) \
                     .group_by(CLIENT_COLS).aggregate([])
    del t
    clients = distinct.to_pandas()
    clients[ID] = clients[ID].astype(str)
    for c in (M0, PAY_DATE):
        clients[c] = _to_datetime(clients[c])
    report["clients_distinct_records"] = len(clients)
    """

    # A customer should have exactly one distinct record; if not, keep the first and report it.
    # There are multiple records due to transaction level data
    conflicts = clients[ID].duplicated(keep=False)
    report["clients_with_conflicting_records"] = int(clients.loc[conflicts, ID].nunique())
    clients = (clients.sort_values([ID, M0, PAY_DATE]).drop_duplicates(subset=ID, keep="first")
                      .reset_index(drop=True))
    report["clients_after"] = len(clients)

    # Log features and a ratio
    for c in MONEY_CLIENT:
        clients[f"{c}_slog"] = signed_log1p(clients[c].astype("float32")).astype("float32")
    clients["arrears_share_of_due"] = (clients["m0_arrearsamount"] /
                                       clients["m0_total_due_with_arrears"].replace(0, np.nan)).astype("float32")
    clients["days_payment_before_m0"] = (clients[M0] - clients[PAY_DATE]).dt.days.astype("Int16")
    return clients


def build_transactions(get_table, clients, batch_customers, report, id_as_category=True, verbose=True):
    """De-duplicate transactions (exact duplicates on all transaction fields), batch by customer for memory."""
    t = get_table(TXN_COLS)
    report["txn_raw_rows"] = t.num_rows
    cats = {c: pd.CategoricalDtype(_categories(t[c])) for c in TXN_CATEGORICALS}
    id_dtype = pd.CategoricalDtype(clients[ID].tolist())
    m0_by_id = clients.set_index(ID)[M0]

    ids_all = clients[ID].to_numpy()
    parts, n_matched, n_null, n_dup = [], 0, 0, 0
    n_batches = int(np.ceil(len(ids_all) / batch_customers))
    t0 = time.time()
    for b in range(n_batches):
        batch_ids = ids_all[b * batch_customers:(b + 1) * batch_customers]
        mask = pc.is_in(t[ID], value_set=pa.array(batch_ids, type=pa.string()))
        sub = t.filter(mask).to_pandas()
        sub[ID] = sub[ID].astype(str)
        sub[TXN_DATE] = _to_datetime(sub[TXN_DATE])
        n_matched += len(sub)

        n0 = len(sub)
        sub = sub.dropna(subset=[ID, TXN_DATE, AMT])
        n_null += n0 - len(sub)

        n0 = len(sub)
        sub = sub.drop_duplicates(subset=TXN_COLS)
        n_dup += n0 - len(sub)

        for c in MONEY_TXN:
            sub[f"{c}_slog"] = signed_log1p(sub[c].astype("float32")).astype("float32")
        sub["days_before_m0"] = (sub[ID].map(m0_by_id) - sub[TXN_DATE]).dt.days.astype("Int16")
        for c in TXN_CATEGORICALS:
            sub[c] = sub[c].astype(cats[c])
        sub[ID] = sub[ID].astype(id_dtype) if id_as_category else sub[ID]
        parts.append(sub.sort_values([ID, TXN_DATE]).reset_index(drop=True))
        if verbose:
            print(f"  transactions batch {b + 1}/{n_batches}: {len(sub):,} rows kept ({time.time() - t0:,.0f}s)")
        del sub

    txns = pd.concat(parts, ignore_index=True)          # batches are consecutive slices of the sorted client ids
    report["txn_rows_without_client"] = report["txn_raw_rows"] - n_matched
    report["txn_rows_dropped_missing_key_fields"] = n_null
    report["txn_exact_duplicates_removed"] = n_dup
    report["txn_after"] = len(txns)
    return txns


def load_and_clean(path, batch_customers=10_000, id_as_category=True, verbose=True):
    """
    Read `path` (.parquet or .csv) and return (clients, txns, report).

    clients: one row per customer. Raw columns, `<money>_slog` log features, arrears_share_of_due,
             days_payment_before_m0. Duplicates removed on the customer-level columns only.
    txns:    one row per distinct transaction. Raw columns, amount_slog, balance_slog, days_before_m0.
             Exact duplicates removed on the transaction fields only.
    report:  dict of row counts before/after and what was removed.
    """
    t0 = time.time()
    report = {}
    get_table = _table_getter(path)
    if verbose:
        print("Building clients table ...")
    clients = build_clients(get_table, report)
    if verbose:
        print(f"  {report['clients_raw_rows']:,} raw rows -> {report['clients_after']:,} clients "
              f"({time.time() - t0:,.0f}s)\nBuilding transactions table ...")
    txns = build_transactions(get_table, clients, batch_customers, report, id_as_category, verbose)

    # Integrity checks between the two tables
    report["clients_without_transactions"] = len(clients) - txns[ID].nunique()
    if verbose:
        print_report(report)
        print(f"Done in {time.time() - t0:,.0f}s")
    return clients, txns, report


def print_report(r):
    rows = [
        ("CLIENTS", ""),
        ("  raw rows (customer fields repeated per transaction)", f"{r['clients_raw_rows']:,}"),
        ("  distinct customer records", f"{r['clients_distinct_records']:,}"),
        ("  customers with conflicting records (kept first)", f"{r['clients_with_conflicting_records']:,}"),
        ("  clients after cleaning", f"{r['clients_after']:,}"),
        ("TRANSACTIONS", ""),
        ("  raw rows", f"{r['txn_raw_rows']:,}"),
        ("  rows with no matching client", f"{r['txn_rows_without_client']:,}"),
        ("  dropped (missing id / date / amount)", f"{r['txn_rows_dropped_missing_key_fields']:,}"),
        ("  exact duplicates removed", f"{r['txn_exact_duplicates_removed']:,} "
                                      f"({r['txn_exact_duplicates_removed'] / max(r['txn_raw_rows'], 1):.2%} of raw)"),
        ("  transactions after cleaning", f"{r['txn_after']:,}"),
        ("CHECKS", ""),
        ("  clients without any transaction", f"{r['clients_without_transactions']:,}"),
    ]
    print("\n".join(f"{a:<58}{b:>16}" for a, b in rows))


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
    args, _ = ap.parse_known_args()
    c, t, _ = load_and_clean(args.path, args.batch_customers, id_as_category=not args.plain_ids)
    save(c, t, args.out_dir)