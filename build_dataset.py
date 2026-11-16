"""
build_dataset.py - one cleaned, leak-free modelling dataset from the snapshot, for testing.

    python build_dataset.py --path snapshot/snapshot.parquet --out-dir dataset
    python build_dataset.py --path snapshot/snapshot.parquet --out-dir dataset_small --sample-customers 2000   # quick test run

Pipeline
  1. clean_data.load_and_clean      -> clients (one row per obs_id) and txns (distinct transactions per obs_id)
  2. checks / cleaning              -> drop exclude_flag==1, observations without usable transactions, non-finite money,
                                       label audit (target_rule vs target_value), duplicate ids
  3. features_transactions          -> running-state features at M0 (state after the last pre-M0 transaction)
  4. build_features                 -> pay-day / pay-cycle / per-period income and spend features (batched by customer)
  5. static block                   -> point-in-time attributes known at M0 (the baseline the time-aware features must beat)
  6. assemble                       -> X_static, X_time, X_all (index = obs_id as text), labels, customer-grouped split
  7. QA report + leak tripwire      -> qa_report.txt, manifest.json

Outputs (in --out-dir)
  X_static.parquet   point-in-time attributes only              (one row per obs_id, index obs_id)
  X_time.parquet     time-aware transactional features only
  X_all.parquet      static + time-aware
  labels.parquet     obs_id, customer, target_value, target_rule (number 2-8), exclude_flag, dates, target_from_rule
                     (NEVER model inputs; customer is only for grouped splits)
  split.parquet      obs_id, customer, split ('train' / 'test'), customer-grouped, fixed seed
  qa_report.txt, manifest.json

Use in a notebook
    from build_dataset import load_dataset, impute_train_only
    D = load_dataset("dataset")
    X, y, groups, tr, te = D["X_all"], D["y"], D["groups"], D["train"], D["test"]     # same shapes as select_features expects
    Xi = impute_train_only(X, tr)                                                       # median fill fitted on train rows only

Missing values are LEFT as NaN (the reason is in the *_flag columns); fill them inside the train-only pipeline.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import build_features as bf
import clean_data as cd
import features_transactions as ft

ID, OBS, M0 = "cifwithcheckdigit", "obs_id", "m0_monthenddate"
TARGET, RULE = "target_value", "target_rule"

# Never inputs: outcome period, the label, ids, dates, bookkeeping. Keep them out of X (they live in labels.parquet).
NEVER_FEATURES = ["payment_count", "total_paid", "total_paid_slog", TARGET, RULE, "exclude_flag", ID, OBS, M0,
                  "payment_date", "days_from_payment", "days_payment_before_m0", "obs_seq_raw"]
# Point-in-time attributes taken from `clients` (everything else there is excluded or derived elsewhere)
STATIC_CATEGORICAL_MAX_LEVELS = 30
MAX_ONEHOT_LEVELS = 12               # text columns with more levels than this are dropped from the time block
RARE_SHARE = 0.01                      # one-hot levels rarer than this are pooled into "other"

# Rule -> expected target, from the rule table (2,3,6 -> 0; 4,5,7,8 -> 1). Used ONLY to audit the labels.
RULE_TO_TARGET = {2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 1, 8: 1}


# ----------------------------------------------------------------------------------------------
def log(msg, buf=None):
    print(msg)
    if buf is not None:
        buf.append(msg)


def _text_index(s):
    return pd.Index(pd.Series(s).astype(str).to_numpy(), name=OBS)


def rule_number(s):
    """'4 Payments received <= 0' / 4 / '4' -> 4 (Int64)."""
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64")
    return s.astype("string").str.extract(r"^\s*(\d+)", expand=False).astype("Int64")


# ----------------------------------------------------------------------------------------------
# 2. checks and cleaning
# ----------------------------------------------------------------------------------------------
def clean_tables(clients, txns, drop_excluded=True, rep=None):
    r = rep if rep is not None else []
    n0 = len(clients)
    clients = clients.copy()
    clients[OBS] = clients[OBS].astype(str)
    txns = txns.copy()
    txns[OBS] = txns[OBS].astype(str)

    if clients[OBS].duplicated().any():
        k = int(clients[OBS].duplicated().sum())
        log(f"  {k} duplicate obs_id rows in clients -> kept the first", r)
        clients = clients.drop_duplicates(OBS)

    if drop_excluded and "exclude_flag" in clients:
        bad = clients["exclude_flag"].fillna(0).astype(int) == 1
        log(f"  exclude_flag == 1: dropped {int(bad.sum()):,} observations", r)
        clients = clients[~bad]

    for c in (M0, "payment_date"):
        if c in clients:
            clients[c] = pd.to_datetime(clients[c], errors="coerce")
    miss_m0 = clients[M0].isna()
    if miss_m0.any():
        log(f"  {int(miss_m0.sum()):,} observations without M0 date -> dropped", r)
        clients = clients[~miss_m0]

    if TARGET in clients:
        miss_y = clients[TARGET].isna()
        if miss_y.any():
            log(f"  {int(miss_y.sum()):,} observations without target_value -> dropped", r)
            clients = clients[~miss_y]

    # transactions: keep the ones that belong to a kept observation; make money finite
    txns = txns[txns[OBS].isin(clients[OBS])]
    for c in ("amount", "balance"):
        if c in txns:
            bad = ~np.isfinite(txns[c].astype("float64"))
            if bad.any():
                log(f"  {int(bad.sum()):,} transactions with non-finite {c} -> set missing", r)
                txns.loc[bad, c] = np.nan
    if "amount" in txns:
        na = txns["amount"].isna()
        if na.any():
            log(f"  {int(na.sum()):,} transactions without an amount -> dropped", r)
            txns = txns[~na]

    # observations with no transaction at all cannot carry transactional features
    has_tx = clients[OBS].isin(txns[OBS].unique())
    if (~has_tx).any():
        log(f"  {int((~has_tx).sum()):,} observations without any transaction -> dropped", r)
        clients = clients[has_tx]
    log(f"  clients: {n0:,} -> {len(clients):,} observations; transactions: {len(txns):,} rows", r)
    return clients.reset_index(drop=True), txns.reset_index(drop=True)


def label_audit(clients, rep):
    """target_rule vs target_value: the rule table says which rules mean 'arrears' (1). Report disagreement, change nothing."""
    if RULE not in clients or TARGET not in clients:
        return None
    rn = rule_number(clients[RULE])
    exp = rn.map(RULE_TO_TARGET).astype("Int64")
    ok = exp.notna()
    out = {"rows_with_rule": int(ok.sum())}
    log("\nLabel audit (rule table vs target_value)", rep)
    log("  prevalence of target_value: %.4f" % clients[TARGET].astype(float).mean(), rep)
    ct = pd.crosstab(rn, clients[TARGET].astype(int))
    log("  target_value by rule:\n" + ct.to_string(), rep)
    if ok.any():
        dis = float((exp[ok].astype(int) != clients.loc[ok.to_numpy(), TARGET].astype(int).to_numpy()).mean())
        out["disagreement"] = dis
        log(f"  disagreement between target_value and the rule-implied target: {dis:.1%}"
            + ("   <-- confirm the rule->target mapping / column definitions with the data owner" if dis > 0.05 else ""), rep)
    return out


# ----------------------------------------------------------------------------------------------
# 4. pay-day features, batched by customer
# ----------------------------------------------------------------------------------------------
def pay_features(clients, txns, batch_customers=20_000, rep=None):
    """build_features.build_all on clients joined to txns, in customer batches (every feature is per observation, so
    batching changes nothing). Returns (features indexed by obs_id as text)."""
    need = [c for c in bf.OBS_COLS if c not in clients.columns]
    if need:
        raise KeyError(f"clients is missing columns build_features needs: {need}")
    tcols = ["obs_id", "transactiondate", "amount", "balance", "trantypedesc", "spendgroup", "categoryname",
             "days_from_payment", "tx_period"]
    missing = [c for c in tcols if c not in txns.columns]
    if missing:
        raise KeyError(f"txns is missing columns build_features needs: {missing}")
    cust = clients[ID].astype(str)
    codes = pd.factorize(cust)[0]
    nb = max(1, int(np.ceil(cust.nunique() / batch_customers)))
    parts = []
    for b in range(nb):
        obs_b = clients.loc[codes % nb == b, bf.OBS_COLS]
        tx_b = txns.loc[txns[OBS].isin(obs_b[OBS]), tcols]
        df = tx_b.merge(obs_b, on=OBS, how="inner")
        df = bf.normalise_snapshot(df)
        res = bf.build_all(df)
        parts.append(res["obs_features"])
        print(f"  pay-day features batch {b + 1}/{nb}: {len(obs_b):,} observations")
    f = pd.concat(parts, ignore_index=True)
    f[OBS] = f[OBS].astype(str)
    return f.set_index(OBS)


# ----------------------------------------------------------------------------------------------
# 5. static block
# ----------------------------------------------------------------------------------------------
def static_block(clients, rare_share=RARE_SHARE):
    """Point-in-time attributes known at M0: numeric columns as they are, low-cardinality text/categorical one-hot encoded
    (rare levels pooled). Outcome, label, ids, dates and payment-period columns are excluded."""
    c = clients.set_index(OBS)
    cols = [x for x in c.columns if x not in NEVER_FEATURES]
    num, cat = [], []
    for x in cols:
        s = c[x]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            num.append(x)
        elif s.nunique(dropna=True) <= STATIC_CATEGORICAL_MAX_LEVELS:
            cat.append(x)
    X = c[num].astype("float64").replace([np.inf, -np.inf], np.nan)
    for x in cat:
        s = c[x].astype("string").fillna("missing")
        share = s.value_counts(normalize=True)
        s = s.where(~s.isin(share[share < rare_share].index), "other")
        X = X.join(pd.get_dummies(s, prefix=x, dtype="int8"))
    X = X.loc[:, X.nunique(dropna=False) > 1]            # constants carry nothing
    return X


def encode_time_block(F):
    """Text columns from build_features (pay_frequency, income_type_p0..p2 ...) -> ordered / one-hot numbers."""
    F = F.copy()
    prio = {t: i for i, t in enumerate(["none"] + bf.TYPE_PRIORITY[::-1])}      # none=0 ... salary=highest
    for c in [x for x in F.columns if x.startswith("income_type")]:
        F[c + "_rank"] = F[c].map(prio).astype("float64")
    obj = [c for c in F.columns if not pd.api.types.is_numeric_dtype(F[c]) and not pd.api.types.is_bool_dtype(F[c])]
    dropped = []
    for c in obj:
        if c.startswith("income_type") or F[c].nunique() > MAX_ONEHOT_LEVELS:   # ranked above / free text (e.g. "Salary, Pension")
            F = F.drop(columns=c)
            dropped.append(c)
        else:
            F = F.join(pd.get_dummies(F.pop(c).astype("string").fillna("missing"), prefix=c, dtype="int8"))
    F.attrs["dropped_text"] = dropped
    return F.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------------------------
# 7. QA
# ----------------------------------------------------------------------------------------------
def leak_tripwire(X, y, train_ids, thr=0.85, max_rows=30_000, seed=0):
    """Univariate AUC on train rows. A single feature with AUC (or 1-AUC) above `thr` is almost certainly leaking the
    outcome or the label: look before using it."""
    from sklearn.metrics import roc_auc_score
    ids = pd.Index(train_ids)
    if len(ids) > max_rows:
        ids = ids[np.random.default_rng(seed).choice(len(ids), max_rows, replace=False)]
    Xt, yt = X.loc[ids], y.loc[ids].astype(int)
    if yt.nunique() < 2:
        return pd.Series(dtype=float)
    out = {}
    for c in X.columns:
        v = Xt[c]
        if v.nunique() < 2:
            continue
        v = v.fillna(v.median())
        a = roc_auc_score(yt, v)
        out[c] = max(a, 1 - a)
    s = pd.Series(out).sort_values(ascending=False)
    return s[s >= thr]


def customer_split(ids, customers, test_size=0.2, seed=0):
    from sklearn.model_selection import GroupShuffleSplit
    g = customers.reindex(ids).astype(str).to_numpy()
    a, b = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(ids, groups=g))
    sp = pd.Series("train", index=ids, name="split")
    sp.iloc[b] = "test"
    return sp


# ----------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------
def build_dataset(path, out_dir="dataset", as_of=0, sample_customers=None, drop_excluded=True, batch_customers=20_000,
                  test_size=0.2, seed=0, with_pay_features=True, verbose=True):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rep = []
    log(f"1. Load and clean ({path})", rep)
    clients, txns, report = cd.load_and_clean(path, verbose=verbose)

    if sample_customers:
        cust = clients[ID].astype(str).unique()
        keep = set(np.random.default_rng(seed).choice(cust, min(sample_customers, len(cust)), replace=False))
        clients = clients[clients[ID].astype(str).isin(keep)]
        txns = txns[txns[OBS].astype(str).isin(set(clients[OBS].astype(str)))]
        log(f"   sampled {len(keep):,} customers", rep)

    log("\n2. Checks and cleaning", rep)
    clients, txns = clean_tables(clients, txns, drop_excluded, rep)
    audit = label_audit(clients, rep)

    log("\n3. Running-state features (features_transactions)", rep)
    run = ft.build_running_features(clients, txns, as_of=as_of, batch_customers=batch_customers, verbose=False)
    run.index = _text_index(run.index)
    run = run.drop(columns=[c for c in (ID, "customer") if c in run.columns])

    if with_pay_features:
        log("\n4. Pay-day / pay-cycle features (build_features)", rep)
        pay = pay_features(clients, txns, batch_customers, rep)
        pay = pay.drop(columns=[c for c in pay.columns if c in NEVER_FEATURES])
        pay = pay[[c for c in pay.columns if c not in run.columns]]        # keep the running version where names clash
        # static columns from build_features (obs_seq, product_group, ...) are already in the static block
        pay = pay.drop(columns=[c for c in ("obs_seq", "product_group", "m0_start_cd_group", "new_to_arrears_flag",
                                            "crccreditlimit") if c in pay.columns])
        time_block = run.join(encode_time_block(pay), how="left")
    else:
        time_block = run
    log(f"   time-aware block: {time_block.shape[1]} columns", rep)

    log("\n5. Static (point-in-time) block", rep)
    static = static_block(clients)
    log(f"   static block: {static.shape[1]} columns", rep)

    log("\n6. Assemble", rep)
    idx = _text_index(clients[OBS])
    lab = pd.DataFrame({ID: clients[ID].astype(str).to_numpy()}, index=idx)
    lab[TARGET] = clients[TARGET].astype(int).to_numpy()
    if RULE in clients:
        lab[RULE] = rule_number(clients[RULE]).to_numpy()
        lab["target_from_rule"] = lab[RULE].map(RULE_TO_TARGET).astype("Int64")
    for c in ("exclude_flag", M0, "payment_date"):
        if c in clients:
            lab[c] = clients[c].to_numpy()
    static = static.reindex(idx)
    time_block = time_block.reindex(idx)
    X_all = static.join(time_block.rename(columns=lambda c: c if c not in static.columns else c + "_t"))
    for name, X in (("static", static), ("time", time_block), ("all", X_all)):
        assert X.index.equals(idx) and not X.index.duplicated().any(), name
        assert not any(c in X.columns for c in NEVER_FEATURES), f"{name} block contains a never-feature"

    split = customer_split(idx, lab[ID], test_size, seed)
    tr, te = split.index[split == "train"], split.index[split == "test"]
    assert not set(lab.loc[tr, ID]) & set(lab.loc[te, ID]), "customers overlap between train and test"
    log(f"   {len(idx):,} observations | train {len(tr):,} | test {len(te):,} | "
        f"{lab[ID].nunique():,} customers | X_static {static.shape[1]} | X_time {time_block.shape[1]} | X_all {X_all.shape[1]}", rep)

    log("\n7. QA", rep)
    miss = X_all.isna().mean().sort_values(ascending=False)
    log(f"   columns with missing values: {(miss > 0).sum()} (worst: " +
        ", ".join(f"{k} {v:.0%}" for k, v in miss.head(5).items()) + ")", rep)
    const = [c for c in X_all.columns if X_all[c].nunique(dropna=True) < 2]
    log(f"   constant columns: {len(const)}", rep)
    leaks = leak_tripwire(X_all, lab[TARGET], tr)
    log("   leak tripwire (single-feature AUC >= 0.85 on train): " + ("none" if leaks.empty else
        "\n" + leaks.round(3).to_string()), rep)
    for name, X in (("X_static", static), ("X_time", time_block), ("X_all", X_all)):
        a = leak_tripwire(X, lab[TARGET], tr, thr=0.0)
        if len(a):
            log(f"   best single-feature AUC in {name}: {a.iloc[0]:.3f} ({a.index[0]})", rep)

    static.to_parquet(os.path.join(out_dir, "X_static.parquet"))
    time_block.to_parquet(os.path.join(out_dir, "X_time.parquet"))
    X_all.to_parquet(os.path.join(out_dir, "X_all.parquet"))
    lab.to_parquet(os.path.join(out_dir, "labels.parquet"))
    split.reset_index().assign(**{ID: lab[ID].to_numpy()}).to_parquet(os.path.join(out_dir, "split.parquet"), index=False)
    manifest = {"source": path, "as_of_days": as_of, "sample_customers": sample_customers, "seed": seed, "test_size": test_size,
                "observations": int(len(idx)), "customers": int(lab[ID].nunique()), "train": int(len(tr)), "test": int(len(te)),
                "n_static": int(static.shape[1]), "n_time": int(time_block.shape[1]), "n_all": int(X_all.shape[1]),
                "target_prevalence": float(lab[TARGET].mean()), "label_audit": audit,
                "leak_suspects": leaks.round(3).to_dict(), "never_features": NEVER_FEATURES,
                "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(out_dir, "qa_report.txt"), "w") as f:
        f.write("\n".join(rep))
    print(f"\nSaved to {out_dir}/  ({time.time() - t0:,.0f}s)")
    return load_dataset(out_dir)


# ----------------------------------------------------------------------------------------------
# Use
# ----------------------------------------------------------------------------------------------
def load_dataset(out_dir="dataset"):
    """dict(X_static, X_time, X_all, labels, y, groups, train, test). All indexed by obs_id (text);
    y = target_value (Series), groups = customer id (Series), train/test = obs_id Index lists."""
    p = lambda n: os.path.join(out_dir, n)
    lab = pd.read_parquet(p("labels.parquet"))
    sp = pd.read_parquet(p("split.parquet")).set_index(OBS)["split"]
    D = {n: pd.read_parquet(p(f"{n}.parquet")) for n in ("X_static", "X_time", "X_all")}
    D.update(labels=lab, y=lab[TARGET].astype(int), groups=lab[ID].astype(str),
             train=sp.index[sp == "train"], test=sp.index[sp == "test"])
    return D


def impute_train_only(X, train_ids, fill_zero_cols=()):
    """Median fill fitted on the TRAIN rows only, applied to all rows (no test information)."""
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.loc[pd.Index(train_ids)].median()
    out = X.fillna(med).fillna(0.0)
    for c in fill_zero_cols:
        out[c] = X[c].fillna(0.0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="snapshot/snapshot.parquet")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--as-of", type=int, default=0, help="days before M0 at which the running state is taken (0 = M0)")
    ap.add_argument("--sample-customers", type=int, default=None, help="random customers only (quick test run)")
    ap.add_argument("--keep-excluded", action="store_true", help="do not drop exclude_flag == 1")
    ap.add_argument("--batch-customers", type=int, default=20_000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pay-features", action="store_true", help="skip build_features (running-state block only)")
    a, _ = ap.parse_known_args()
    build_dataset(a.path, a.out_dir, a.as_of, a.sample_customers, not a.keep_excluded, a.batch_customers, a.test_size,
                  a.seed, not a.no_pay_features)
