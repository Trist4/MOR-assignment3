"""
cluster_states.py - cluster search on TRANSACTION-LEVEL running states (built by features_running.py).

Idea: after every transaction a customer is in a "state" (balance, how much has come in and gone out so far, how long
since pay, spending mix ...). Cluster those states across all customers to find a small set of financial regimes
(e.g. "paid and comfortable", "living on the overdraft", "cash-out after pay day"). Each observation is then a
PATH through those regimes, and the path is what becomes the feature set: where it ended, how long it spent in each
regime, how often it switched, where it was 30 / 60 days before M0.

    from clean_data import load_and_clean
    from cluster_states import (sample_states, search_states, pick_best, fit_final, state_features,
                                profile_states, target_by_state, evaluate_states)

    clients, txns, _ = load_and_clean("snapshot/snapshot.parquet")
    sample = sample_states(clients, txns)                       # states of ~150k transactions, capped per observation
    results, models = search_states(sample, ks=range(3, 11))    # algorithm x k search, scored without the target
    best = pick_best(results)                                   # e.g. ('kmeans', 6)
    scaler, model = models[best]
    feats = state_features(clients, txns, scaler, model, as_of=(0, 30, 60))   # one row per observation
    print(profile_states(sample, scaler, model))                # what each regime looks like
    print(target_by_state(feats, clients))                      # target rate by final regime
    print(evaluate_states(feats, clients))                      # grouped-CV AUC: static vs state features

Command line:
    python cluster_states.py --path snapshot/snapshot.parquet --out-dir state_clusters --ks 3 10

Search (all unsupervised; the target is never used to pick the clustering)
  algorithms   kmeans (MiniBatchKMeans), gmm (diagonal Gaussian mixture), knn_spectral (spectral clustering on a
               k-nearest-neighbour graph of a sub-sample, new points labelled by their nearest neighbours)
  scores       silhouette, Davies-Bouldin, smallest cluster share, and STABILITY: fit on two disjoint halves of the
               customers, label a common hold-out sample with both, compare with the adjusted Rand index
  selection    rank-average of stability, silhouette and Davies-Bouldin; clusters below MIN_CLUSTER_SHARE are rejected

The first MIN_TXN transactions of an observation are a warm-up (running ratios are noise with few transactions):
they are not clustered and are labelled -1.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

import features_monthly as F
import features_transactions as R

ID, M0, AGE = F.ID, F.M0, F.AGE_COL
KEY = F.KEY
signed_log1p = F.signed_log1p

# ======================================================================================================
# CONFIG
# ======================================================================================================
MIN_TXN = 10                    # states before this many transactions in the observation are warm-up (label -1)
MIN_CLUSTER_SHARE = 0.02        # a clustering with a smaller cluster than this is rejected
SPECTRAL_MAX = 4000             # sub-sample size for the k-NN-graph (spectral) method
KNN_NEIGHBOURS = 15
# state features, grouped in blocks; each block gets the same total weight so the 5 mix shares do not outvote the rest
STATE_BLOCKS = {
    "balance":  ["balance_slog", "min_balance_slog", "neg_balance_share"],
    "cashflow": ["outflow_to_inflow", "net_to_inflow", "spend_per_day_slog", "income_per_day_slog"],
    "activity": ["txn_per_day_slog", "days_since_income", "uif_seen"],
    "mix":      ["essentials_share", "nonessential_share", "gambling_share", "cash_share", "fees_share"],
}
STATE_COLS = [c for cols in STATE_BLOCKS.values() for c in cols]


# ======================================================================================================
# 1. State vector (one row per transaction) and sampling
# ======================================================================================================
def state_vector(t):
    """Scale-free state features from a running-table frame (features_running.running_table)."""
    n = t["run_n_txn"].to_numpy("float64")
    cov = t["days_covered"].clip(lower=1).to_numpy("float64")
    infl, outf = t["run_inflow"].to_numpy("float64"), t["run_outflow"].to_numpy("float64")
    net, spend = t["run_net"].to_numpy("float64"), t["run_spend"].to_numpy("float64")
    pos_in, pos_sp = infl > 0, spend > 0

    o2i = np.where(pos_in, outf / np.where(pos_in, infl, 1.0), np.where(outf > 0, 5.0, 1.0))
    n2i = np.where(pos_in, net / np.where(pos_in, infl, 1.0), np.where(outf > 0, -5.0, 0.0))
    share = lambda col: np.where(pos_sp, t[col].to_numpy("float64") / np.where(pos_sp, spend, 1.0), 0.0)
    dsi = t["days_since_last_income"].to_numpy("float64")
    dsi = np.where(np.isnan(dsi), cov, dsi)
    bal = t["balance"].to_numpy("float64")
    minb = t["run_min_balance"].to_numpy("float64")
    X = pd.DataFrame({
        "balance_slog": signed_log1p(bal),
        "min_balance_slog": signed_log1p(minb),
        "neg_balance_share": t["run_n_neg_balance"].to_numpy("float64") / np.maximum(n, 1),
        "outflow_to_inflow": np.clip(o2i, 0, 5),
        "net_to_inflow": np.clip(n2i, -5, 1),
        "spend_per_day_slog": signed_log1p(spend / cov),
        "income_per_day_slog": signed_log1p(t["run_income"].to_numpy("float64") / cov),
        "txn_per_day_slog": np.log1p(n / cov),
        "days_since_income": np.minimum(dsi, cov),
        "uif_seen": (t["run_uif"].to_numpy("float64") > 0).astype("float64"),
        "essentials_share": share("run_essentials"), "nonessential_share": share("run_nonessential"),
        "gambling_share": share("run_gambling"), "cash_share": share("run_cash"), "fees_share": share("run_fees"),
    }, index=t.index)
    return X[STATE_COLS].astype("float32")


def sample_states(clients, txns, n_target=150_000, max_per_obs=25, as_of=0, seed=0, collisions=None, verbose=True):
    """
    Sample transaction-level states for fitting. Each observation contributes at most `max_per_obs` random states
    (after warm-up), so long histories do not dominate; the pool is then thinned uniformly to `n_target`.
    Returns a frame with the observation key, days_before_m0 and the state columns.
    """
    rng = np.random.default_rng(seed)
    parts, seen = [], 0
    for t in R.running_table(clients, txns, by_group=False, as_of=as_of, verbose=False, collisions=collisions):
        seen += len(t)
        X = state_vector(t)
        keep = (t["run_n_txn"].to_numpy() >= MIN_TXN)
        gid = t.groupby(KEY, sort=False).ngroup().to_numpy()
        rank = pd.Series(rng.random(len(t))).where(keep).groupby(gid).rank(method="first").to_numpy()
        keep &= rank <= max_per_obs
        d = pd.concat([t.loc[keep, KEY + [AGE]].reset_index(drop=True), X[keep].reset_index(drop=True)], axis=1)
        parts.append(d)
    s = pd.concat(parts, ignore_index=True)
    if len(s) > n_target:
        s = s.sample(n_target, random_state=seed).reset_index(drop=True)
    if verbose:
        print(f"sample_states: {len(s):,} states from {seen:,} transactions "
              f"({s[KEY].drop_duplicates().shape[0]:,} observations, {s[ID].nunique():,} customers)")
    return s


# ======================================================================================================
# 2. Scaling (fit once on the sample, reused for every later assignment)
# ======================================================================================================
class StateScaler:
    """Winsorise -> median / IQR scale -> equal block weights -> PCA (keeps `var` of the variance)."""

    def __init__(self, var=0.90, clip=(0.01, 0.99)):
        self.var, self.clip = var, clip

    def fit(self, X):
        X = X[STATE_COLS].astype("float64")
        self.lo, self.hi = X.quantile(self.clip[0]), X.quantile(self.clip[1])
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median()
        iqr = Xc.quantile(.75) - Xc.quantile(.25)
        std = Xc.std()
        self.scale = iqr.where(iqr > 1e-9, std).where(lambda s: s > 1e-9, 1.0)
        self.w = pd.Series({c: 1 / np.sqrt(len(cols)) for cols in STATE_BLOCKS.values() for c in cols})[STATE_COLS]
        self.pca = PCA(self.var, random_state=0).fit(self._z(X))
        return self

    def _z(self, X):
        Xc = X[STATE_COLS].astype("float64").clip(self.lo, self.hi, axis=1)
        return ((Xc - self.med) / self.scale * self.w).to_numpy()

    def transform(self, X):
        return self.pca.transform(self._z(X)).astype("float32")


# ======================================================================================================
# 3. Clustering methods behind one interface: fit(Z) -> object with predict(Z)
# ======================================================================================================
class _KnnLabeller:
    """Labels from a spectral clustering of a sub-sample; new points get the majority label of their neighbours."""

    def __init__(self, Zs, labels):
        self.knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBOURS).fit(Zs, labels)
        self.k = len(np.unique(labels))

    def predict(self, Z):
        return self.knn.predict(Z)


def _fit(algo, k, Z, seed):
    if algo == "kmeans":
        return MiniBatchKMeans(k, n_init=5, batch_size=4096, random_state=seed).fit(Z)
    if algo == "gmm":
        return GaussianMixture(k, covariance_type="diag", n_init=2, reg_covar=1e-4, random_state=seed).fit(Z)
    if algo == "knn_spectral":
        rng = np.random.default_rng(seed)
        Zs = Z[rng.choice(len(Z), min(len(Z), SPECTRAL_MAX), replace=False)]
        lab = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=KNN_NEIGHBOURS, assign_labels="kmeans",
                                 random_state=seed, n_jobs=1).fit_predict(Zs)
        return _KnnLabeller(Zs, lab)
    raise ValueError(f"unknown algorithm {algo}")


def search_states(sample, ks=range(3, 11), algos=("kmeans", "gmm", "knn_spectral"), n_rep=3, eval_n=10_000,
                  seed=0, verbose=True):
    """
    Search algorithm x k. Returns (results table, {(algo, k): (scaler, model)}).
    stability_ari = mean adjusted Rand index between two fits on disjoint halves of the customers, both labelling
    the same hold-out states (1.0 = same partition every time).
    """
    rng = np.random.default_rng(seed)
    scaler = StateScaler().fit(sample)
    Z = scaler.transform(sample)
    cust = sample[ID].to_numpy()
    ev = rng.choice(len(Z), min(len(Z), eval_n), replace=False)
    Zev = Z[ev]
    uc = np.unique(cust)
    rows, models = [], {}
    if verbose:
        print(f"search: {len(Z):,} states -> {Z.shape[1]} components after PCA ({scaler.pca.explained_variance_ratio_.sum():.0%} variance)")
    for algo in algos:
        for k in ks:
            t0 = time.time()
            m = _fit(algo, k, Z, seed)
            lab = m.predict(Zev)
            share = np.bincount(lab, minlength=k) / len(lab)
            row = {"algo": algo, "k": k, "min_share": share.min(), "max_share": share.max()}
            if len(np.unique(lab)) > 1:
                row["silhouette"] = silhouette_score(Zev, lab, sample_size=min(5000, len(lab)), random_state=seed)
                row["davies_bouldin"] = davies_bouldin_score(Zev, lab)
                row["calinski_harabasz"] = calinski_harabasz_score(Zev, lab)
            aris = []
            for r in range(n_rep):                          # customer-level halves, never splitting a customer
                perm = np.random.default_rng(seed + 1 + r).permutation(uc)
                a, b = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
                ia, ib = np.isin(cust, list(a)), np.isin(cust, list(b))
                ma, mb = _fit(algo, k, Z[ia], seed + r), _fit(algo, k, Z[ib], seed + 100 + r)
                aris.append(adjusted_rand_score(ma.predict(Zev), mb.predict(Zev)))
            row["stability_ari"] = float(np.mean(aris))
            row["stability_sd"] = float(np.std(aris))
            row["seconds"] = time.time() - t0
            rows.append(row)
            models[(algo, k)] = (scaler, m)
            if verbose:
                print(f"  {algo:<13} k={k:<2} silhouette {row.get('silhouette', np.nan):.3f}  "
                      f"DB {row.get('davies_bouldin', np.nan):.2f}  stability {row['stability_ari']:.2f}  "
                      f"smallest {row['min_share']:.1%}  ({row['seconds']:.0f}s)")
    res = pd.DataFrame(rows)
    res["rejected"] = res["min_share"] < MIN_CLUSTER_SHARE
    ok = ~res["rejected"]
    rk = pd.DataFrame({"s": res["stability_ari"].where(ok).rank(ascending=False),
                       "sil": res["silhouette"].where(ok).rank(ascending=False),
                       "db": res["davies_bouldin"].where(ok).rank(ascending=True)})
    res["score"] = rk.mean(axis=1)                            # lower is better
    return res.sort_values(["rejected", "score"]).reset_index(drop=True), models


def pick_best(results):
    """(algo, k) with the best rank-average among the clusterings that were not rejected."""
    ok = results[~results["rejected"]]
    if ok.empty:
        raise ValueError("every clustering has a cluster smaller than MIN_CLUSTER_SHARE; lower k or the threshold")
    r = ok.iloc[0]
    return (r["algo"], int(r["k"]))


# ======================================================================================================
# 4. Profile of the regimes
# ======================================================================================================
def label_states(X, scaler, model):
    return model.predict(scaler.transform(X))


def profile_states(sample, scaler, model):
    """Regime sizes and the median of every state feature (the table to read when naming the regimes)."""
    lab = label_states(sample, scaler, model)
    prof = sample[STATE_COLS].groupby(lab).median()
    prof.insert(0, "share", pd.Series(lab).value_counts(normalize=True).sort_index())
    prof.index.name = "state"
    return prof


# ======================================================================================================
# 5. Assignment of every transaction state -> one row per observation
# ======================================================================================================
def _path_features(t, lab, as_of, k):
    """Dwell time and path features from one batch of the running table with its state labels."""
    t = t[KEY + [AGE]].copy()
    t["state"] = lab
    g = t.groupby(KEY, sort=False)
    nxt = g[AGE].shift(-1)
    t["dwell"] = (t[AGE] - nxt).fillna(t[AGE] - as_of).clip(lower=0)      # state lasts until the next transaction
    d = t.groupby(KEY + ["state"], sort=False)["dwell"].sum().unstack("state").reindex(columns=range(-1, k)).fillna(0.0)
    d.columns = ["state_dwell_warmup"] + [f"state_dwell_{j}" for j in range(k)]
    total = d.sum(axis=1).replace(0, np.nan)

    last = t.groupby(KEY, sort=False).tail(1).set_index(KEY)
    out = d.copy()
    for c in list(d.columns):
        out[c.replace("dwell", "share")] = d[c] / total
    out["state_final"] = last["state"]
    # consecutive run of the final state: walk back until the state changes
    run_break = (t["state"] != g["state"].shift()).astype(int)
    run_id = run_break.groupby([t[c] for c in KEY], sort=False).cumsum()
    final_run = run_id.groupby([t[c] for c in KEY], sort=False).transform("max")
    out["state_final_days"] = t.loc[run_id == final_run].groupby(KEY, sort=False)["dwell"].sum()
    real = t[t["state"] >= 0]
    changes = (real["state"] != real.groupby(KEY, sort=False)["state"].shift()) & real.groupby(KEY, sort=False)["state"].shift().notna()
    out["state_n_changes"] = changes.groupby([real[c] for c in KEY], sort=False).sum()
    out["state_n_changes"] = out["state_n_changes"].fillna(0)
    return out


def assign_states(clients, txns, scaler, model, as_of=0, batch_customers=20_000, collisions=None, verbose=True):
    """State path features for every observation at `as_of` days before M0 (see module doc). One row per observation (KEY)."""
    clients_r, idx = R._prep_clients(clients, collisions, verbose)
    k = _n_states(model)
    parts = []
    for t in R.running_table(clients_r, txns, batch_customers=batch_customers, by_group=False, as_of=as_of, verbose=False):
        X = state_vector(t)
        lab = np.where(t["run_n_txn"].to_numpy() >= MIN_TXN, label_states(X, scaler, model), -1)
        parts.append(_path_features(t, lab, as_of, k))
    out = pd.concat(parts).reindex(idx)
    dwell = [c for c in out.columns if c.startswith("state_dwell_")]
    out[dwell] = out[dwell].fillna(0.0)
    out["state_final"] = out["state_final"].fillna(-1).astype("int16")      # no transactions in the window: -1
    out["state_final_days"] = out["state_final_days"].fillna(0.0)
    out["state_n_changes"] = out["state_n_changes"].fillna(0)
    return out


def _n_states(model):
    return int(getattr(model, "n_clusters", None) or getattr(model, "n_components", None) or model.k)


def state_features(clients, txns, scaler, model, as_of=(0, 30, 60), **kw):
    """
    Observation-level features from the state paths. as_of=0 gives the full path to M0; earlier cut-offs give the
    regime that observation was in 30 / 60 days before M0 (columns suffixed _at30 ...), so the move between regimes
    is visible without monthly buckets.
    """
    frames = []
    for a in as_of:
        f = assign_states(clients, txns, scaler, model, as_of=a, **kw)
        if a != as_of[0]:
            f = f[["state_final", "state_final_days"]]
        f.columns = [f"{c}_at{a}" if (a != 0) else c for c in f.columns]
        frames.append(f)
    out = pd.concat(frames, axis=1)
    if 30 in as_of:                                      # regime change over the last 30 days
        out["state_changed_last30"] = (out["state_final"] != out["state_final_at30"]).astype("int8")
    return out


# ======================================================================================================
# 6. Does the path explain the target? (reported, never used to choose the clustering)
# ======================================================================================================
def _wilson(p, n, z=1.96):
    n = np.maximum(n, 1)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def target_by_state(feats, clients, col="state_final", target="target_value", collisions=None, segment=None):
    """
    Target rate, Wilson 95% interval and lift for each regime of `col` (default: the regime at M0).
    segment: a clients column (e.g. "m0_start_cd_group", "product_group", "new_to_arrears_flag") to break the table
    down by; lift is then relative to the target rate of that segment, so the regime effect is not just the
    segment effect (CD stage, product and new-to-arrears already differ in risk).
    """
    c = F.resolve_observations(clients, collisions, verbose=False).set_index(KEY)
    d = pd.DataFrame({"state": feats[col], "y": c[target].reindex(feats.index).astype(float)})
    by = ["state"]
    if segment:
        d[segment] = c[segment].reindex(feats.index).astype(str)
        by = [segment, "state"]
    d = d.dropna()
    g = d.groupby(by)["y"].agg(n="size", rate="mean")
    g["ci_lo"], g["ci_hi"] = _wilson(g["rate"].to_numpy(), g["n"].to_numpy())
    base = d.groupby(segment)["y"].transform("mean") if segment else d["y"].mean()
    g["lift"] = g["rate"] / (d.assign(b=base).groupby(by)["b"].first() if segment else base)
    tot = g.groupby(level=0)["n"].transform("sum") if segment else g["n"].sum()
    g["share_of_segment" if segment else "share_of_obs"] = g["n"] / tot
    return g


def evaluate_states(feats, clients, target="target_value", n_splits=5, seed=0, collisions=None):
    """
    Grouped-by-customer CV AUC of: static fields only, state-path features only, and both.
    Logistic regression (scaled) and gradient boosting. The static baseline uses only fields known at M0.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    c = F.resolve_observations(clients, collisions, verbose=False).set_index(KEY)
    c = c.reindex(feats.index)
    y = c[target].to_numpy().astype(int)
    groups = feats.index.get_level_values(0).to_numpy()
    static = F.static_fields(clients, collisions).reindex(feats.index)     # incl. CD stage, product, new-to-arrears
    cat_cols = [x for x in feats.columns if re.fullmatch(r"state_final(_at\d+)?", x)]     # regime ids -> one-hot
    num = feats.drop(columns=cat_cols)
    onehot = pd.concat([pd.get_dummies(feats[x].astype(int), prefix=x, dtype=float) for x in cat_cols], axis=1)
    state = pd.concat([num.astype("float64"), onehot], axis=1)
    sets = {"static fields": static, "state paths": state, "static + state paths": pd.concat([static, state], axis=1)}
    models = {"logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5)),
              "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, random_state=seed)}
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    rows = []
    for sname, X in sets.items():
        X = X.fillna(X.median()).fillna(0.0).to_numpy()
        for mname, mk in models.items():
            aucs = []
            for tr, te in cv.split(X, y, groups):
                m = mk().fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            rows.append({"features": sname, "model": mname, "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs)})
    return pd.DataFrame(rows)


# ======================================================================================================
# 7. Final fit on the full sample and CLI
# ======================================================================================================
def fit_final(sample, algo, k, seed=0):
    """Fit one chosen clustering on the whole sample. Returns (scaler, model)."""
    scaler = StateScaler().fit(sample)
    return scaler, _fit(algo, k, scaler.transform(sample), seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="snapshot/snapshot.parquet")
    ap.add_argument("--out-dir", default="state_clusters")
    ap.add_argument("--ks", type=int, nargs=2, default=[3, 10], metavar=("MIN", "MAX"))
    ap.add_argument("--algos", nargs="+", default=["kmeans", "gmm", "knn_spectral"])
    ap.add_argument("--n-sample", type=int, default=150_000)
    ap.add_argument("--max-per-obs", type=int, default=25)
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--as-of", type=int, nargs="+", default=[0, 30, 60])
    ap.add_argument("--algo", default=None, help="skip selection: use this algorithm ...")
    ap.add_argument("--k", type=int, default=None, help="... and this number of clusters")
    ap.add_argument("--collisions", default=None, help="keep_largest | drop_month | drop_customer | error")
    args, _ = ap.parse_known_args()

    import joblib
    from clean_data import load_and_clean
    os.makedirs(args.out_dir, exist_ok=True)
    clients, txns, _ = load_and_clean(args.path, verbose=False)
    sample = sample_states(clients, txns, args.n_sample, args.max_per_obs, collisions=args.collisions)
    if args.algo and args.k:
        scaler, model = fit_final(sample, args.algo, args.k)
        best = (args.algo, args.k)
    else:
        res, models = search_states(sample, range(args.ks[0], args.ks[1] + 1), tuple(args.algos), args.n_rep)
        res.to_csv(os.path.join(args.out_dir, "search_results.csv"), index=False)
        best = pick_best(res)
        scaler, model = models[best]
    print(f"\nchosen: {best[0]} with k={best[1]}")
    prof = profile_states(sample, scaler, model)
    prof.to_csv(os.path.join(args.out_dir, "state_profiles.csv"))
    print("\nstate profiles (median of each feature):\n", prof.round(2).to_string())
    feats = state_features(clients, txns, scaler, model, tuple(args.as_of), collisions=args.collisions)
    feats.reset_index().to_parquet(os.path.join(args.out_dir, "state_features.parquet"), index=False)
    joblib.dump({"scaler": scaler, "model": model, "algo": best[0], "k": best[1]}, os.path.join(args.out_dir, "state_model.joblib"))
    if "target_value" in clients:
        print("\ntarget rate by regime at M0:\n", target_by_state(feats, clients, collisions=args.collisions).round(3).to_string())
        print("\ngrouped-CV AUC:\n", evaluate_states(feats, clients, collisions=args.collisions).round(3).to_string(index=False))
    print(f"\nSaved to {args.out_dir}/")
