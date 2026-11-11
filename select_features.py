"""
select_features.py - shrink the ~141 transaction features to a small, stable, explainable set.

For the wide table used in the clustering / kNN notebook: X has ONE ROW PER obs_id (index = obs_id), y is the target
indexed by obs_id, customers is the customer id per obs_id (only used to keep a customer's rows together).

Everything that looks at the target is fitted on TRAIN rows only. Make the split once, keep the test rows untouched
until the very last comparison, and tune k / the feature set with grouped CV inside the train rows.

    from select_features import (split, prune, stable_groups, compare_sets)
    tr, te = split(y, customers)                                   # obs_id lists, customer-grouped
    pr   = prune(X, tr, y)                                         # 1. aliases, constants, correlated families
    print(pr["table"])                                             #    what was dropped and why
    st   = stable_groups(X, y, customers, tr, pr["groups"])        # 2. which families matter, repeatedly, on unseen customers
    print(st["table"])
    sets = {"all": list(X.columns), "pruned": pr["keep"], "selected": st["keep"]}
    print(compare_sets(X, y, customers, tr, te, sets))             # 3. same split, same model, fewer inputs

Repeat step 1-2 at each cut-off (as_of 60 / 30 / 0) and keep the families that are selected at all of them.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split(y, customers, test_size=0.2, seed=0):
    """Customer-grouped train / test obs_id lists."""
    idx = y.index
    g = customers.reindex(idx).astype(str).to_numpy()
    a, b = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, groups=g))
    return idx[a], idx[b]


def _clean(X):
    return X.replace([np.inf, -np.inf], np.nan)


def _uni_auc(x, y):
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return 0.5
    return roc_auc_score(y, x)


def prune(X, train_ids, y=None, corr_thr=0.9, dup_thr=0.999, const_share=0.99, max_rows=20_000, seed=0,
          prefer=("_slog", "_share"), verbose=True):
    """
    Unsupervised pruning, judged on TRAIN rows. Four steps, each reported in `table`:
      constant       one value on >= const_share of the rows (or all missing)
      alias          |Spearman| >= dup_thr with another column (e.g. fees_share_of_spend == grp_bank_fees_interest_share)
      correlated     the remaining columns are grouped into families by hierarchical clustering on 1-|Spearman|
                     (cut at 1-corr_thr); ONE representative per family is kept
    Representative = the member with the strongest univariate train AUC when y is given (ties and the no-y case:
    columns containing a `prefer` tag first, then the shorter name). Raw money columns lose to their _slog twin.
    Returns dict(keep=[cols], groups={family: [members]}, table=DataFrame).
    """
    Xt = _clean(X.loc[pd.Index(train_ids)])
    if len(Xt) > max_rows:
        Xt = Xt.sample(max_rows, random_state=seed)
    ytr = None if y is None else y.reindex(Xt.index).astype(int)
    rows = []
    cols = []
    for c in X.columns:
        v = Xt[c].dropna()
        if v.empty or v.value_counts(normalize=True).iloc[0] >= const_share:
            rows.append((c, "constant", "", np.nan))
        else:
            cols.append(c)

    R = Xt[cols].rank().corr().abs().fillna(0.0).to_numpy().copy()
    np.fill_diagonal(R, 1.0)

    def pref_key(c, i):
        tag = 0 if any(t in c for t in prefer) else 1
        auc = -abs(_uni_auc(Xt[c], ytr) - 0.5) if ytr is not None else 0.0
        return (round(auc, 3), tag, len(c), c)

    # step 2: aliases
    alive = list(range(len(cols)))
    dropped = set()
    for i in sorted(alive, key=lambda i: pref_key(cols[i], i)):
        if i in dropped:
            continue
        for j in alive:
            if j != i and j not in dropped and R[i, j] >= dup_thr:
                dropped.add(j)
                rows.append((cols[j], "alias", cols[i], float(R[i, j])))
    alive = [i for i in alive if i not in dropped]

    # step 3: families
    groups, keep = {}, []
    if len(alive) > 1:
        D = 1.0 - R[np.ix_(alive, alive)]
        np.fill_diagonal(D, 0.0)
        lab = fcluster(linkage(squareform((D + D.T) / 2, checks=False), "average"), t=1 - corr_thr, criterion="distance")
    else:
        lab = np.ones(len(alive), int)
    fam = {}
    for pos, i in enumerate(alive):
        fam.setdefault(lab[pos], []).append(i)
    for members in fam.values():
        members = sorted(members, key=lambda i: pref_key(cols[i], i))
        rep = cols[members[0]]
        groups[rep] = [cols[i] for i in members]
        keep.append(rep)
        for i in members[1:]:
            rows.append((cols[i], "correlated", rep, float(R[members[0], i])))
    table = pd.DataFrame(rows, columns=["feature", "reason", "kept_instead", "abs_spearman"])
    table = pd.concat([table, pd.DataFrame({"feature": keep, "reason": "kept", "kept_instead": "", "abs_spearman": np.nan})],
                      ignore_index=True)
    if verbose:
        vc = table.reason.value_counts()
        print(f"prune: {X.shape[1]} -> {len(keep)} inputs  (constant {vc.get('constant', 0)}, alias {vc.get('alias', 0)}, "
              f"correlated {vc.get('correlated', 0)}); {sum(len(v) > 1 for v in groups.values())} families with >1 member")
    return {"keep": keep, "groups": groups, "table": table}


def make_model(kind="knn", k=7):
    est = KNeighborsClassifier(n_neighbors=k, weights="distance") if kind == "knn" else LogisticRegression(C=0.3, max_iter=2000)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def _score(model, X, y):
    return balanced_accuracy_score(y, model.predict(X))


def group_permutation_importance(model, Xv, yv, groups, n_repeats=3, seed=0):
    """Fall in balanced accuracy when a whole FAMILY of columns is shuffled together (rows moved jointly)."""
    rng = np.random.default_rng(seed)
    base = _score(model, Xv, yv)
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            Xp[cols] = Xv[cols].to_numpy()[rng.permutation(len(Xv))]
            drops.append(base - _score(model, Xp, yv))
        out[name] = np.mean(drops)
    return pd.Series(out)


def stable_groups(X, y, customers, train_ids, groups, kind="knn", k=25, n_splits=5, n_repeats=3, min_frac=0.8, min_t=3.0, seed=0,
                  verbose=True):
    """
    Which families keep mattering on customers the model has not seen? Grouped K-fold INSIDE the train rows; each
    family's importance is measured on the validation fold only. A family is kept when its mean importance is > 0 and
    it is positive in >= min_frac of the folds and its t-statistic (mean / (sd / sqrt(folds))) is >= min_t - the t rule
    is what keeps pure-noise columns out when 140 families are tested at once. `groups` is prune()['groups'] (or any {name: [columns]}).
    `kind` is the model used to judge ("knn" like the notebook, or "logit"); using both and keeping the union is safer.
    Returns dict(keep=[representative columns], keep_groups={name: members}, table=DataFrame).
    """
    ids = pd.Index(train_ids)
    Xt, yt = _clean(X.loc[ids]), y.loc[ids].astype(int)
    g = customers.reindex(ids).astype(str).to_numpy()
    reps = list(groups)
    imps = []
    for f, (a, b) in enumerate(GroupKFold(n_splits).split(Xt, yt, groups=g)):
        m = make_model(kind, min(k, len(a) - 1)).fit(Xt.iloc[a][reps], yt.iloc[a])
        imps.append(group_permutation_importance(m, Xt.iloc[b][reps], yt.iloc[b], {r: [r] for r in reps},
                                                 n_repeats, seed + f))
    I = pd.concat(imps, axis=1)
    table = pd.DataFrame({"mean": I.mean(axis=1), "sd": I.std(axis=1), "frac_folds_positive": (I > 0).mean(axis=1),
                          "family_size": [len(groups[r]) for r in I.index]}).sort_values("mean", ascending=False)
    table["t"] = table["mean"] / (table["sd"] / np.sqrt(n_splits)).replace(0, np.nan)
    table["keep"] = (table["mean"] > 0) & (table.frac_folds_positive >= min_frac) & (table.t >= min_t)
    keep = table.index[table.keep].tolist()
    if verbose:
        print(f"stable_groups ({kind}): {len(reps)} families -> {len(keep)} kept "
              f"(positive in >= {min_frac:.0%} of {n_splits} folds and t >= {min_t})")
    return {"keep": keep, "keep_groups": {r: groups[r] for r in keep}, "table": table}


def compare_sets(X, y, customers, train_ids, test_ids, sets, kinds=("knn", "logit"), k=7, n_splits=5):
    """
    Same split, same models, different input sets. cv_* = grouped CV inside train (use this to choose);
    test_* = the untouched test rows (report this once, for the set you chose).
    """
    tr, te = pd.Index(train_ids), pd.Index(test_ids)
    Xc = _clean(X)
    gtr = customers.reindex(tr).astype(str).to_numpy()
    rows = []
    for name, cols in sets.items():
        for kind in kinds:
            cv = []
            for a, b in GroupKFold(n_splits).split(tr, groups=gtr):
                m = make_model(kind, min(k, len(a) - 1)).fit(Xc.loc[tr[a], cols], y.loc[tr[a]].astype(int))
                cv.append(_score(m, Xc.loc[tr[b], cols], y.loc[tr[b]].astype(int)))
            m = make_model(kind, k).fit(Xc.loc[tr, cols], y.loc[tr].astype(int))
            p = m.predict_proba(Xc.loc[te, cols])[:, 1]
            rows.append({"inputs": name, "n_inputs": len(cols), "model": kind, "cv_bal_acc": np.mean(cv),
                         "cv_sd": np.std(cv), "test_bal_acc": _score(m, Xc.loc[te, cols], y.loc[te].astype(int)),
                         "test_auc": roc_auc_score(y.loc[te].astype(int), p)})
    return pd.DataFrame(rows)
