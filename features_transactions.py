"""
features_transactions.py - RUNNING-STATE feature engineering (an alternative to the monthly blocks in features.py).

Instead of cutting the window into 30-day months, every observation's transactions are replayed in time order and
a running state is kept: "how much has come in / gone out so far", "what is my balance now", "how long since I was
last paid". The state after the last transaction is the CURRENT state at M0 (or at any earlier cut-off: as_of).

    from clean_data import load_and_clean
    from features_transactions import build_running_features, running_table, build_running_snapshots

    clients, txns, report = load_and_clean("snapshot/snapshot.parquet")

    state = build_running_features(clients, txns)                         # 1 row per observation: state at M0
    snaps = build_running_snapshots(clients, txns, as_of=(60, 30, 0))     # state at 60, 30 and 0 days before M0
    for tbl in running_table(clients, txns):                              # 1 row per transaction (sequence format)
        ...                                                               # (batched generator: memory safe)

Three outputs, three uses
  build_running_features   one row per observation (cifwithcheckdigit, obs_id): the CURRENT state. Use for k-NN / SOM /
                           trees, exactly like the monthly features.
  build_running_snapshots  the same state at several cut-offs, stacked long (extra index level as_of_days). Gives
                           time-awareness (state then vs now) without monthly buckets, and extra training rows.
  running_table            one row per transaction with the running state AFTER that transaction. Use for sequence
                           models, plots of a customer's trajectory, or to build your own windows.

Category groups, masks, and configuration are defined locally in this module, independently of features_monthly.py. Window: the last WINDOW_DAYS days before M0, 'pre' rows only (no outcome leakage).

Columns (all prefixed run_ = running totals since the start of the window up to the as-of point)
  Flows           run_inflow, run_outflow, run_net, run_n_txn
  Headline        run_spend, run_income, run_essentials, run_nonessential, run_gambling, run_cash, run_fees,
                  run_live_better_in, run_uif, run_n_purchases, run_n_gambling, run_n_fees, run_n_insufficient_fee,
                  run_n_bounced, run_n_cash
  Groups          run_grp_<outflow group>, run_in_<inflow group>
  Balance         balance (after the latest txn), run_min_balance, run_max_balance, run_n_neg_balance
  Recency         days_since_last_income, days_since_last_uif, days_since_last_txn, no_income_seen, unemployed_now
  Coverage        days_covered (days of history in the extract, up to the as-of point)
  Rates / ratios  *_per_day, outflow_to_inflow, spend_to_income, net_to_inflow, balance_runway_days,
                  <headline>_share_of_spend, grp_<group>_share, committed_share_of_spend, spend_group_entropy
"""
import argparse
import os

import numpy as np
import pandas as pd

ID, OBS, M0 = "cifwithcheckdigit", "obs_id", "m0_monthenddate"
KEY = [OBS]
TXN_TYPE, AMT, BAL, SPEND, CAT = "trantypedesc", "amount", "balance", "spendgroup", "categoryname"

# ======================================================================================================
# CONFIG
# ======================================================================================================
MONTH_LEN = 30                  # days per "month" block
N_MONTHS = 3                    # number of blocks (the extract holds ~98 days, so 3 x 30 is safe)
AGE_COL = "days_before_m0"      # days from the transaction to the reference date (>= 0 = before it)

# tx_period looks like a pre / post flag. TODO: confirm what 'post' means. Rows after the payment date may already
# reflect the outcome, so only 'pre' rows are used. Set PRE_COL = None to switch the filter off.
PRE_COL, PRE_VALUE = "tx_period", "pre"

# CardAuthorisationTransaction and CardTransaction look like the same purchases recorded twice.
# TODO: confirm, then e.g. DROP_SPENDGROUPS = ["CardAuthorisationTransaction"].
DROP_SPENDGROUPS = []

# ------------------------------------------------------------------------------------------------------
# 1. CATEGORY -> GROUP MAPPINGS  (DRAFT: review with check_mappings() / mapping_table())
# ------------------------------------------------------------------------------------------------------
# Money OUT (amount < 0)
OUTFLOW_GROUPS = {
    # ---- everyday needs
    "food_groceries":     ["Groceries", "Other Food"],
    "housing_utilities":  ["Rent", "Municipal Bill", "Levies", "Electricity", "Water", "Gas"],
    "communication":      ["Cellphone", "Internet", "Telephone", "Other Communication"],
    "transport":          ["Fuel", "Public Transport", "Parking", "Tolls", "Other Transport", "Vehicle Maintenance",
                           "Licence", "Vehicle Tracking"],
    "insurance":          ["Funeral Cover", "Life Insurance", "Other Insurance", "Home Insurance", "Vehicle Insurance"],
    "health":             ["Medical Aid", "Doctors & Therapists", "Pharmacy", "Other Medical"],
    "family_education":   ["Education", "Children & Dependants", "Child Support", "Books/Stationery"],
    # ---- discretionary
    "eating_out":         ["Takeaways", "Restaurants"],
    "alcohol_tobacco":    ["Alcohol", "Tobacco"],
    "gambling":           ["Betting/Lottery"],
    "personal_shopping":  ["Clothing & Shoes", "Personal Care", "Other Personal & Family", "Gadgets", "Online Store",
                           "Pets", "Gifts"],
    "leisure":            ["Sport & Hobbies", "Activities", "Holiday", "Movies", "Other Entertainment", "Going Out",
                           "Software/Games", "Digital Subscriptions", "Donations"],
    "household_upkeep":   ["Home Maintenance", "Home Improvements", "Garden", "Furniture & Appliances", "Housekeeping",
                           "Security", "Other Household"],
    # ---- credit, costs and money handling
    "debt_repayment":     ["Loan Payments", "Home Loan Payments", "Vehicle Payments", "Other Loans & Accounts",
                           "Store Account Payments", "Credit Card Payments", "Loans"],
    "bank_fees_interest": ["Fees", "Interest", "Commission"],
    "fines_legal_tax":    ["Traffic Fines", "Legal Fees", "Tax"],
    "cash_withdrawal":    ["Cash Withdrawal"],
    "p2p_payments":       ["Digital Payments", "Send Cash"],       # DRAFT: Digital Payments is large and ambiguous
    # ---- not spending: money put away or moved between own accounts
    "savings_investments": ["Savings", "Automatic Savings", "Investments", "Other Savings & Investments"],
    "transfers":          ["Transfer"],
    "unclassified":       ["Uncategorised"],                       # also receives every name not listed above
}

# Money IN (amount > 0)
INFLOW_GROUPS = {
    "income_employment":  ["Salary", "Wage", "Overtime", "Bonus", "Commission"],
    "income_other":       ["Other Income", "Allowance", "Pension", "Rental Income", "Investment Income", "Child Support"],
    "income_informal":    ["Payment Received", "Account Payment Received", "Cash Deposit"],   # DRAFT: may include salary
    "income_unemployment": ["UIF Received"],
    "borrowing":          ["Loans", "Credit Card Payments", "Other Loans & Accounts"],          # DRAFT: direction unclear
    "savings_drawdown":   ["Savings", "Investments", "Other Savings & Investments"],
    "refunds_windfalls":  ["Refunds", "Employee Refund", "Insurance Payout", "Banking Rewards", "Partner Rewards",
                           "Live Better Benefit", "Betting/Lottery", "Tax"],
    "transfers_in":       ["Transfer", "Automatic Savings"],
    "other_inflow":       ["Uncategorised", "Interest"],                                        # also the default
}

# ------------------------------------------------------------------------------------------------------
# 2. LENSES built from groups
# ------------------------------------------------------------------------------------------------------
NON_SPEND_GROUPS = ["savings_investments", "transfers"]       # outflows that are not spending
NON_SPEND_SPENDGROUPS = ["InternalTransfer"]
ESSENTIAL_GROUPS = ["food_groceries", "housing_utilities", "communication", "transport", "insurance", "health",
                    "family_education"]
DISCRETIONARY_GROUPS = ["eating_out", "alcohol_tobacco", "gambling", "personal_shopping", "leisure"]
COMMITTED_GROUPS = ["housing_utilities", "insurance", "debt_repayment", "communication"]   # fixed monthly obligations
INCOME_GROUPS = ["income_employment", "income_other"]          # "income" = employment + other recurring income
NON_INCOME_SPENDGROUPS = ["UnpaidOrBouncedDebitOrder", "InternalTransfer"]   # reversals / own transfers are not income
UIF_GROUP = "income_unemployment"                              # inflow here in a month => unemployed that month

# ------------------------------------------------------------------------------------------------------
# 3. Other rules
# ------------------------------------------------------------------------------------------------------
FEE_GROUPS = ["bank_fees_interest"]
FEE_SPENDGROUPS = ["Fee"]
FEE_TRAN_REGEX = r"Fee|Interest Charged|Penalty"
INSUFFICIENT_REGEX = r"Insufficient Funds"
BOUNCED_SPENDGROUPS = ["UnpaidOrBouncedDebitOrder"]            # reversals of failed debit orders (positive amounts)
CASH_GROUPS = ["cash_withdrawal"]
CASH_SPENDGROUPS = ["CashWithdrawal"]
LIVE_BETTER_TRAN_REGEX = r"Live Better"                        # outflow leg = money going into Live Better
LIVE_BETTER_BENEFIT_CATEGORIES = ["Live Better Benefit"]
LIVE_BETTER_BENEFIT_SPENDGROUPS = ["LiveBetterBenefit"]
PURCHASE_TRAN_REGEX = r"\bPurchase\b"                          # "Interest Charged for Purchases" does NOT match



def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _cat_flags(series, test):
    """Evaluate `test` (Index -> bool array) once per category, then map to rows. Fast for categorical columns."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        flag = np.asarray(test(series.cat.categories), dtype=bool)
        if len(flag) == 0:
            return np.zeros(len(series), dtype=bool)
        codes = series.cat.codes.to_numpy()
        return np.where(codes >= 0, flag[np.clip(codes, 0, None)], False)
    return np.asarray(test(pd.Index(series.astype(str))), dtype=bool)


def _isin(df, col, values):
    if col not in df or not values:
        return np.zeros(len(df), dtype=bool)
    return _cat_flags(df[col], lambda ix: ix.astype(str).isin([str(v) for v in values]))


def _regex(df, col, pattern):
    if col not in df:
        return np.zeros(len(df), dtype=bool)
    return _cat_flags(df[col], lambda ix: ix.astype(str).str.contains(pattern, regex=True, na=False))


def _group_ids(df, groups, default):
    """Group id (position in `groups`) for every row, from the category -> group lookup (once per category)."""
    names = list(groups)
    unk = names.index(default)
    lut = {str(c): i for i, g in enumerate(names) for c in groups[g]}
    if CAT not in df:
        return np.full(len(df), unk, dtype=np.int16)
    s = df[CAT]
    if isinstance(s.dtype, pd.CategoricalDtype):
        per_cat = np.array([lut.get(str(c), unk) for c in s.cat.categories], dtype=np.int16)
        if len(per_cat) == 0:
            return np.full(len(df), unk, dtype=np.int16)
        codes = s.cat.codes.to_numpy()
        return np.where(codes >= 0, per_cat[np.clip(codes, 0, None)], unk).astype(np.int16)
    return s.astype(str).map(lut).fillna(unk).astype(np.int16).to_numpy()


def _validate_config():
    """Catch typos: every group used by a lens or rule must exist."""
    bad = []
    for lens in (NON_SPEND_GROUPS, ESSENTIAL_GROUPS, DISCRETIONARY_GROUPS, COMMITTED_GROUPS, FEE_GROUPS, CASH_GROUPS):
        bad += [g for g in lens if g not in OUTFLOW_GROUPS]
    bad += [g for g in INCOME_GROUPS + [UIF_GROUP] if g not in INFLOW_GROUPS]
    if "unclassified" not in OUTFLOW_GROUPS or "other_inflow" not in INFLOW_GROUPS:
        bad += ["unclassified / other_inflow (default groups must exist)"]
    if bad:
        raise ValueError(f"Group names not defined in OUTFLOW_GROUPS / INFLOW_GROUPS: {sorted(set(bad))}")


# ======================================================================================================
# CONFIG
# ======================================================================================================
WINDOW_DAYS = 92       # how far back from M0 the running state starts (the extract holds ~98 days)
# Category / spendgroup definitions are local to this module.

SPEND_GROUPS = [g for g in OUTFLOW_GROUPS if g not in NON_SPEND_GROUPS]
HEADLINE = ["spend", "income", "essentials", "nonessential", "gambling", "cash", "fees", "live_better_in", "uif"]
COUNTS = ["n_txn", "n_purchases", "n_gambling", "n_fees", "n_insufficient_fee", "n_bounced", "n_cash"]


def build_masks(sub):
    """Boolean row masks (numpy), split amounts and group ids, computed once per batch."""
    amt = sub[AMT].to_numpy("float64")
    is_out, is_in = amt < 0, amt > 0
    out_g = _group_ids(sub, OUTFLOW_GROUPS, "unclassified")
    in_g = _group_ids(sub, INFLOW_GROUPS, "other_inflow")
    out_names, in_names = list(OUTFLOW_GROUPS), list(INFLOW_GROUPS)
    og = lambda names: np.isin(out_g, [out_names.index(n) for n in names])
    ig = lambda names: np.isin(in_g, [in_names.index(n) for n in names])
    spd = lambda v: _isin(sub, SPEND, v)

    spend = is_out & ~(og(NON_SPEND_GROUPS) | spd(NON_SPEND_SPENDGROUPS))
    fee = is_out & (og(FEE_GROUPS) | spd(FEE_SPENDGROUPS) | _regex(sub, TXN_TYPE, FEE_TRAN_REGEX))
    return {
        "out": np.where(is_out, -amt, 0.0), "in": np.where(is_in, amt, 0.0),
        "is_out": is_out, "is_in": is_in, "out_g": out_g, "in_g": in_g,
        "spend": spend,
        "purchase": is_out & _regex(sub, TXN_TYPE, PURCHASE_TRAN_REGEX),
        "essential": spend & og(ESSENTIAL_GROUPS),
        "nonessential": spend & og(DISCRETIONARY_GROUPS),
        "gambling": spend & og(["gambling"]),
        "cash": is_out & (og(CASH_GROUPS) | spd(CASH_SPENDGROUPS)),
        "fee": fee,
        "insufficient_fee": fee & _regex(sub, TXN_TYPE, INSUFFICIENT_REGEX),
        "bounced": spd(BOUNCED_SPENDGROUPS),
        "live_better_in": is_out & _regex(sub, TXN_TYPE, LIVE_BETTER_TRAN_REGEX),
        "live_better_benefit": is_in & (spd(LIVE_BETTER_BENEFIT_SPENDGROUPS) | _isin(sub, CAT, LIVE_BETTER_BENEFIT_CATEGORIES)),
        "income": is_in & ig(INCOME_GROUPS) & ~spd(NON_INCOME_SPENDGROUPS),
        "uif": is_in & ig([UIF_GROUP]),
    }


def resolve_observations(clients, collisions=None, verbose=True):
    """Validate and return one client row per obs_id."""
    c = clients.copy()
    if OBS not in c:
        raise KeyError(f"clients needs '{OBS}'")
    if c[OBS].isna().any():
        raise ValueError("clients contains missing obs_id values")
    if c[OBS].duplicated().any():
        raise ValueError("clients must contain exactly one row per obs_id")
    c[ID] = c[ID].astype(str)
    return c.sort_values(OBS).reset_index(drop=True)


def attach_key(txns, chosen):
    """Filter transactions to observations represented in clients; obs_id is the key."""
    if OBS not in txns:
        raise KeyError(f"txns needs '{OBS}'")
    return txns.loc[txns[OBS].isin(chosen[OBS])].copy()


# ======================================================================================================
# Window, ordering, running state
# ======================================================================================================
def _window(txns, as_of, chosen):
    """Rows inside [as_of, WINDOW_DAYS) days before M0, 'pre' only, sorted oldest -> newest inside each observation."""
    if AGE_COL not in txns:
        raise KeyError(f"'{AGE_COL}' missing: build txns with clean_data.load_and_clean first")
    age = txns[AGE_COL].to_numpy(dtype="float64", na_value=np.nan)
    keep = (age >= as_of) & (age < WINDOW_DAYS)
    if PRE_COL and PRE_COL in txns:
        keep &= _isin(txns, PRE_COL, [PRE_VALUE])
    if DROP_SPENDGROUPS:
        keep &= ~_isin(txns, SPEND, DROP_SPENDGROUPS)
    sub = txns.loc[keep].copy()
    sub[ID] = sub[ID].astype(str)
    sub["age"] = age[keep]
    sub = attach_key(sub, chosen)          # retain only known obs_id values
    gid = pd.factorize(sub[OBS])[0]
    # newest = smallest days_before_m0, so sort age DESCENDING; ties keep the original file order
    order = np.lexsort((np.arange(len(sub)), -sub["age"].to_numpy(), gid))
    return sub.iloc[order].reset_index(drop=True), gid[order]


def _running(sub, gid, by_group=True):
    """Running table: one row per transaction, state AFTER that transaction. Returns (table, internal columns)."""
    M = build_masks(sub)
    n = len(sub)
    ones = np.ones(n)
    add = {
        "n_txn": ones,
        "inflow": M["in"], "outflow": M["out"], "net": sub[AMT].to_numpy("float64"),
        "spend": np.where(M["spend"], M["out"], 0.0),
        "income": np.where(M["income"], M["in"], 0.0),
        "essentials": np.where(M["essential"], M["out"], 0.0),
        "nonessential": np.where(M["nonessential"], M["out"], 0.0),
        "gambling": np.where(M["gambling"], M["out"], 0.0),
        "cash": np.where(M["cash"], M["out"], 0.0),
        "fees": np.where(M["fee"], M["out"], 0.0),
        "live_better_in": np.where(M["live_better_in"], M["out"], 0.0),
        "uif": np.where(M["uif"], M["in"], 0.0),
        "n_purchases": M["purchase"].astype(float), "n_gambling": M["gambling"].astype(float),
        "n_fees": M["fee"].astype(float), "n_insufficient_fee": M["insufficient_fee"].astype(float),
        "n_bounced": M["bounced"].astype(float), "n_cash": M["cash"].astype(float),
    }
    if by_group:
        for i, g in enumerate(OUTFLOW_GROUPS):
            add[f"grp_{g}"] = np.where(M["is_out"] & (M["out_g"] == i), M["out"], 0.0)
        for i, g in enumerate(INFLOW_GROUPS):
            add[f"in_{g}"] = np.where(M["is_in"] & (M["in_g"] == i), M["in"], 0.0)

    cum = pd.DataFrame(add).groupby(gid).cumsum()
    cum.columns = ["run_" + c for c in cum.columns]

    age = sub["age"].to_numpy()
    bal = sub[BAL].to_numpy("float64") if BAL in sub else np.full(n, np.nan)
    b = pd.DataFrame({"b": bal, "neg": (bal < 0).astype(float)})
    cum["balance"] = bal
    cum["run_min_balance"] = b.groupby(gid)["b"].cummin().to_numpy()
    cum["run_max_balance"] = b.groupby(gid)["b"].cummax().to_numpy()
    cum["run_n_neg_balance"] = b.groupby(gid)["neg"].cumsum().to_numpy()

    # age (days before M0) of the most recent income / UIF payment seen so far; NaN = none yet
    last_inc = pd.Series(np.where(M["income"], age, np.nan)).groupby(gid).ffill().to_numpy()
    last_uif = pd.Series(np.where(M["uif"], age, np.nan)).groupby(gid).ffill().to_numpy()
    max_age = pd.Series(age).groupby(gid).transform("max").to_numpy()

    out = pd.DataFrame({ID: sub[ID].to_numpy(), OBS: sub[OBS].to_numpy(), AGE_COL: age})
    if "transactiondate" in sub:
        out["transactiondate"] = sub["transactiondate"].to_numpy()
    out = pd.concat([out, cum], axis=1)
    out["days_since_last_income"] = last_inc - age
    out["days_since_last_uif"] = last_uif - age
    out["days_covered"] = max_age - age + 1
    hidden = pd.DataFrame({"_last_income_age": last_inc, "_last_uif_age": last_uif, "_max_age": max_age})
    return out, hidden


# ======================================================================================================
# Current state per observation
# ======================================================================================================
def _state(sub, gid, as_of, by_group=True):
    """State at the as-of point = running state after each observation's last transaction in the window."""
    tbl, hid = _running(sub, gid, by_group)
    last = np.r_[gid[1:] != gid[:-1], True] if len(gid) else np.zeros(0, bool)
    s = pd.concat([tbl, hid], axis=1)[last].reset_index(drop=True)
    s = s.drop(columns=["transactiondate", "days_since_last_income", "days_since_last_uif", "days_covered"], errors="ignore")
    # recency / coverage measured from the as-of point, not from the last transaction
    s["days_since_last_txn"] = s[AGE_COL] - as_of
    s["days_since_last_income"] = s["_last_income_age"] - as_of
    s["days_since_last_uif"] = s["_last_uif_age"] - as_of
    s["days_covered"] = s["_max_age"] - as_of + 1
    return s.drop(columns=[AGE_COL, "_last_income_age", "_last_uif_age", "_max_age"]).set_index(OBS)


def add_state_ratios(s):
    """Scale-free ratios and rates from the running totals (better for distance-based methods)."""
    d = s["days_covered"].where(s["days_covered"] > 0)
    for nm in ("inflow", "outflow", "net", "spend", "income"):
        s[f"{nm}_per_day"] = s[f"run_{nm}"] / d
    s["n_txn_per_day"] = s["run_n_txn"] / d
    s["outflow_to_inflow"] = s["run_outflow"] / s["run_inflow"].where(s["run_inflow"] > 0)
    s["spend_to_income"] = s["run_spend"] / s["run_income"].where(s["run_income"] > 0)
    s["net_to_inflow"] = s["run_net"] / s["run_inflow"].where(s["run_inflow"] > 0)
    spd = s["spend_per_day"].where(s["spend_per_day"] > 0)
    s["balance_runway_days"] = (s["balance"] / spd).where(s["balance"] > 0)          # days the balance would last at the running spend rate

    sp = s["run_spend"].where(s["run_spend"] > 0)
    for nm in ("essentials", "nonessential", "gambling", "cash", "fees"):
        s[f"{nm}_share_of_spend"] = s[f"run_{nm}"] / sp
    gcols = [f"run_grp_{g}" for g in SPEND_GROUPS if f"run_grp_{g}" in s]
    if gcols:
        for c in gcols:
            s[c.replace("run_grp_", "grp_") + "_share"] = s[c] / sp
        s["committed_share_of_spend"] = s[[f"run_grp_{g}" for g in COMMITTED_GROUPS]].sum(axis=1) / sp
        P = s[gcols].div(sp, axis=0).fillna(0.0)
        s["spend_group_entropy"] = (-(P * np.log(P.where(P > 0, 1.0))).sum(axis=1)).where(sp.notna())
        s["n_spend_groups_used"] = (s[gcols] > 0).sum(axis=1).astype("int8")

    s["no_income_seen"] = s["days_since_last_income"].isna().astype("int8")
    s["unemployed_now"] = (s["run_uif"] > 0).astype("int8")          # any UIF received so far in the window
    # no income / UIF seen: recency = the whole covered history (a lower bound), flagged by no_income_seen
    s["days_since_last_income"] = s["days_since_last_income"].fillna(s["days_covered"])
    s["days_since_last_uif"] = s["days_since_last_uif"].fillna(s["days_covered"])
    return s


# ======================================================================================================
# Assembly (batched over whole customers)
# ======================================================================================================
def _batches(clients, txns, batch_customers):
    codes = pd.factorize(txns[ID])[0]
    n_batches = max(1, int(np.ceil(clients[ID].nunique() / batch_customers)))
    bid = codes % n_batches
    order = np.argsort(bid, kind="stable")
    bounds = np.searchsorted(bid[order], np.arange(n_batches + 1))
    for b in range(n_batches):
        yield b, n_batches, txns.iloc[order[bounds[b]:bounds[b + 1]]]


def _prep_clients(clients, collisions=None, verbose=True):
    """Validate one row per obs_id and return its index."""
    clients = resolve_observations(clients, collisions, verbose)
    return clients, pd.Index(clients[OBS], name=OBS)


def build_running_features(clients, txns, as_of=0, batch_customers=20_000, by_group=True, log_money=True, verbose=True,
                           collisions=None):
    """
    One row per observation: the running state at `as_of` days before M0 (0 = M0, i.e. the current state).
    Observations with no transactions in the window get zero totals, NaN balance and days_covered 0.
    """
    _validate_config()
    clients, idx = _prep_clients(clients, collisions, verbose)
    parts = []
    for b, nb, chunk in _batches(clients, txns, batch_customers):
        sub, gid = _window(chunk, as_of, clients)
        if len(sub):
            parts.append(_state(sub, gid, as_of, by_group))
        if verbose:
            print(f"  running features batch {b + 1}/{nb} done")
    s = pd.concat(parts) if parts else pd.DataFrame(index=pd.Index([], name=OBS))
    s = s.reindex(idx)
    zero = [c for c in s.columns if c.startswith("run_") and c not in ("run_min_balance", "run_max_balance")] + ["days_covered"]
    s[zero] = s[zero].fillna(0.0)
    s["days_since_last_txn"] = s["days_since_last_txn"]            # NaN = no transactions in the window
    s = add_state_ratios(s)

    if log_money:
        money = [c for c in s.columns if c.startswith("run_") and not c.startswith("run_n_")] + ["balance"]
        logs = pd.DataFrame({c + "_slog": signed_log1p(s[c]) for c in money}, index=s.index)
        s = pd.concat([s, logs], axis=1)
    return s


def build_running_snapshots(clients, txns, as_of=(60, 30, 0), **kw):
    """
    The running state at several cut-offs, stacked long. Index = (obs_id, as_of_days).
    Compare state at 60 / 30 / 0 days before M0 to see direction without monthly buckets, or train on every cut-off
    (keep grouped-by-customer splits). Use a cut-off >= the embargo you need for honest early-warning tests.
    """
    clients, _ = _prep_clients(clients, kw.pop("collisions", None), kw.get("verbose", True))   # resolve once
    kw["verbose"] = kw.get("verbose", True)
    frames = []
    for a in as_of:
        s = build_running_features(clients, txns, as_of=a, **kw)
        s["as_of_days"] = a
        frames.append(s.set_index("as_of_days", append=True))
    return pd.concat(frames).sort_index()


def running_table(clients, txns, batch_customers=20_000, by_group=True, as_of=0, verbose=True, collisions=None):
    """
    GENERATOR of per-transaction running tables (one DataFrame per batch of whole customers).
    Columns: customer, obs_id, days_before_m0, [transactiondate], run_* (state after this transaction), balance,
    run_min/max_balance, days_since_last_income / uif (as at that transaction), days_covered.
    """
    _validate_config()
    clients, _ = _prep_clients(clients, collisions, verbose)
    for b, nb, chunk in _batches(clients, txns, batch_customers):
        sub, gid = _window(chunk, as_of, clients)
        if verbose:
            print(f"  running table batch {b + 1}/{nb}: {len(sub):,} rows")
        if len(sub):
            tbl, _ = _running(sub, gid, by_group)
            yield tbl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="snapshot/snapshot.parquet")
    ap.add_argument("--out", default="features_running.parquet")
    ap.add_argument("--as-of", type=int, nargs="+", default=[0], help="cut-offs in days before M0, e.g. 60 30 0")
    ap.add_argument("--table-dir", default=None, help="also write the per-transaction running tables here (parquet parts)")
    ap.add_argument("--batch-customers", type=int, default=20_000)
    ap.add_argument("--no-groups", action="store_true")
    args, _ = ap.parse_known_args()
    from clean_data import load_and_clean
    clients, txns, _ = load_and_clean(args.path)
    kw = dict(batch_customers=args.batch_customers, by_group=not args.no_groups)
    out = (build_running_features(clients, txns, as_of=args.as_of[0], **kw) if len(args.as_of) == 1
           else build_running_snapshots(clients, txns, as_of=args.as_of, **kw))
    out.reset_index().to_parquet(args.out, index=False)
    print(f"Saved {out.shape[0]:,} rows x {out.shape[1]} columns to {args.out}")
    if args.table_dir:
        os.makedirs(args.table_dir, exist_ok=True)
        for i, t in enumerate(running_table(clients, txns, **kw)):
            t.to_parquet(os.path.join(args.table_dir, f"part_{i:04d}.parquet"), index=False)
        print(f"Per-transaction running tables written to {args.table_dir}/")
