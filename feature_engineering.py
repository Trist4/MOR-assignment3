"""
feature_engineering.py - skeleton for per-observation behavioural features, built from the cleaned tables.

    from clean_data import load_and_clean
    from feature_engineering import check_mappings, build_features

    clients, txns, report = load_and_clean("snapshot/snapshot.parquet")
    check_mappings(txns)                       # 1. fill in / correct the category mappings first
    feats = build_features(clients, txns)      # 2. one row per observation (customer + obs_id)

"Months": the window is cut into 30-day blocks counted back from the reference date (days_before_m0).
    m0 = the latest block (0-29 days before M0), m1 = 30-59 days, m2 = 60-89 days. Change N_MONTHS / MONTH_LEN below.

Requested feature                              -> columns produced
-----------------------------------------------------------------------------------------------------------
Number of purchases per person                 -> n_purchases_m0..m2, n_purchases_total
Money spent per month                          -> spend_m0..m2, spend_total
Change in spending trend                       -> spend_slope, spend_change_last_vs_prev   (income_* too)
Money spent per category                       -> cat_spend_<category> (top K + other), cat_share_<category>
Monthly income                                 -> income_m0..m2, income_total
Money spent on essentials                      -> essentials_m0..m2, essentials_total, essentials_share_of_spend
Money spent on non-essentials / gambling       -> nonessential_*, gambling_*, n_gambling_total, *_share_of_spend
Money going into Live Better                   -> live_better_in_m0..m2, live_better_in_total
Bank fees + penalties                          -> fees_*, n_fees_total, n_insufficient_fee_total, n_bounced_debits_total
Amount of cash drawn                           -> cash_m0..m2, cash_total, n_cash_withdrawals_total
Installments behind                             -> instalment_amount_est, instalments_behind_est
UIF received => unemployed that month          -> uif_received_m*, unemployed_m0..m2, months_unemployed, unemployed_latest

Everything marked DRAFT / TODO is an assumption to confirm with the data owner.
"""
import argparse
import re

import numpy as np
import pandas as pd

ID, OBS = "cifwithcheckdigit", "obs_id"
KEY = [ID, OBS]
TXN_TYPE, AMT, BAL, SPEND, CAT = "trantypedesc", "amount", "balance", "spendgroup", "categoryname"

# ======================================================================================================
# CONFIG
# ======================================================================================================
MONTH_LEN = 30                  # days per "month" block
N_MONTHS = 3                    # number of blocks (the extract holds ~98 days, so 3 x 30 is safe)
AGE_COL = "days_before_m0"      # days from the transaction to the reference date (>= 0 = before it)
TOP_K_CATEGORIES = 15           # categories that get their own cat_spend_* column; the rest go to "other"

# tx_period looks like a pre / post flag. TODO: confirm what 'post' means. Rows that fall after the payment date
# may already reflect the outcome, so only 'pre' rows are used. Set PRE_COL = None to switch the filter off.
PRE_COL, PRE_VALUE = "tx_period", "pre"

# CardAuthorisationTransaction and CardTransaction have near-identical counts and amounts: probably each purchase
# is recorded twice. TODO: confirm, then list the spend group to drop, e.g. ["CardAuthorisationTransaction"].
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
 

# ======================================================================================================
# Helpers
# ======================================================================================================
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
 
 
def _empty(cols):
    return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_arrays([[], []], names=KEY), dtype="float64")
 
 
def monthly_wide(sub, mask, values, name, per_month=True):
    """Sum `values` over rows in `mask`, per observation and month block. Columns: name_m0.. and name_total."""
    month_cols = [f"{name}_m{k}" for k in range(N_MONTHS)]
    total = f"{name}_total"
    cols = (month_cols if per_month else []) + [total]
    if not mask.any():
        return _empty(cols)
    d = sub.loc[mask, KEY + ["month_idx"]].copy()
    d["v"] = values[mask]
    g = (d.groupby(KEY + ["month_idx"], observed=True)["v"].sum().unstack("month_idx")
           .reindex(columns=range(N_MONTHS)).fillna(0.0))
    g.columns = month_cols
    g[total] = g.sum(axis=1)
    return g[cols]
 
 
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
# Window and masks
# ======================================================================================================
def restrict_window(txns):
    """Keep rows inside the look-back window (and 'pre' rows), add month_idx (0 = latest block)."""
    if AGE_COL not in txns:
        raise KeyError(f"'{AGE_COL}' missing: build txns with clean_data.load_and_clean first")
    age = txns[AGE_COL].to_numpy(dtype="float64", na_value=np.nan)
    keep = (age >= 0) & (age < N_MONTHS * MONTH_LEN)
    if PRE_COL and PRE_COL in txns:
        keep &= _isin(txns, PRE_COL, [PRE_VALUE])
    if DROP_SPENDGROUPS:
        keep &= ~_isin(txns, SPEND, DROP_SPENDGROUPS)
    sub = txns.loc[keep].copy()
    sub["month_idx"] = (age[keep] // MONTH_LEN).astype("int8")
    sub[ID] = sub[ID].astype(str)
    return sub
 
 
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
 


# ======================================================================================================
# One function per feature group. Each returns a frame indexed by (customer, obs_id).
# ======================================================================================================
def f_coverage(sub, M):
    """How much of the window actually has data (guards against reading missing history as zero spend)."""
    g = sub.groupby(KEY, observed=True).agg(n_txn_window=("month_idx", "size"), n_months_with_txn=("month_idx", "nunique"))
    return g.astype("float64")
 
 
def f_purchases(sub, M):
    return monthly_wide(sub, M["purchase"], np.ones(len(sub)), "n_purchases")
 
 
def f_monthly_spend(sub, M):
    """Money spent per month (all outflows except savings / internal movements). Trends: add_trends()."""
    return monthly_wide(sub, M["spend"], M["out"], "spend")
 
 
def _group_totals(sub, rows, group_ids, names, values, prefix):
    cols = [f"{prefix}{g}_total" for g in names]
    if not rows.any():
        return _empty(cols)
    d = sub.loc[rows, KEY].copy()
    d["g"] = group_ids[rows]
    d["v"] = values[rows]
    g = (d.groupby(KEY + ["g"], observed=True)["v"].sum().unstack("g")
           .reindex(columns=range(len(names))).fillna(0.0))
    g.columns = cols
    return g
 
 
def f_group_outflow(sub, M):
    """Condensed 'money spent per category': one total per outflow group (includes savings / transfers)."""
    return _group_totals(sub, M["is_out"], M["out_g"], list(OUTFLOW_GROUPS), M["out"], "grp_")
 
 
def f_group_inflow(sub, M):
    """One total per inflow group (income types, borrowing, savings drawdown, refunds ...)."""
    return _group_totals(sub, M["is_in"], M["in_g"], list(INFLOW_GROUPS), M["in"], "in_")
 
 
def f_income(sub, M):
    """Monthly income = inflows in INCOME_GROUPS (reversed debit orders and internal transfers excluded)."""
    return monthly_wide(sub, M["income"], M["in"], "income")
 
 
def f_essentials(sub, M):
    return monthly_wide(sub, M["essential"], M["out"], "essentials")
 
 
def f_nonessentials(sub, M):
    """Non-essential spend (DISCRETIONARY_GROUPS) and gambling on its own (gambling is also inside non-essentials)."""
    return pd.concat([monthly_wide(sub, M["nonessential"], M["out"], "nonessential"),
                      monthly_wide(sub, M["gambling"], M["out"], "gambling"),
                      monthly_wide(sub, M["gambling"], np.ones(len(sub)), "n_gambling", per_month=False)], axis=1)
 
 
def f_live_better(sub, M):
    return pd.concat([monthly_wide(sub, M["live_better_in"], M["out"], "live_better_in"),
                      monthly_wide(sub, M["live_better_benefit"], M["in"], "live_better_benefit", per_month=False)], axis=1)
 
 
def f_fees(sub, M):
    return pd.concat([monthly_wide(sub, M["fee"], M["out"], "fees"),
                      monthly_wide(sub, M["fee"], np.ones(len(sub)), "n_fees", per_month=False),
                      monthly_wide(sub, M["insufficient_fee"], np.ones(len(sub)), "n_insufficient_fee", per_month=False),
                      monthly_wide(sub, M["bounced"], np.ones(len(sub)), "n_bounced_debits", per_month=False)], axis=1)
 
 
def f_cash(sub, M):
    return pd.concat([monthly_wide(sub, M["cash"], M["out"], "cash"),
                      monthly_wide(sub, M["cash"], np.ones(len(sub)), "n_cash_withdrawals", per_month=False)], axis=1)
 
 
def f_unemployment(sub, M):
    return monthly_wide(sub, M["uif"], M["in"], "uif_received")
 
 
def f_instalments_behind(clients):
    """
    Instalments behind, estimated from information known AT M0 (never from payment_count / total_paid: those
    describe the outcome period and would leak the target).
    TODO: confirm with the data owner. Assumes m0_total_due_with_arrears = arrears + the next `due_count` instalments.
    """
    c = clients.copy()
    c[ID] = c[ID].astype(str)
    c = c.set_index(KEY)
    arrears = c["m0_arrearsamount"].astype("float64")
    total_due = c["m0_total_due_with_arrears"].astype("float64")
    n_due = c["due_count"].astype("float64").replace(0, np.nan)
    inst = ((total_due - arrears) / n_due).where(lambda s: s > 0)
    return pd.DataFrame({"instalment_amount_est": inst, "instalments_behind_est": arrears / inst})


# ======================================================================================================
# Derived features (run after missing months are zero-filled)
# ======================================================================================================
def add_unemployment(feats):
    """UIF received in a month => unemployed in that month."""
    for k in range(N_MONTHS):
        feats[f"unemployed_m{k}"] = (feats[f"uif_received_m{k}"] > 0).astype("int8")
    feats["months_unemployed"] = feats[[f"unemployed_m{k}" for k in range(N_MONTHS)]].sum(axis=1).astype("int8")
    feats["unemployed_latest"] = feats["unemployed_m0"]
    return feats
 
 
def add_trends(feats, names=("spend", "income")):
    """OLS slope per month (oldest -> latest) and relative change latest vs previous month."""
    if N_MONTHS < 2:
        return feats
    x = np.arange(N_MONTHS, dtype="float64")
    x -= x.mean()
    for nm in names:
        A = feats[[f"{nm}_m{k}" for k in range(N_MONTHS)]].to_numpy("float64")     # column 0 = latest month
        feats[f"{nm}_slope"] = (A[:, ::-1] * x).sum(axis=1) / (x ** 2).sum()
        prev = A[:, 1]
        feats[f"{nm}_change_last_vs_prev"] = np.where(prev > 0, (A[:, 0] - prev) / np.where(prev > 0, prev, 1.0), np.nan)
    return feats
 
 
def add_ratios(feats):
    """Shares of spend (scale-free: better for distance-based methods) and the condensed summary features."""
    sp = feats["spend_total"].where(feats["spend_total"] > 0)
    spend_groups = [g for g in OUTFLOW_GROUPS if g not in NON_SPEND_GROUPS]
    gcols = [f"grp_{g}_total" for g in spend_groups]
    for g in spend_groups:
        feats[f"grp_{g}_share"] = feats[f"grp_{g}_total"] / sp
    for nm in ("essentials", "nonessential", "gambling", "cash", "fees"):
        feats[f"{nm}_share_of_spend"] = feats[f"{nm}_total"] / sp
    feats["committed_share_of_spend"] = feats[[f"grp_{g}_total" for g in COMMITTED_GROUPS]].sum(axis=1) / sp
    P = feats[gcols].div(sp, axis=0).fillna(0.0)
    feats["spend_group_entropy"] = (-(P * np.log(P.where(P > 0, 1.0))).sum(axis=1)).where(sp.notna())
    feats["n_spend_groups_used"] = (feats[gcols] > 0).sum(axis=1).astype("int8")
    return feats
 
# ======================================================================================================
# Assembly
# ======================================================================================================
def txn_features(sub):
    sub = restrict_window(sub)
    if sub.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=KEY))
    M = build_masks(sub)
    parts = [f_coverage(sub, M), f_purchases(sub, M), f_monthly_spend(sub, M), f_group_outflow(sub, M),
             f_group_inflow(sub, M), f_income(sub, M), f_essentials(sub, M), f_nonessentials(sub, M),
             f_live_better(sub, M), f_fees(sub, M), f_cash(sub, M), f_unemployment(sub, M)]
    return pd.concat(parts, axis=1)
 
 
def build_features(clients, txns, batch_customers=20_000, log_money=True, verbose=True):
    """
    Returns one row per observation (index = [customer, obs_id]).
    Memory-safe: transactions are processed in batches of whole customers.
    log_money: also add <col>_slog for money totals (sign(x) * log(1 + |x|)).
    """
    _validate_config()
    clients = clients.copy()
    clients[ID] = clients[ID].astype(str)
    idx = pd.MultiIndex.from_frame(clients[KEY])
 
    codes = pd.factorize(txns[ID])[0]
    n_batches = max(1, int(np.ceil(clients[ID].nunique() / batch_customers)))
    bid = codes % n_batches                                   # every customer lands wholly in one batch
    order = np.argsort(bid, kind="stable")
    bounds = np.searchsorted(bid[order], np.arange(n_batches + 1))
 
    parts = []
    for b in range(n_batches):
        parts.append(txn_features(txns.iloc[order[bounds[b]:bounds[b + 1]]]))
        if verbose:
            print(f"  features batch {b + 1}/{n_batches} done")
    tf = pd.concat(parts) if parts else pd.DataFrame(index=idx)
 
    feats = tf.reindex(idx).fillna(0.0)        # no in-window transactions => amounts and counts are 0
    feats = feats.join(f_instalments_behind(clients))
    feats = add_unemployment(feats)
    feats = add_trends(feats)
    feats = add_ratios(feats)
 
    if log_money:
        money = [c for c in feats.columns if (c.endswith("_total") and not c.startswith("n_")) or c == "instalment_amount_est"]
        for c in money:
            feats[c + "_slog"] = signed_log1p(feats[c])
    return feats


# ======================================================================================================
# Mapping diagnostics
# ======================================================================================================
def mapping_table():
    """Category -> outflow group / inflow group, one row per category named in the config (save it to review/share)."""
    out_lut = {str(c): g for g, cs in OUTFLOW_GROUPS.items() for c in cs}
    in_lut = {str(c): g for g, cs in INFLOW_GROUPS.items() for c in cs}
    cats = sorted(set(out_lut) | set(in_lut))
    return pd.DataFrame({"category": cats,
                         "outflow_group": [out_lut.get(c, "(default: unclassified)") for c in cats],
                         "inflow_group": [in_lut.get(c, "(default: other_inflow)") for c in cats]})
 
 
def check_mappings(txns, top=25):
    """What the groups look like on your data, which categories are unmapped, and the pre/post window check."""
    _validate_config()
    for label, groups in (("outflow", OUTFLOW_GROUPS), ("inflow", INFLOW_GROUPS)):
        seen, dup = {}, []
        for g, cs in groups.items():
            for c in cs:
                if c in seen:
                    dup.append(f"{c} ({seen[c]} and {g})")
                seen[c] = g
        if dup:
            print(f"WARNING: category listed in two {label} groups (the later one wins): {dup}")
    both = set(ESSENTIAL_GROUPS) & set(DISCRETIONARY_GROUPS)
    if both:
        print(f"WARNING: groups in both essential and discretionary: {sorted(both)}")
 
    cats = txns[CAT]
    if isinstance(cats.dtype, pd.CategoricalDtype):
        codes, names = cats.cat.codes.to_numpy(), cats.cat.categories
    else:
        codes, names = pd.factorize(cats)
    amt = txns[AMT].to_numpy("float64")
    ok = codes >= 0
    mo, mi = (amt < 0) & ok, (amt > 0) & ok
    stats = pd.DataFrame({"n": np.bincount(codes[ok], minlength=len(names)),
                          "outflow": np.bincount(codes[mo], weights=-amt[mo], minlength=len(names)),
                          "inflow": np.bincount(codes[mi], weights=amt[mi], minlength=len(names))},
                         index=pd.Index(names.astype(str), name=CAT))
    out_lut = {str(c): g for g, cs in OUTFLOW_GROUPS.items() for c in cs}
    in_lut = {str(c): g for g, cs in INFLOW_GROUPS.items() for c in cs}
    stats["outflow_group"] = [out_lut.get(c, "unclassified") for c in stats.index]
    stats["inflow_group"] = [in_lut.get(c, "other_inflow") for c in stats.index]
 
    for label, col, lut in (("OUTFLOW", "outflow", out_lut), ("INFLOW", "inflow", in_lut)):
        g = stats.groupby(f"{col}_group")[col].agg(["sum"]).rename(columns={"sum": col})
        g["share_%"] = g[col] / g[col].sum() * 100
        g["n_categories"] = stats[stats[col] > 0].groupby(f"{col}_group").size()
        print(f"\n{label} groups on your data (share of total {col}):")
        print(g.sort_values(col, ascending=False).to_string(float_format=lambda v: f"{v:,.1f}"))
 
    named = set(out_lut) | set(in_lut)
    unmapped = stats[~stats.index.isin(named)].assign(total=lambda d: d.outflow + d.inflow).sort_values("total", ascending=False)
    print(f"\nCategories in the data that no group names ({len(unmapped)}; they fall into unclassified / other_inflow):")
    print(unmapped[["n", "outflow", "inflow"]].head(top).to_string(float_format=lambda v: f"{v:,.0f}"))
    missing = sorted(named - set(stats.index))
    if missing:
        print(f"\nNamed in the config but absent from this data ({len(missing)}): {missing}")
 
    print("\nWindow checks")
    if PRE_COL and PRE_COL in txns:
        print(f"{PRE_COL} counts: {txns[PRE_COL].value_counts().to_dict()}  (only '{PRE_VALUE}' rows are used)")
        extra = [c for c in (AGE_COL, "days_from_payment") if c in txns]
        if extra:
            print(txns.groupby(PRE_COL, observed=True)[extra].agg(["min", "max"]).to_string())
    if SPEND in txns:
        sg = set(txns[SPEND].astype(str).unique())
        if {"CardAuthorisationTransaction", "CardTransaction"} <= sg and not DROP_SPENDGROUPS:
            print("NOTE: CardAuthorisationTransaction and CardTransaction are both present and DROP_SPENDGROUPS is empty:"
                  " card spend may be double counted.")
    return unmapped

 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="snapshot/snapshot.parquet")
    ap.add_argument("--out", default="features.parquet")
    ap.add_argument("--batch-customers", type=int, default=20_000)
    args, _ = ap.parse_known_args()
    from clean_data import load_and_clean
    clients, txns, _ = load_and_clean(args.path)
    check_mappings(txns)
    mapping_table().to_csv("category_groups.csv", index=False)
    feats = build_features(clients, txns, args.batch_customers)
    feats.reset_index().to_parquet(args.out, index=False)
    print(f"Saved {feats.shape[0]:,} observations x {feats.shape[1]} features to {args.out} (mapping in category_groups.csv)")