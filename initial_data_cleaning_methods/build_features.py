"""
Build observation-level pay-day tables and features from the Capitec snapshot.

Run from the project folder:
    python build_features.py
    python build_features.py --input snapshot/snapshot.parquet --out output

Writes (all one file each, in --out):
    payday_tbl.parquet     one row per income day per observation (salary + estimated income),
                           with is_pay_event = the days that count as a pay day
    pay_cycles.parquet     one row per pay cycle per observation (pay day -> next pay day,
                           last cycle ends at the payment date), with spend, bal7 and
                           committed-spend measures per cycle
    per_period.parquet     one row per observation per 30-day period (p0 = most recent)
    obs_features.parquet   one row per observation: static columns, target and all features.
                           Missing features are filled (see FILL_MISSING) and the reason is kept
                           in 0/1 flag columns: no_income_detected, few_pay_events,
                           no_complete_cycle, no_bal7_history

Use in other files:
    import pandas as pd
    obs = pd.read_parquet('output/obs_features.parquet')
or
    from build_features import load_table
    obs = load_table('obs_features')
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# Settings
# =====================================================================
DEFAULT_INPUT = 'snapshot/snapshot.parquet'
DEFAULT_OUT = 'output'

PERIOD_DAYS = 30        # length of one "month", counted back from the payment date
N_PERIODS = 3           # periods before the payment date (p0 = most recent 30 days)
INCLUDE_POST = False    # True: include the week after the payment date (added to p0)

WINDOW = 3              # max days apart for pay days to count as regular
MIN_OCCURRENCES = 3     # similar credits needed to detect a pay frequency
BAND = 0.15             # credits within ~15% of each other count as "similar"
PAY_MIN_SHARE = 0.5     # a payment is a "pay day" if >= 50% of the median payment
HALF_LIFE = 1           # periods; recency weight halves every HALF_LIFE periods back

BAL_DAYS_AFTER_PAY = 7  # bal7 = end-of-day balance after the first 7 days of a cycle (pay day = day 1)
COMMITTED_DAYS = 3      # "straight after pay" window for committed spend (pay day + next 2 days)
DEBIT_ORDER_PATTERNS = ['debit order']   # case-insensitive match on trantypedesc; check your values

SALARY_CATS = ['Salary', 'Wage']
INCOME_CATS = ['Pension', 'UIF Received', 'Commission', 'Allowance',
               'Child Support', 'Rental Income', 'Other Income']
AMBIGUOUS_CATS = ['Payment Received']
NOT_SPEND = ['Savings', 'Automatic Savings', 'Investments', 'Other Savings & Investments']

FILL_MISSING = True     # flag why a feature is missing, then fill it (no rows need to be dropped)
# Values that mean "no change / nothing" for a feature; every other missing feature gets the median
FILL_NEUTRAL = {
    'n_cycles': 0,                     # no complete pay cycle
    'recent_vs_avg_income': 1,         # recent income = usual income
    'last_income_change': 0,
    'income_trend': 0,
    'avg_abs_income_change': 0,
    'bal7_ratio_change': 0,            # latest week-after-pay balance = usual
    'committed_share_after_pay': 0,    # no outflows straight after pay -> no debit orders either
}

TYPE_PRIORITY = ['salary','estimated_weekly', 'estimated_fortnightly',
                 'estimated_regular', 'estimated_irregular']
RANK_MAP = {t: i for i, t in enumerate(TYPE_PRIORITY)}

OBS_COLS = ['obs_id', 'cifwithcheckdigit', 'obs_seq', 'm0_monthenddate', 'payment_date',
            'product_group', 'm0_start_cd_group', 'new_to_arrears_flag', 'crccreditlimit',
            'target_value', 'target_rule', 'exclude_flag']


# =====================================================================
# Helpers
# =====================================================================
def safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def contains_any(s, patterns):
    s = s.astype('string').fillna('')
    return s.str.contains('|'.join(map(re.escape, patterns)), case=False, regex=True).astype(bool)


def max_day_gap(days):
    d = days.to_numpy()
    diff = np.abs(d[:, None] - d[None, :])
    diff = np.minimum(diff, 31 - diff)  # 31st and 1st are 1 day apart
    return diff.max()


# =====================================================================
# 0. Load
# =====================================================================
def load_snapshot(path=DEFAULT_INPUT):
    df = pd.read_parquet(path)
    # Categorical columns break string operations and make groupby build every combination
    for c in df.select_dtypes('category').columns:
        df[c] = df[c].astype(df[c].cat.categories.dtype)
    df['target_rule'] = (df['target_rule'].astype('string')
                         .str.extract(r'^(\d+)', expand=False).astype('Int64'))
    df = df.drop_duplicates()
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    df['payment_date'] = pd.to_datetime(df['payment_date'])
    df['m0_monthenddate'] = pd.to_datetime(df['m0_monthenddate'])
    print(f'{len(df):,} rows | {df["obs_id"].nunique():,} observations | '
          f'{df["cifwithcheckdigit"].nunique():,} clients')
    return df


# =====================================================================
# 1. Transactions per observation, assigned to 30-day periods
# =====================================================================
def prepare_transactions(df):
    key = ['obs_id', 'transactiondate', 'amount', 'balance', 'trantypedesc']
    tx = df.dropna(subset=['transactiondate']).drop_duplicates(subset=key).copy()

    tx['days_before_pay'] = -tx['days_from_payment']           # positive = before payment date
    if INCLUDE_POST:
        tx['days_before_pay'] = tx['days_before_pay'].clip(lower=0)
    tx['period'] = tx['days_before_pay'] // PERIOD_DAYS          # 0 = most recent 30 days
    tx = tx[tx['period'].between(0, N_PERIODS - 1)].copy()
    tx['period'] = tx['period'].astype(int)

    tx['is_transfer'] = contains_any(tx['spendgroup'], ['transfer'])
    tx['is_salary'] = tx['categoryname'].isin(SALARY_CATS)
    tx['is_income'] = tx['categoryname'].isin(INCOME_CATS + AMBIGUOUS_CATS) & ~tx['is_transfer']
    tx['has_salary'] = tx.groupby(['obs_id', 'period'])['is_salary'].transform('any')
    tx['is_debit_order'] = (tx['amount'] < 0) & contains_any(tx['trantypedesc'], DEBIT_ORDER_PATTERNS)
    return tx


# =====================================================================
# 2. Income / pay-day table
# =====================================================================
def build_income_tbl(tx):
    rename_map = {'transactiondate': 'pay_date', 'amount': 'income_amount', 'categoryname': 'category'}
    cols = ['obs_id', 'pay_date', 'income_amount', 'category', 'income_type', 'period']

    # Salary credits
    salary_tbl = (tx[tx['is_salary'] & (tx['amount'] > 0)]
        [['obs_id', 'transactiondate', 'amount', 'categoryname', 'period']]
        .rename(columns=rename_map)
        .assign(income_type='salary'))

    # Highest income credit per observation-period without salary
    credits = tx[(tx['amount'] > 0) & tx['is_income'] & ~tx['has_salary']]
    idx = credits.groupby(['obs_id', 'period'])['amount'].idxmax()
    est_tbl = (credits.loc[idx, ['obs_id', 'transactiondate', 'amount', 'categoryname', 'period']]
        .rename(columns=rename_map))
    est_tbl['day'] = est_tbl['pay_date'].dt.day
    grp = est_tbl.groupby('obs_id')['day']
    est_tbl['n_periods'] = grp.transform('size')
    est_tbl['day_gap'] = grp.transform(max_day_gap)
    est_tbl['income_type'] = np.where(
        (est_tbl['n_periods'] >= 2) & (est_tbl['day_gap'] <= WINDOW),
        'estimated_regular', 'estimated_irregular')

    # Pay frequency detection (weekly / fortnightly / monthly)
    inc = tx[(tx['amount'] > 0) & (tx['is_income'] | tx['is_salary'])][
        ['obs_id', 'transactiondate', 'amount', 'categoryname', 'period', 'has_salary']].copy()
    inc['amount_band'] = np.round(np.log(inc['amount']) / np.log(1 + BAND)).astype(int)
    inc = inc.sort_values(['obs_id', 'amount_band', 'transactiondate'])
    inc['gap'] = inc.groupby(['obs_id', 'amount_band'])['transactiondate'].diff().dt.days
    inc['weekday'] = inc['transactiondate'].dt.dayofweek

    pattern = inc.groupby(['obs_id', 'amount_band']).agg(
        n=('amount', 'size'), total=('amount', 'sum'), median_gap=('gap', 'median'))
    wd = inc.groupby(['obs_id', 'amount_band', 'weekday']).size()
    pattern['weekday_share'] = wd.groupby(level=[0, 1]).max() / wd.groupby(level=[0, 1]).sum()
    pattern = pattern.reset_index()
    pattern = pattern[pattern['n'] >= MIN_OCCURRENCES]
    pattern['pay_frequency'] = np.select(
        [pattern['median_gap'].between(5, 9),
         pattern['median_gap'].between(12, 16),
         pattern['median_gap'].between(26, 35)],
        ['weekly', 'fortnightly', 'monthly'], default='irregular')

    main = pattern.sort_values('total', ascending=False).drop_duplicates('obs_id')
    pay_freq = main.set_index('obs_id')[['pay_frequency', 'median_gap', 'weekday_share', 'n']]

    # Replace single-max rows with all payments for weekly/fortnightly
    multi = main[main['pay_frequency'].isin(['weekly', 'fortnightly'])][
        ['obs_id', 'amount_band', 'pay_frequency']]
    freq_tbl = (inc.merge(multi, on=['obs_id', 'amount_band'])
        .loc[lambda d: ~d['has_salary']]
        .rename(columns=rename_map)
        .assign(income_type=lambda d: 'estimated_' + d['pay_frequency']))
    replaced = freq_tbl[['obs_id', 'period']].drop_duplicates()
    est_keep = (est_tbl
        .merge(replaced, on=['obs_id', 'period'], how='left', indicator=True)
        .loc[lambda d: d['_merge'] == 'left_only'])

    # Combine, same-day payments merged
    income_tbl = pd.concat([salary_tbl[cols], est_keep[cols], freq_tbl[cols]])
    income_tbl['pay_date'] = income_tbl['pay_date'].dt.normalize()
    income_tbl['type_rank'] = income_tbl['income_type'].map(RANK_MAP)
    income_tbl = (income_tbl
        .sort_values(['obs_id', 'pay_date', 'type_rank'])
        .groupby(['obs_id', 'pay_date'], as_index=False)
        .agg(income_amount=('income_amount', 'sum'),
             n_payments=('income_amount', 'size'),
             category=('category', lambda c: ', '.join(sorted(c.dropna().astype(str).unique()))),
             income_type=('income_type', 'first'),
             type_rank=('type_rank', 'first'),
             period=('period', 'first')))
    income_tbl = (income_tbl
        .merge(pay_freq['pay_frequency'], left_on='obs_id', right_index=True, how='left')
        .fillna({'pay_frequency': 'unknown'})
        .sort_values(['obs_id', 'pay_date'])
        .reset_index(drop=True))

    allowed = SALARY_CATS + INCOME_CATS + AMBIGUOUS_CATS
    cat_check = income_tbl['category'].str.split(', ').explode()
    print('Unexpected categories:', (~cat_check.isin(allowed) & cat_check.ne('')).sum())
    return income_tbl, pay_freq


# =====================================================================
# 3. Income per 30-day period
# =====================================================================
def build_per_period(income_tbl, pay_freq, obs_index):
    grid = pd.MultiIndex.from_product([obs_index, range(N_PERIODS)],
                                      names=['obs_id', 'period']).to_frame(index=False)
    per_period = (income_tbl
        .groupby(['obs_id', 'period'], as_index=False)
        .agg(income=('income_amount', 'sum'),
             n_payments=('n_payments', 'sum'),
             type_rank=('type_rank', 'min')))
    per_period = grid.merge(per_period, on=['obs_id', 'period'], how='left')
    per_period['income'] = per_period['income'].fillna(0)
    per_period['n_payments'] = per_period['n_payments'].fillna(0).astype(int)
    per_period['income_type'] = (per_period['type_rank']
                                 .map(dict(enumerate(TYPE_PRIORITY))).fillna('none'))
    per_period = (per_period.drop(columns='type_rank')
        .sort_values(['obs_id', 'period'], ascending=[True, False])   # oldest -> newest
        .reset_index(drop=True))

    # Wide: income_p0 (most recent) ... income_p2 (oldest)
    wide = per_period.pivot(index='obs_id', columns='period',
                            values=['income', 'n_payments', 'income_type'])
    wide.columns = [f'{val}_p{p}' for val, p in wide.columns]
    wide = wide[sorted(wide.columns, key=lambda c: (c.rsplit('_p', 1)[1], c))]

    per_period['has_income'] = per_period['income'] > 0
    summary = per_period.groupby('obs_id').agg(
        periods_with_income=('has_income', 'sum'),
        avg_period_income=('income', 'mean'),
        income_std=('income', 'std'))
    summary['income_cv'] = safe_div(summary['income_std'], summary['avg_period_income'])
    summary = (summary.drop(columns='income_std')
        .join(pay_freq['pay_frequency'])
        .fillna({'pay_frequency': 'unknown'}))
    return per_period, wide, summary


# =====================================================================
# 4. Pay cycles and spending
# =====================================================================
def build_pay_cycles(income_tbl, tx, obs_base):
    pay = income_tbl[['obs_id', 'pay_date', 'income_amount']].copy()
    med = pay.groupby('obs_id')['income_amount'].transform('median')
    pay_events = (pay[pay['income_amount'] >= PAY_MIN_SHARE * med]
        .sort_values(['obs_id', 'pay_date']).copy())
    pay_events['next_pay'] = pay_events.groupby('obs_id')['pay_date'].shift(-1)
    pay_events['cycle_days'] = (pay_events['next_pay'] - pay_events['pay_date']).dt.days
    pay_events['gap'] = pay_events.groupby('obs_id')['pay_date'].diff().dt.days
    cycles = pay_events.dropna(subset=['next_pay'])            # completed cycles only

    spend_mask = (tx['amount'] < 0) & ~tx['is_transfer'] & ~tx['categoryname'].isin(NOT_SPEND)
    spend = (tx.loc[spend_mask, ['obs_id', 'transactiondate', 'amount']]
        .assign(spend=lambda d: -d['amount'])
        .sort_values('transactiondate'))

    s = pd.merge_asof(spend, cycles.sort_values('pay_date'),
                      left_on='transactiondate', right_on='pay_date',
                      by='obs_id', direction='backward')
    s = s[s['pay_date'].notna() & (s['transactiondate'] < s['next_pay'])].copy()
    s['days_after_pay'] = (s['transactiondate'] - s['pay_date']).dt.days
    s = s.sort_values(['obs_id', 'pay_date', 'transactiondate'])
    s['cum_share'] = s.groupby(['obs_id', 'pay_date'])['spend'].cumsum() / s['income_amount']
    s['spend_first3'] = s['spend'].where(s['days_after_pay'] < 3, 0)

    agg = s.groupby(['obs_id', 'pay_date']).agg(
        spend=('spend', 'sum'), spend_first3=('spend_first3', 'sum'))
    half = (s[s['cum_share'] >= 0.5]
        .groupby(['obs_id', 'pay_date'])['days_after_pay'].min()
        .rename('days_to_half'))

    per_cycle = (cycles.set_index(['obs_id', 'pay_date'])
        .join(agg).join(half)
        .fillna({'spend': 0, 'spend_first3': 0}))
    per_cycle['days_to_half'] = per_cycle['days_to_half'].fillna(per_cycle['cycle_days'])
    per_cycle['spend_ratio'] = per_cycle['spend'] / per_cycle['income_amount']
    per_cycle['spend_per_day'] = per_cycle['spend'] / per_cycle['cycle_days']
    per_cycle['frac_to_half'] = per_cycle['days_to_half'] / per_cycle['cycle_days']
    per_cycle['first3_share'] = per_cycle['spend_first3'] / per_cycle['income_amount']
    return pay_events, per_cycle


# =====================================================================
# 5. Balance after pay (bal7) and committed spend after pay
# =====================================================================
def build_after_pay(pay_events, tx, obs_base):
    """Per-cycle bal7 and committed-spend measures, for ALL cycles including the last one.

    The last cycle (last pay before the payment date) ends at the payment date, so it
    shows the situation going into the collection. Earlier cycles are the client's "usual".
    """
    ev = pay_events[['obs_id', 'pay_date', 'income_amount', 'next_pay']].merge(
        obs_base[['payment_date']], left_on='obs_id', right_index=True, how='left')
    ev['is_last'] = ev['next_pay'].isna()
    ev['cycle_end'] = ev['next_pay'].fillna(ev['payment_date'].dt.normalize())
    ev['cycle_days_obs'] = (ev['cycle_end'] - ev['pay_date']).dt.days

    # --- bal7: end-of-day balance on day 7 of the cycle (pay day = day 1), as share of the pay
    bal = (tx.dropna(subset=['balance'])[['obs_id', 'transactiondate', 'balance']]
           .sort_values(['obs_id', 'transactiondate'], kind='mergesort'))
    bal['day'] = bal['transactiondate'].dt.normalize()
    daily = bal.groupby(['obs_id', 'day'], as_index=False)['balance'].last()

    ev['bal_day'] = ev['pay_date'] + pd.Timedelta(days=BAL_DAYS_AFTER_PAY - 1)
    ev = pd.merge_asof(ev.sort_values('bal_day'),
                       daily.rename(columns={'balance': 'bal7'}).sort_values('day'),
                       left_on='bal_day', right_on='day', by='obs_id', direction='backward')
    # Only valid if the whole first week was observed before the next pay / the payment date
    observed = (ev['cycle_days_obs'] >= BAL_DAYS_AFTER_PAY) & (ev['day'] >= ev['pay_date'])
    ev['bal7'] = ev['bal7'].where(observed)
    ev['bal7_ratio'] = ev['bal7'] / ev['income_amount']
    ev = ev.drop(columns=['bal_day', 'day'])

    # --- Committed spend: share of outflows in the first COMMITTED_DAYS days that are debit orders
    out = tx.loc[(tx['amount'] < 0) & ~tx['is_transfer'],
                 ['obs_id', 'transactiondate', 'amount', 'is_debit_order']]
    out = out.merge(obs_base[['payment_date']], left_on='obs_id', right_index=True)
    out = out[out['transactiondate'].dt.normalize() < out['payment_date'].dt.normalize()]  # exclude the collection day
    out = pd.merge_asof(out.drop(columns='payment_date').sort_values('transactiondate'),
                        ev[['obs_id', 'pay_date']].sort_values('pay_date'),
                        left_on='transactiondate', right_on='pay_date',
                        by='obs_id', direction='backward')
    days_after = (out['transactiondate'].dt.normalize() - out['pay_date']).dt.days
    out = out[days_after.between(0, COMMITTED_DAYS - 1)].copy()
    out['outflow'] = -out['amount']
    out['debit_order_outflow'] = out['outflow'].where(out['is_debit_order'], 0.0)
    per_cyc = out.groupby(['obs_id', 'pay_date'])[['outflow', 'debit_order_outflow']].sum()
    ev = ev.merge(per_cyc.reset_index(), on=['obs_id', 'pay_date'], how='left')
    ev['committed_share'] = safe_div(ev['debit_order_outflow'], ev['outflow'])

    # --- Observation-level features
    # "Latest" = most recent cycle whose whole first week was observed before the payment date
    # (a pay landing on/just before the payment date has no full week yet, so it is skipped).
    valid = ev[ev['bal7_ratio'].notna()].sort_values(['obs_id', 'pay_date'])
    is_latest = valid.groupby('obs_id').cumcount(ascending=False) == 0
    feats = pd.DataFrame(index=obs_base.index)
    feats['bal7_ratio_last'] = valid[is_latest].set_index('obs_id')['bal7_ratio']
    feats['bal7_ratio_avg'] = valid[~is_latest].groupby('obs_id')['bal7_ratio'].mean()   # earlier cycles
    feats['bal7_ratio_change'] = feats['bal7_ratio_last'] - feats['bal7_ratio_avg']      # <0 = less left than usual
    tot = ev.groupby('obs_id')[['outflow', 'debit_order_outflow']].sum(min_count=1)
    feats['committed_share_after_pay'] = safe_div(tot['debit_order_outflow'], tot['outflow'])

    ev = ev.sort_values(['obs_id', 'pay_date']).reset_index(drop=True)
    return ev, feats


# =====================================================================
# 6. Observation-level features
# =====================================================================
def build_obs_features(obs_base, summary, wide, per_period, pay_events, per_cycle, after_pay_feats):
    spend_feats = per_cycle.groupby('obs_id').agg(
        n_cycles=('spend', 'size'),
        avg_spend_between_pays=('spend', 'mean'),
        avg_spend_ratio=('spend_ratio', 'mean'),
        avg_spend_per_day=('spend_per_day', 'mean'),
        avg_days_to_half=('days_to_half', 'mean'),
        avg_frac_to_half=('frac_to_half', 'mean'),
        avg_first3_share=('first3_share', 'mean'))

    reg_feats = pay_events.groupby('obs_id')['gap'].agg(pay_gap_mean='mean', pay_gap_std='std')
    reg_feats['pay_gap_cv'] = reg_feats['pay_gap_std'] / reg_feats['pay_gap_mean']

    pp = per_period.copy()   # sorted oldest -> newest
    mean_inc = pp.groupby('obs_id')['income'].transform('mean')
    pp['rel_change'] = (pp.groupby('obs_id')['income'].diff() / mean_inc).replace([np.inf, -np.inf], np.nan)
    pp['abs_rel_change'] = pp['rel_change'].abs()
    t_centred = (N_PERIODS - 1 - pp['period']) - (N_PERIODS - 1) / 2   # oldest = lowest
    pp['t_income'] = t_centred * pp['income']
    sxx = sum((k - (N_PERIODS - 1) / 2) ** 2 for k in range(N_PERIODS))
    chg_feats = pp.groupby('obs_id').agg(
        avg_abs_income_change=('abs_rel_change', 'mean'),
        last_income_change=('rel_change', 'last'),
        slope_sum=('t_income', 'sum'),
        mean_income=('income', 'mean'))
    chg_feats['income_trend'] = (chg_feats['slope_sum'] / sxx / chg_feats['mean_income']
                                 ).replace([np.inf, -np.inf], np.nan)
    chg_feats = chg_feats[['avg_abs_income_change', 'last_income_change', 'income_trend']]

    pp['w'] = 0.5 ** (pp['period'] / HALF_LIFE)
    pp['w_income'] = pp['income'] * pp['w']
    sums = pp.groupby('obs_id')[['w_income', 'w']].sum()
    recency_feats = pd.DataFrame({'recency_weighted_income': sums['w_income'] / sums['w']})
    recency_feats['recent_vs_avg_income'] = (
        recency_feats['recency_weighted_income'] / summary['avg_period_income']
    ).replace([np.inf, -np.inf], np.nan)

    return (obs_base
        .join([summary, reg_feats, spend_feats, chg_feats, recency_feats, after_pay_feats, wide])
        .reset_index())


# =====================================================================
# 7. Missing values: flag the reason, then fill
# =====================================================================
FLAG_COLS = ['no_income_detected', 'few_pay_events', 'no_complete_cycle', 'no_bal7_history']


def fill_missing(obs_features, non_feature_cols):
    """Add 0/1 flags for WHY features are missing, then fill the gaps.

    Missingness is structural, not random:
      no_income_detected : no income found at all            -> income_cv, recent_vs_avg_income, ...
      few_pay_events     : < 3 pay days, gap variation undefined -> pay_gap_cv, pay_gap_std
      no_complete_cycle  : < 2 pay days, no full pay cycle    -> avg_spend_*, avg_frac_to_half, ...
      no_bal7_history    : no earlier full-week cycle to compare with -> bal7_ratio_avg / _change
    Fills use FILL_NEUTRAL where a "no change" value exists, otherwise the median of the
    modelling population (exclude_flag == 0).
    """
    f = obs_features.copy()
    f['no_income_detected'] = (f['periods_with_income'] == 0).astype(int)
    f['few_pay_events'] = f['pay_gap_cv'].isna().astype(int)
    f['no_complete_cycle'] = f['n_cycles'].isna().astype(int)
    f['no_bal7_history'] = f['bal7_ratio_change'].isna().astype(int)

    feat_cols = [c for c in f.columns
                 if c not in non_feature_cols + FLAG_COLS and pd.api.types.is_numeric_dtype(f[c])]
    missing_before = f[feat_cols].isna().mean()

    f = f.fillna({c: v for c, v in FILL_NEUTRAL.items() if c in f.columns})
    medians = f.loc[f['exclude_flag'] == 0, feat_cols].median()
    f[feat_cols] = f[feat_cols].fillna(medians)

    print('\nMissing share before filling (features with any missing):')
    print(missing_before[missing_before > 0].round(3).to_string())
    print('Share flagged:')
    print(f.loc[f['exclude_flag'] == 0, FLAG_COLS].mean().round(3).to_string())
    print('Missing values left in features:', int(f[feat_cols].isna().sum().sum()))
    return f


# =====================================================================
# Pipeline
# =====================================================================
def build_all(df):
    obs_base = df[OBS_COLS].drop_duplicates('obs_id').set_index('obs_id')
    tx = prepare_transactions(df)
    income_tbl, pay_freq = build_income_tbl(tx)
    per_period, wide, summary = build_per_period(income_tbl, pay_freq, obs_base.index)
    pay_events, per_cycle = build_pay_cycles(income_tbl, tx, obs_base)
    pay_cycles, after_pay_feats = build_after_pay(pay_events, tx, obs_base)
    obs_features = build_obs_features(obs_base, summary, wide, per_period,
                                      pay_events, per_cycle, after_pay_feats)
    if FILL_MISSING:
        obs_features = fill_missing(obs_features, non_feature_cols=OBS_COLS)

    # Pay-day table: every income day, flagged if it counts as a pay day
    pe_keys = pay_events[['obs_id', 'pay_date']].assign(is_pay_event=True)
    payday_tbl = (income_tbl.drop(columns='type_rank')
                  .merge(pe_keys, on=['obs_id', 'pay_date'], how='left')
                  .fillna({'is_pay_event': False}))
    payday_tbl['is_pay_event'] = payday_tbl['is_pay_event'].astype(bool)

    # Pay cycles: add the spend measures of completed cycles
    spend_cols = ['spend', 'spend_ratio', 'spend_per_day', 'days_to_half', 'frac_to_half', 'first3_share']
    pay_cycles = pay_cycles.merge(per_cycle[spend_cols].reset_index(), on=['obs_id', 'pay_date'], how='left')

    print(income_tbl['income_type'].value_counts())
    print('Observations:', len(obs_features), '| Columns:', obs_features.shape[1])
    return {'payday_tbl': payday_tbl, 'pay_cycles': pay_cycles,
            'per_period': per_period, 'obs_features': obs_features}


def save_tables(tables, out_dir=DEFAULT_OUT):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        path = out / f'{name}.parquet'
        t.to_parquet(path, index=False)
        print(f'Saved {path}  ({len(t):,} rows, {t.shape[1]} columns)')


def load_table(name='obs_features', out_dir=DEFAULT_OUT):
    """Load one of the saved tables: 'obs_features', 'payday_tbl', 'pay_cycles', 'per_period'."""
    return pd.read_parquet(Path(out_dir) / f'{name}.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build pay-day tables and observation features.')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='snapshot parquet file')
    parser.add_argument('--out', default=DEFAULT_OUT, help='folder for the output parquet files')
    args = parser.parse_args()

    tables = build_all(load_snapshot(args.input))
    save_tables(tables, args.out)

