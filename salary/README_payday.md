# Capitec transaction segmentation: pay-day features and clustering

This folder holds three files:

| File                       | What it does                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `build_features.py`      | Reads the Capitec snapshot and builds pay-day, pay-cycle and observation-level feature tables. Writes them to `output/`. |
| `cluster_kmeans_1.ipynb` | K-means segmentation on the features.                                                                                      |
| `cluster_gmm_1.ipynb`    | Gaussian mixture (GMM) segmentation on the same features, plus a comparison with k-means.                                  |

Run order: `build_features.py` → `cluster_kmeans_1.ipynb` → `cluster_gmm_1.ipynb`. The GMM notebook compares itself with the k-means output, so k-means must run first.

```bash
python build_features.py                                    # defaults below
python build_features.py --input snapshot/snapshot.parquet --out output
```

---

## 1. Key ideas used everywhere

- **Observation (`obs_id`)**: one client, account and m0 month. A client can have several observations. **Every table is keyed on `obs_id`, never on `cifwithcheckdigit`.** All features are built per observation.
- **Payment date**: the anchor for each observation's transaction window. All time is counted *backwards* from it.
- **Period**: a 30-day block counted back from the payment date.
  - `p0` = 0–29 days before the payment date (most recent)
  - `p1` = 30–59 days before
  - `p2` = 60–89 days before (oldest)
- **Post-payment week is ignored** (`INCLUDE_POST = False`). Every feature uses only transactions *before* the payment date. No feature uses information from after m0, so there is no leakage into the target.
- **Pay day / pay event**: an income day whose amount is at least 50% of that observation's median income-day amount. Small side credits are kept in the income table but don't start a new pay cycle.
- **Pay cycle**: from one pay event up to (but not including) the next one. The last cycle before the payment date has no next pay, so it ends at the payment date.
- **Spend**: outflows (`amount < 0`) that are not transfers (`spendgroup` contains "transfer") and not savings/investments (`Savings`, `Automatic Savings`, `Investments`, `Other Savings & Investments`). Uncategorised outflows count as spend.
- **Income categories**:
  - Salary: `Salary`, `Wage`
  - Other income: `Pension`, `UIF Received`, `Commission`, `Allowance`, `Child Support`, `Rental Income`, `Other Income`
  - Ambiguous: `Payment Received` (only used when the client has no salary in that period)
  - Transfers are never income.

### How income is found

1. **Salary**: every salary/wage credit is a pay day.
2. **No salary in a period**: the single largest income credit in that period is taken as the estimated pay.
   - `estimated_regular`: found in 2+ periods and always within 3 days of the same day of the month.
   - `estimated_irregular`: otherwise.
3. **Frequency detection**: credits of similar size (within ~15%) are grouped. If a group has 3+ credits, its median gap gives the frequency: weekly (5–9 days), fortnightly (12–16), monthly (26–35), else irregular. For weekly/fortnightly earners without salary, *all* those credits are used as pay days instead of only the largest one per period.
4. Several income credits on the same day are summed into one pay day.

---

## 2. Output tables (`output/`)

### 2.1 `payday_tbl.parquet`: the salary / pay-day table

**One row per income day per observation.** This is the "salary" table: every day income was received, and whether it counts as a pay day.

| Column            | Meaning                                                                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `obs_id`        | Observation key.                                                                                                                                                                                                        |
| `pay_date`      | Date the income was received (time removed).                                                                                                                                                                            |
| `income_amount` | Total income received that day (R). Several credits on the same day are summed.                                                                                                                                         |
| `n_payments`    | Number of income credits combined into this day.                                                                                                                                                                        |
| `category`      | Transaction categories of those credits, comma-separated (e.g. `Salary` or `Pension, Other Income`).                                                                                                                |
| `income_type`   | How the income was identified, in priority order: `salary`, `estimated_weekly`, `estimated_fortnightly`, `estimated_regular`, `estimated_irregular`. If a day mixes types, the highest-priority one is shown. |
| `period`        | 0, 1 or 2: which 30-day period before the payment date (0 = most recent).                                                                                                                                               |
| `pay_frequency` | The observation's detected main pay frequency: `weekly`, `fortnightly`, `monthly`, `irregular`, or `unknown` (no group of 3+ similar credits found). Same value on every row of an observation.               |
| `is_pay_event`  | `True` if this day counts as a pay day (amount ≥ 50% of the observation's median income-day amount). Only these days start pay cycles.                                                                               |

### 2.2 `pay_cycles.parquet`

**One row per pay event per observation**, including the last cycle before the payment date.

| Column                  | Meaning                                                                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `obs_id`              | Observation key.                                                                                                                                                  |
| `pay_date`            | Pay day that starts this cycle.                                                                                                                                   |
| `income_amount`       | Pay received on `pay_date` (R).                                                                                                                                 |
| `next_pay`            | Next pay day. Empty for the last cycle.                                                                                                                           |
| `payment_date`        | The observation's payment (collection) date.                                                                                                                      |
| `is_last`             | `True` for the last cycle (no next pay before the payment date).                                                                                                |
| `cycle_end`           | `next_pay`, or the payment date for the last cycle.                                                                                                             |
| `cycle_days_obs`      | Days from `pay_date` to `cycle_end`.                                                                                                                          |
| `bal7`                | End-of-day account balance on day 7 of the cycle (pay day = day 1). Empty if the cycle was shorter than 7 days or there was no balance reading since the pay day. |
| `bal7_ratio`          | `bal7 / income_amount`: share of the pay still in the account a week later. Can be negative or above 1 (the balance includes money from before the pay).        |
| `outflow`             | Total non-transfer outflows in the first 3 days of the cycle (pay day + 2), excluding the payment date itself (R).                                                |
| `debit_order_outflow` | Part of `outflow` that is debit orders (`trantypedesc` contains "debit order").                                                                               |
| `committed_share`     | `debit_order_outflow / outflow`.                                                                                                                                |
| `spend`               | Total spend in the cycle (R). **Completed cycles only**; empty for the last cycle.                                                                          |
| `spend_ratio`         | `spend / income_amount`.                                                                                                                                        |
| `spend_per_day`       | `spend / cycle length in days`.                                                                                                                                 |
| `days_to_half`        | Days after pay until cumulative spend reaches 50% of the pay. Equals the cycle length if it never does.                                                           |
| `frac_to_half`        | `days_to_half / cycle length`. Low = pay is spent quickly.                                                                                                      |
| `first3_share`        | Spend in the first 3 days (pay day + 2) / pay.                                                                                                                    |

### 2.3 `per_period.parquet`

**One row per observation per 30-day period** (always 3 rows per observation, even with no income). Sorted oldest → newest.

| Column          | Meaning                                                               |
| --------------- | --------------------------------------------------------------------- |
| `obs_id`      | Observation key.                                                      |
| `period`      | 0 = most recent 30 days, 2 = oldest.                                  |
| `income`      | Total income in the period (R); 0 if none.                            |
| `n_payments`  | Number of income credits in the period.                               |
| `income_type` | Best (highest-priority) income type found in the period, or `none`. |
| `has_income`  | `True` if `income > 0`.                                           |

### 2.4 `obs_features.parquet`: the main feature table

**One row per observation.** This is what the notebooks cluster on and what should be merged with other features.

#### Static and target columns (copied from the snapshot)

| Column                                                                                | Meaning                                                                                                   |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `obs_id`, `cifwithcheckdigit`, `obs_seq`                                        | Observation key, client number, client's 1st/2nd/3rd… observation.                                       |
| `m0_monthenddate`, `payment_date`                                                 | Observation month and payment date.                                                                       |
| `product_group`, `m0_start_cd_group`, `new_to_arrears_flag`, `crccreditlimit` | Static attributes (see the data notes from Capitec).                                                      |
| `target_value`                                                                      | 1 = bad, 0 = good. **Validation only, never a clustering feature.**                                 |
| `target_rule`                                                                       | Target rule **number only** (e.g. `5` for "5) Payments Received < Payments Due").                 |
| `exclude_flag`                                                                      | 1 = excluded from modelling (total due < R50 or < 3 dues). The notebooks keep only `exclude_flag == 0`. |

#### Income level and stability

| Column                                      | Meaning                                                     |
| ------------------------------------------- | ----------------------------------------------------------- |
| `periods_with_income`                     | Number of the 3 periods with any income (0–3).             |
| `avg_period_income`                       | Mean income per 30-day period (R).                          |
| `income_cv`                               | Std / mean of the 3 period incomes. High = volatile income. |
| `pay_frequency`                           | Detected pay frequency (as in `payday_tbl`). Text column. |
| `income_p0`, `income_p1`, `income_p2` | Income per period (p0 = most recent).                       |
| `n_payments_p0..p2`                       | Income credits per period.                                  |
| `income_type_p0..p2`                      | Best income type per period (text).                         |

#### Income direction (time-aware)

| Column                      | Meaning                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `avg_abs_income_change`   | Mean absolute period-to-period change in income, relative to mean income. High = income jumps around. |
| `last_income_change`      | Change from p1 to p0, relative to mean income. Negative = latest month dropped.                       |
| `income_trend`            | Slope of income over p2 → p1 → p0, divided by mean income. Negative = income falling.               |
| `recency_weighted_income` | Weighted mean income with weights 1, 0.5, 0.25 for p0, p1, p2 (half-life of 1 period).                |
| `recent_vs_avg_income`    | `recency_weighted_income / avg_period_income`. Below 1 = recent income below usual.                 |

#### Pay-day regularity

| Column           | Meaning                                                                         |
| ---------------- | ------------------------------------------------------------------------------- |
| `pay_gap_mean` | Mean days between pay events.                                                   |
| `pay_gap_std`  | Std of those gaps.                                                              |
| `pay_gap_cv`   | `pay_gap_std / pay_gap_mean`. High = irregular pay days. Needs 3+ pay events. |

#### Spending within pay cycles (completed cycles, averaged)

| Column                     | Meaning                                                   |
| -------------------------- | --------------------------------------------------------- |
| `n_cycles`               | Number of completed pay cycles.                           |
| `avg_spend_between_pays` | Mean spend per cycle (R).                                 |
| `avg_spend_ratio`        | Mean spend / pay. Above 1 = spending more than the pay.   |
| `avg_spend_per_day`      | Mean spend per day (R).                                   |
| `avg_days_to_half`       | Mean days to spend half the pay.                          |
| `avg_frac_to_half`       | Same, as a fraction of cycle length. Low = pay goes fast. |
| `avg_first3_share`       | Mean share of pay spent in the first 3 days.              |

#### Balance and committed spend after pay

| Column                        | Meaning                                                                                                                            |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `bal7_ratio_last`           | `bal7_ratio` of the latest cycle that had a full first week before the payment date.                                             |
| `bal7_ratio_avg`            | Mean `bal7_ratio` of the *earlier* cycles (the client's "usual").                                                              |
| `bal7_ratio_change`         | `bal7_ratio_last − bal7_ratio_avg`. Below 0 = less left a week after pay than usual.                                            |
| `committed_share_after_pay` | Across all cycles: debit-order outflows / all outflows in the first 3 days after pay. High = pay is pre-committed to debit orders. |

#### Missing-data flags (0/1)

Missing values are **structural** (e.g. no income means no pay cycles), so each gap is flagged and then filled.

| Flag                   | Set when                                   | Features affected                                               |
| ---------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| `no_income_detected` | No income in any period                    | `income_cv`, `recent_vs_avg_income`, income-change features |
| `few_pay_events`     | Fewer than 3 pay events                    | `pay_gap_cv`, `pay_gap_std`                                 |
| `no_complete_cycle`  | Fewer than 2 pay events                    | All `avg_spend_*`, `avg_*_to_half`, `avg_first3_share`    |
| `no_bal7_history`    | No earlier full-week cycle to compare with | `bal7_ratio_avg`, `bal7_ratio_change`                       |

**How gaps are filled:** features with a natural "no change" value get it (`recent_vs_avg_income = 1`, `n_cycles`, `last_income_change`, `income_trend`, `avg_abs_income_change`, `bal7_ratio_change`, `committed_share_after_pay = 0`). Everything else gets the median over `exclude_flag == 0`. Always look at the flag together with the filled value: a filled median is not real behaviour.

---

## 3. What happens in `build_features.py`

| Step | Function                 | What it does                                                                                                                                                    |
| ---- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0    | `load_snapshot`        | Reads the parquet, turns categoricals into plain types, keeps only the number of `target_rule`, drops exact duplicate rows, parses dates.                     |
| 1    | `prepare_transactions` | Drops duplicate transactions per observation, assigns each to period 0/1/2, flags transfers, salary, income, debit orders.                                      |
| 2    | `build_income_tbl`     | Finds pay days (salary, estimated pay, weekly/fortnightly detection) → income table and pay frequency. Prints a check for unexpected categories (should be 0). |
| 3    | `build_per_period`     | Income per period, long and wide, plus income level/volatility summary.                                                                                         |
| 4    | `build_pay_cycles`     | Picks pay events, builds completed cycles, attaches spend to cycles, computes spend speed measures.                                                             |
| 5    | `build_after_pay`      | `bal7` and committed-spend measures for every cycle, including the last.                                                                                      |
| 6    | `build_obs_features`   | Rolls everything up to one row per observation, adds trend and recency features.                                                                                |
| 7    | `fill_missing`         | Adds flags, fills gaps, prints the missing share before filling and the flagged shares.                                                                         |

Settings at the top of the file can be changed (period length, number of periods, frequency tolerance, pay-event threshold, debit-order text pattern, etc.).

**Check before relying on it:** `DEBIT_ORDER_PATTERNS = ['debit order']` is matched against `trantypedesc`. Confirm this matches the real values in the data, otherwise `committed_share_after_pay` will be 0 for everyone.

---

## 4. What happens in the notebooks

Both notebooks share the same setup and validation; only the clustering model differs.

### Shared steps

1. **Settings**: list of the 11 clustering features, which ones get a log transform, the 4 missing-data flags and `FLAG_WEIGHT`.
   | Feature                       | Captures                             |
   | ----------------------------- | ------------------------------------ |
   | `avg_period_income` (log)   | Income level                         |
   | `income_cv`                 | Income volatility                    |
   | `recent_vs_avg_income`      | Income direction                     |
   | `pay_gap_cv`                | Pay-day regularity                   |
   | `avg_spend_ratio`           | Spend vs pay                         |
   | `avg_spend_per_day` (log)   | Spend level                          |
   | `avg_frac_to_half`          | How fast pay is spent                |
   | `avg_first3_share`          | Spending straight after pay          |
   | `bal7_ratio_avg`            | Usual share of pay left after a week |
   | `bal7_ratio_change`         | Latest cycle vs usual                |
   | `committed_share_after_pay` | Debit-order pressure after pay       |
2. **Load**: reads `obs_features.parquet`, keeps `exclude_flag == 0`, checks there are no missing values left, shows the share of each flag.
3. **Prepare**: `log1p` on income and spend level, winsorize each feature at the 1st/99th percentile, standardise to z-scores. The flags are appended as 0/1 × `FLAG_WEIGHT` (not standardised). A Spearman correlation heatmap shows features that overlap (|r| > 0.7 → consider dropping one).
4. **Cluster profile**: size, share and bad rate per cluster (ordered by bad rate), median raw feature values per cluster, a heatmap of mean z-scores (red = above average, blue = below), and the share of flagged observations per cluster. Use this to name the clusters. If a cluster is mostly flagged rows, it is splitting on missing data → lower `FLAG_WEIGHT`.
5. **Over time**: overall bad rate, cluster share by m0 month (rows should look similar if clusters are stable) and bad rate by cluster and month.
6. **Static attributes**: cluster vs `product_group` and vs `m0_start_cd_group`. This shows whether the behavioural clusters say something the static segments don't.
7. **PCA view**: 2D scatter of a 5,000-point sample, coloured by cluster.

### `cluster_kmeans_1.ipynb`

- **Choose K**: fits K = 2…8 and plots inertia (elbow), silhouette (higher is better; computed on a 10,000-row sample) and Davies-Bouldin (lower is better).
- **Stability**: for K = 3…6, refits on 10 bootstrap samples and compares with the full fit using the adjusted Rand index (ARI; above ~0.8 = stable).
- **Final fit**: `K = 4` (change after looking at the plots).
- **Output**: `output/clusters_kmeans.parquet` with `obs_id`, `cluster`.

### `cluster_gmm_1.ipynb`

A Gaussian mixture allows clusters of different shapes and sizes and gives each observation a probability per cluster.

- **Choose K and covariance type**: fits K = 2…8 for each covariance type (`full`, `diag`, `tied`, `spherical`) and plots BIC and AIC (lower is better). BIC often keeps falling, so pick where it flattens and clusters are interpretable.
- **Silhouette / Davies-Bouldin** for the chosen covariance type, for comparison with k-means (these favour round clusters, so they can undersell a GMM).
- **Final fit**: `K = 4`, `COVARIANCE_TYPE = 'full'`, 10 random starts.
- **Assignment confidence**: histogram and per-cluster boxplot of `max_prob`, and the share of observations below 0.6 (sitting between clusters).
- **Stability**: same bootstrap ARI check as k-means.
- **Compare with k-means**: ARI between the two labelings and a crosstab. Note it looks for `output/clusters_kmeans.parquet`, so run the k-means notebook first. Cluster numbers are arbitrary, so use the crosstab, not the numbers, to match clusters across methods.
- **Output**: `output/clusters_gmm.parquet` with `obs_id`, `cluster`, `max_prob`, `prob_0` … `prob_{K-1}`.

---

## 5. Merging with other feature sets

1. **Join on `obs_id`, one row per observation.** Do not join on `cifwithcheckdigit`: one client can have several observations with different windows and targets.
2. **Keep the static/target columns from one side only**, otherwise pandas creates `_x` / `_y` duplicates.
3. **Use the same time window.** These features use only transactions *before* `payment_date`, in 30-day periods counted back from it. Features built on a different window (e.g. including the post-payment week) are not directly comparable.
4. **Never use post-m0 columns as features**: `due_count`, `payment_count`, `total_paid`, `m0_total_due_with_arrears`, `target_value`, `target_rule`. They describe what happened after observation and define the target.
5. **Filter `exclude_flag == 0`** before clustering, and fill any new missing values the same way (flag + neutral value or median over `exclude_flag == 0`).

```python
import pandas as pd

STATIC = ['cifwithcheckdigit', 'obs_seq', 'm0_monthenddate', 'payment_date', 'product_group',
          'm0_start_cd_group', 'new_to_arrears_flag', 'crccreditlimit',
          'target_value', 'target_rule', 'exclude_flag']

mine = pd.read_parquet('output/obs_features.parquet')
theirs = pd.read_parquet('path/to/their_features.parquet')

theirs = theirs.drop(columns=[c for c in STATIC if c in theirs.columns])
merged = mine.merge(theirs, on='obs_id', how='left', validate='one_to_one')

print(len(mine), len(merged))                     # should match
print(merged.isna().sum().loc[lambda s: s > 0])  # new gaps to flag and fill
merged.to_parquet('output/obs_features_merged.parquet', index=False)
```

To cluster on the merged set, point `FEATURES_FILE` in the notebooks at the merged file, add the new feature names to `CLUSTER_FEATURES` (and to `LOG_FEATURES` if they are skewed amounts), and re-check the Spearman heatmap for overlapping features.
