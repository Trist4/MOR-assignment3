# Early-Arrears EDA: Findings and Actions

**Source:** executed `arrears_eda.ipynb` (first version, 36 cells, run on the full dataset).
**Scope:** 52,025,014 transaction rows, 120,424 customers, 6 monthly M0 cohorts (Sep 2025 to Feb 2026).

**How to read evidence labels:** **[Observed]** is printed directly in the notebook. **[Derived]** is arithmetic I did on the printed outputs. **[Hypothesis]** is an interpretation that needs checking with the data owner or a follow-up cell.

---

## 1. Executive summary

1. **This run does not yet answer the core question.** It profiles the data and ranks single features. The controlled comparison of time-aware vs static features (hypotheses H1 to H3 in the second version of the notebook) has not been run on this data.
2. **Fix four data issues before any modelling.** (a) 24.3% of rows are exact duplicates. (b) The target disagrees with its own definition for 16.1% of customers. (c) Transfers, round-ups and bounced-debit-order reversals are counted as income and spend. (d) The transaction extract appears to be anchored on `payment_date`, not M0.
3. **The strongest single predictor is a point-in-time snapshot, not a trend.** Closing balance (`bal_last`) separates the classes more than any other feature. Whatever "static" baseline you compare against must include it, so H2 (time-aware features vs the best non-temporal baseline) is the test that matters.
4. **Collections method is the strongest static variable.** "Early Debit Order, linked to clients paydate" customers miss payments 51.6% of the time vs 28.0% for plain Debit Order. This may reflect how customers were assigned a method rather than any effect of the method itself.
5. **Risk is concentrated at near-zero closing balances and low activity.** Customers whose closing balance falls in the second decile miss payments about 78% of the time vs 43.5% overall. Overdrawn customers sit near the baseline.
6. **The target rate drifts a lot between cohorts** (50.9% in Sep, 29.2% in Jan). A 70/30 out-of-time split puts Jan and Feb in the test set, so train and test event rates differ materially (about 47.8% vs 33.5%).
7. **History is only about 98 days per customer.** Several features in the second notebook version assume up to 180 days and need adjusting.

---

## 2. Data scope and structure

| Item | Value | Label |
|---|---|---|
| Rows / columns | 52,025,014 / 16 | Observed |
| Customers | 120,424, each with exactly one M0 and one `payment_date` | Observed |
| M0 cohorts | 6 month-ends, Sep 2025 to Feb 2026 | Observed |
| Rows per customer | median 327, mean 432, 5th pct 52, 95th pct 1,165, max 8,232 | Observed |
| Missing values | none in any column | Observed |
| Transactions dated after M0 | 0 | Observed |
| In-memory size | 3.6 GB (narrow dtypes: float32, uint8, category) | Observed |
| Share of rows that are inflows | 22.6% | Observed |
| Share of transaction rows with a negative balance | 7.5% | Observed |

**Good news:** no temporal leakage in the raw data, no missing values, no repeated customers, and every customer has pre-M0 history. The data is rich: roughly 4 transactions per customer per day, which looks like a primary account.

---

## 3. Data-integrity issues to fix first

### 3.1 Exact duplicate rows (24.3%)
- **Evidence [Observed]:** 12,623,849 fully duplicated rows. The notebook's check for "one copy per `payment_date`" came back negative (max 1 payment date per customer), so this is **not** a join artefact from payments.
- **Why it matters:** counts and sums (`n_txn`, `inflow_sum`, `outflow_sum`) are inflated, probably unevenly across customers. This adds noise to every volume feature.
- **Action:** find out why the duplicates exist (ingestion double-load, or two source tables). If they are true duplicates, deduplicate (`DEDUPE_TXNS = True`) before computing any feature, and re-run the feature ranking.

### 3.2 Likely double-recorded card purchases
- **Evidence [Observed]:** `CardAuthorisationTransaction` (8.75M rows, mean amount -259) and `CardTransaction` (8.55M rows, mean -259) are almost identical in size and amount, and their per-customer shares correlate at 0.88.
- **Hypothesis:** each purchase appears once as an authorisation and once as the settled transaction.
- **Action:** keep one of the two for spend features (normally the settled record) after confirming with the data owner.

### 3.3 Internal movements and reversals are counted as income and spend
- **Evidence [Observed]:**
  - `InternalTransfer` is 18.7% of all rows and 45% of them are inflows.
  - "Round-up Transfer" and "Live Better Round-up Transfer" have identical row counts (1,032,395) and equal and opposite amounts.
  - `UnpaidOrBouncedDebitOrder` is 100% inflows and "DebiCheck Debit Order Insufficient Funds" is a positive amount, so these are reversals of failed collections, not income.
  - Monthly total inflow and outflow track each other almost exactly (the monthly chart shows the two lines nearly on top of each other).
- **Why it matters:** `inflow_sum`, `outflow_sum`, `net_flow` and `outflow_to_inflow` mostly measure money moving between the customer's own accounts. This probably also explains the counter-intuitive `net_flow` result in section 6.
- **Action:** build "true income" (salary, `Other Income`, `Payment Received`, `UIF Received`, and similar) and "true spend" (card, cash, debit orders, payments), and exclude transfers, round-ups and reversals from both. Keep the excluded flows as separate features, for example bounced-debit-order count.

### 3.4 The target does not match its stated definition (16.1% disagreement)
- **Evidence [Observed]:** the stated rule `target = 1 if payment_count < due_count` disagrees with `target_value` for 19,401 customers (16.11%).
- **Evidence [Derived from the crosstab]:** 15,721 customers have `payment_count >= due_count` but `target = 1`, and 3,680 have `payment_count < due_count` but `target = 0`.
- **Other signs `payment_count` is not an instalment count [Observed]:** mean 3.99, 99th percentile 19, max 161, while `due_count` is capped at 3 (98.3% of customers have exactly 3).
- **A better-fitting definition [Observed]:** `paid_to_due_ratio` separates the classes cleanly. Median 1.00 (25th percentile 1.00) for `target = 0`, versus median 0.05 (25th percentile 0.00) for `target = 1`. Median `total_paid` is R3,317 vs R100.
- **Hypothesis:** the target is amount-based (paid less than due), not count-based.
- **Action:** confirm the definition with the data owner. Then check the alternative:

```python
for due_col in ["m0_total_due_with_arrears", "m0_arrearsamount"]:
    alt = (cust["total_paid"] < cust[due_col]).astype(int)
    print(due_col, "agreement:", (alt == cust[TARGET]).mean().round(4))
    print(pd.crosstab(cust[TARGET], alt, normalize="index").round(3))
```

  A 16% label disagreement caps achievable model performance and makes every downstream metric hard to interpret until it is resolved.

### 3.5 The transaction window looks anchored on `payment_date`, not M0
- **Evidence [Observed]:**
  - `payment_date` is always on or before M0 (median 5 days before, up to 30 days before), so it is **not** a future payment date.
  - Maximum history before M0 averages 98.3 days (sd 7.3), which is close to 90 days plus the mean `payment_date` offset of 6.7 days.
  - The histogram of days before M0 is thin in the last ~25 days and rises towards day ~90.
- **Hypothesis:** the extract takes the 90 days before `payment_date`, so the days between `payment_date` and M0 are missing for most customers. In that case `bal_last` is the balance at `payment_date`, not at M0, and any "days since last transaction" feature measured from M0 mostly encodes the `payment_date` offset.
- **Action:** run this check, then measure every recency feature from the customer's last transaction date or from `payment_date`:

```python
g = df.groupby(ID).agg(last_txn=(TXN_DATE, "max"), pay=(PAY_DATE, "first"), m0=(M0, "first"))
print("payment_date - last txn (days):\n", (g["pay"] - g["last_txn"]).dt.days.describe())
print("M0 - last txn (days):\n", (g["m0"] - g["last_txn"]).dt.days.describe())
```

- **Also:** `payment_date` itself was never analysed against the target. Add `(M0 - payment_date).days` as a candidate feature once its meaning is confirmed.

### 3.6 Smaller issues
| Issue | Evidence | Action |
|---|---|---|
| `active_days` is mis-computed | Counts unique timestamps (with milliseconds), so values like 146 to 192 exceed the 90-day window | Count unique calendar dates instead. Drop it from SOM inputs until fixed |
| Extreme values | `amount` ranges -3.0M to +3.5M, `balance` -566k to +9.2M | Signed-log or winsorise before any distance-based method |
| Negative amounts due or paid | `m0_total_due_with_arrears` min -597, `total_paid` min -13,311 | Check whether these are credit balances or refunds; flag them |
| Inconsistent category taxonomy | `spendgroup` (38 levels) mixes channel (`CardTransaction`) and meaning (`MoneyIn`) | Build behavioural shares from `categoryname` (103 levels), not `spendgroup` |
| `due_count` has almost no information | 98.3% of customers have 3; AUC 0.507 | Drop from the static set or treat as a minor flag |

---

## 4. The target and cohort drift

| M0 month | Customers | Target rate |
|---|---|---|
| Sep 2025 | 35,252 | 50.9% |
| Oct 2025 | 19,489 | 47.2% |
| Nov 2025 | 14,262 | 45.3% |
| Dec 2025 | 15,127 | 44.0% |
| Jan 2026 | 21,887 | **29.2%** |
| Feb 2026 | 14,407 | 39.9% |

- **Overall [Observed]:** 52,382 positives (43.5%), imbalance about 1:1.3. This is **not** a rare-event problem, so the "class imbalance" concerns raised earlier do not apply. Use ROC-AUC, PR-AUC relative to the 43.5% base rate, and calibration.
- **Out-of-time split [Derived]:** a 70/30 split by M0 puts Jan and Feb in the test set. Train rate is about 47.8% and test rate about 33.5%. Ranking metrics should hold up, but probability calibration and fixed thresholds will not.
- **Correction to my earlier advice:** I suggested `EMBARGO_DAYS = 90`. With only 6 monthly cohorts that would leave only the September cohort for training. Keep it at 0 (or 30 at most).
- **Why January dropped is unknown [Hypothesis]:** candidates are seasonality (December bonuses and January pay), a change in collections practice or the early-debit-order policy, or a label or extraction difference. Investigate before treating the drop as real behaviour.
- **Actions:**
  1. Use rolling-origin validation: train Sep to Oct and test Nov; train Sep to Nov and test Dec; and so on. Report AUC per test month.
  2. Recalibrate on recent data (for example isotonic or Platt) before using probabilities.
  3. Never use M0 month as a model feature. Track it as a monitoring variable.

---

## 5. Static (account-level) variables

| Variable | Finding | Label |
|---|---|---|
| **Collections method** | Early Debit Order, linked to clients paydate: 79,080 customers (65.7%), **51.6%** miss (CI 51.3 to 52.0). Debit Order: 39,521 customers, **28.0%**. Credit push / EFT: 1,823 customers, 26.4% | Observed |
| Predictive weight of method alone | Treating it as binary gives AUC of roughly 0.61 | Derived |
| `arrears_share_of_due` | Best static variable: AUC **0.635** (median 0.31 vs 0.27) | Observed |
| `m0_arrearsamount` | AUC 0.538 (median R745 vs R668) | Observed |
| `m0_total_due_with_arrears` | AUC 0.492 (no signal on its own) | Observed |
| `due_count` | AUC 0.507 (no signal) | Observed |

**Reading this:**
- Relative arrears (arrears as a share of what is due) is much more informative than the arrears amount. Build further ratio features of this kind.
- **Be careful with collections method.** It may be assigned to customers on the basis of perceived risk, or the early debit date may collide with the pay cycle. In either case, treat it as an operational variable and not as a customer trait. Always report model results both with and without it, and within each method group.
- The static baseline in the comparison must contain method, arrears share, total due, and the balance snapshot (see section 6). A weak baseline would make the time-aware result meaningless.

---

## 6. Behavioural signals (90 days before M0)

### 6.1 Univariate ranking
Direction is shown as "higher in missers" (AUC above 0.5) or "lower in missers" (AUC below 0.5).

| Feature | AUC | Median, missers vs payers | Reading |
|---|---|---|---|
| `bal_last` | 0.344 | R34 vs R248 | **Lower** closing balance. Strongest single feature (point-in-time) |
| `outflow_sum` | 0.392 | R55k vs R89k | Lower money throughput |
| `bal_max` | 0.406 | R9.8k vs R13.9k | Lower peak balance |
| `avg_outflow_size` | 0.407 | R293 vs R367 | Smaller payments |
| `outflow_to_inflow` | 0.410 | 1.01 vs 1.10 | Missers do **not** overspend relative to inflows |
| `bal_mean` | 0.414 | R479 vs R1,036 | Lower average balance |
| `n_txn` | 0.431 | 234 vs 307 | Less activity |
| `net_flow` | 0.588 | -R374 vs -R6,115 | Payers run larger net outflows (see 3.3) |
| `pct_neg_bal` | 0.556 | 8.8% vs 6.9% | Slightly more time overdrawn |
| `share_other` | 0.582 | 1.05% vs 0.70% | More spend in an "Other" bucket |
| `share_debitorder` | 0.448 | 5.5% vs 7.7% | Fewer committed debit orders |
| `bal_min` | 0.532 | -R10.5k vs -R11.5k | Weak |

**Reading this [Hypothesis]:** the people who miss payments look like **lower-liquidity, lower-activity customers with depleted closing balances**, not heavy overspenders. Their outflow-to-inflow ratio is actually lower. Volume features are strongly size-driven, so convert them to ratios (for example closing balance divided by average daily outflow, which is "days of cover").

### 6.2 Non-linear effects (decile plots)
- **`bal_last` is non-monotonic [Observed]:** the target rate in the lowest decile is about 42% (close to baseline), jumps to about **78% in the second decile**, then falls steadily to about 20% in the top decile.
- **Interpretation [Hypothesis]:** the lowest decile is likely overdrawn customers (possibly with an overdraft facility), while the second decile is likely balances at or near zero. Many customers have a closing balance of exactly 0.00, and the plot ranks ties arbitrarily, so exact cut-offs are not visible.
- **`outflow_sum`, `bal_max`, `avg_outflow_size` [Observed]:** these decline smoothly and then tick up slightly in the top decile (about 36% vs 33% in decile 9). That could be a different customer type (for example business-like accounts).
- **Actions:**
  1. Print the decile edges and add flags: `bal_last == 0`, `0 < bal_last < R100`, `bal_last < 0`.
  2. Prefer tree models or binned features, because linear terms will miss these shapes.
  3. Do not feed raw `bal_last` into a SOM without a transform that separates overdrawn from near-zero customers.

### 6.3 Strong redundancy among features [Observed]
Spearman correlations: `n_outflow` vs `n_txn` 0.99, `active_days` vs `n_txn` 0.98, `outflow_sum` vs `inflow_sum` 0.92, `bal_max` vs `max_inflow` 0.94, `m0_arrearsamount` vs `bal_min` -0.82.

- The last pair is unusually high. **Check what `balance` represents** (an account with overdraft, or a balance that includes a credit product). If `balance` already reflects the arrears exposure, balance features partly restate the target.
- For a SOM, pick one representative per cluster (see section 9).

---

## 7. Category-level risk signals

Customers who transacted in the category during the 90 days before M0 (overall rate 43.5%). Only categories with reasonable counts are shown.

| Category | Customers | Target rate | Lift | Possible meaning (hypothesis) |
|---|---|---|---|---|
| Insurance Payout | 191 | 69.6% | 1.60 | Claim or shock event (small group) |
| **Savings** | 14,115 | 55.1% | 1.27 | Drawing on savings |
| **UIF Received** | 5,640 | 51.8% | 1.19 | Unemployment benefit, income shock |
| Loans | 9,335 | 49.4% | 1.14 | Taking on new credit |
| Loan Payments | 30,144 | 46.1% | 1.06 | Existing debt burden |
| Credit Card Payments | 77,430 | 45.5% | 1.05 | Mild |
| Betting/Lottery | 47,142 | 44.0% | 1.01 | **No signal on presence alone** |
| UnpaidOrBouncedDebitOrder | 82,627 | 46.4% | 1.07 | Present for 68.6% of customers, so weak as a flag |
| Medical Aid | 2,325 | 31.6% | 0.73 | Stable formal obligations |
| Municipal Bill | 2,815 | 31.4% | 0.72 | Stable household |
| Bonus | 912 | 30.7% | 0.71 | Income cushion |
| StandingOrder* spend groups | 1,484 to 3,632 | 30.1% to 31.9% | 0.69 to 0.73 | Structured, planned payments |

**Reading this:**
- **Presence flags are weak when the behaviour is common.** Betting appears in 39% of customers and bounced debit orders in 69%, with lift near 1. **Intensity** (count, amount, share of spend, trend) is the better test. Build those features.
- **Stability markers** (municipal bill, medical aid, standing orders, bonus) go with lower risk. A "count of formal recurring obligations" feature is a good candidate.
- **Hardship markers** (UIF received, savings activity, new loans, insurance payout) go with higher risk. Treat them as event flags and use the timing (first appearance in the last 30 days vs earlier).

---

## 8. What this means for the core objective (time-aware vs static)

**What the data already tells us**
- Static account attributes are individually weak (best AUC 0.635), so **H1 (time-aware alone beats static alone) is likely to be easy and is not very informative**.
- The best single feature is a **balance snapshot**. Any fair "non-temporal" baseline must include it, plus time-agnostic 90-day aggregates. **H2 (time-aware features add value on top of that baseline) and H3 (the gain disappears when timing is shuffled) are the decisive tests.**
- Customers differ strongly in scale (activity, throughput), so raw sums mostly learn size. Ratio features are needed so the comparison tests behaviour over time and not volume.

**Changes to make in the second notebook version before running it on this data**
1. **Set the look-back to about 90 days.** `MAX_W = 180` and the "last 60 vs prior 120 days" trend are not supported by about 98 days of history. Use windows such as 14, 30, 60 and 90 days, and compare "last 30" against "prior 60" only.
2. **Re-anchor recency features** to the customer's last transaction date or `payment_date` (see 3.5).
3. **Make features pay-cycle aware.** The days-before-M0 histogram has regular spikes near days 30, 60 and 90 (month-end) and near days 39 and 69 (roughly the 25th). Calendar windows will contain one or two paydays depending on the customer. Add "days since last salary-like inflow", "inflow gap vs usual gap", and compare same-position windows in the pay cycle.
4. **Clean the transactions first** (duplicates, double-recorded card purchases, transfers, reversals) and compute income and spend from the cleaned set.
5. **Switch spend-mix features to `categoryname`** and add intensity features: bounced-debit-order count, insufficient-funds fee count, betting share and its trend, cash-withdrawal share, loan-payment count, UIF and savings flags.
6. **Add affordability ratios:** arrears divided by average monthly true income, closing balance divided by average daily spend, and balance-zero flags.
7. **Validation:** rolling-origin by M0 month, embargo 0 to 30 days, report AUC per test month and lift at top 10%, and recalibrate for the base-rate shift. Resolve the label definition first (3.4).
8. **Stratify the headline comparison** by collections method. If time-aware features only help in one method group, that is an important finding.

---

## 9. Suggested SOM inputs (once data is cleaned)

Use roughly 15 to 20 weakly correlated features, scaled with signed-log then robust or standard scaling. Hold out the target and the outcome-period columns (`payment_count`, `total_paid`).

| Theme | Candidate features |
|---|---|
| Liquidity | closing balance (with zero and overdrawn flags), days of cover, mean balance, share of time negative |
| Capacity | true monthly income, true monthly spend, spend to income ratio, arrears to income ratio |
| Regularity | inflow gap and gap variability, days since last salary-like inflow, weekly net-flow volatility |
| Dynamics | 30 vs prior 60 day change in income, spend and balance; weekly balance slope |
| Stress signals | bounced debit order count, insufficient-funds fee count, overdraft crossings |
| Mix and stability | betting share, cash share, loan-payment count, count of formal recurring obligations |

Cautions:
- Without ratio features the SOM will mostly organise customers by **size and activity**, which is the dominant axis in this data.
- Overlay the target rate **per node with confidence intervals** and a minimum node size, and compare maps across random seeds.
- Check whether the map simply rediscovers collections method. If so, build one map per method group.

---

## 10. Prioritised action plan

| Priority | Action | Why | Effort |
|---|---|---|---|
| **P0** | Confirm the target definition and test the amount-based alternative (3.4) | 16% label disagreement; everything downstream depends on it | Low |
| **P0** | Investigate and remove duplicates (3.1) | 24% of rows; inflates all volume features | Low to medium |
| **P0** | Confirm window anchor and `payment_date` meaning (3.5) | Affects every recency and "balance at M0" feature | Low |
| **P0** | Clean transactions: card double-recording, transfers, round-ups, reversals (3.2, 3.3) | Income and spend features are currently wrong | Medium |
| **P1** | Update the notebook: 90-day windows, categoryname shares, intensity and ratio features, `active_days` fix | Needed for a meaningful H1 to H3 test | Medium |
| **P1** | Rolling-origin validation with per-month AUC and recalibration (section 4) | Large cohort drift | Medium |
| **P1** | Investigate the January drop (29.2%) | Could be policy or extraction | Low |
| **P1** | Add decile edges and balance-zero flags (6.2) | Largest non-linear effect | Low |
| **P2** | Run the full H1 to H3 comparison, stratified by collections method | The core objective | Medium |
| **P2** | Build SOM on the cleaned, ratio-based feature set (section 9) | Segment discovery | Medium |
| **P3** | Business follow-up: review early-debit-order timing against pay dates; pilot outreach for near-zero-balance and UIF-recipient segments | Segments with 1.2 to 1.8 times baseline risk. Correlation only; test with a controlled trial | Business-led |

---

## 11. Questions for the data owner

1. What exactly defines `target_value`, and what do `payment_count` and `total_paid` count (instalments, any payment, payments across what period)?
2. What is `payment_date`: last payment attempt, scheduled due date, or something else? Why is it always on or before M0?
3. Is the transaction extract the 90 days before `payment_date` or before M0?
4. Why do 24% of rows repeat exactly, and are `CardAuthorisationTransaction` and `CardTransaction` the same purchases?
5. What does `balance` represent (current account with overdraft, or a credit product)?
6. How is the collections method assigned? Is "Early Debit Order, linked to clients paydate" a risk-based assignment?
7. Did anything change in collections practice or extraction in January 2026?

---

## Appendix: run notes
- The notebook ran end to end on all 52M rows (3.6 GB in memory) with no errors. My earlier memory estimate (100 GB or more) was far too pessimistic for this file: it came from a synthetic benchmark using wider data types, whereas your data uses compact types.
- Run time of the heaviest feature cell should be minutes at this scale. Model fitting works on a 120,424-row customer table and is fast.
- Figures [Derived] here (label mismatch split, method AUC of roughly 0.61, train and test event rates) come from arithmetic on printed outputs, not from re-running the data. Re-check them in the notebook when you rerun.
