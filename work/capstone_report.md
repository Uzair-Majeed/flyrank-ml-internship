# Capstone Report

* Author: Uzair
* Lane: Content Refresh / Opportunity Scoring
* Repo: <paste your repo URL here>
* Date: 2026-08-07

## 0. Abstract

Content editors can't manually audit thousands of pages a week to decide what needs a refresh, so most rely on brittle age-and-CTR rules. Using 30,000 content assets across 32 clients from a March 2026 search-performance release, a Random Forest was trained on five decision-time-knowable features to rank pages by refresh urgency and validated against a client-grouped holdout to prevent leakage. Against the same split, the model reached 70% precision in its top 20 picks and 68% in its top 50, versus 20% and 34% for a rule baseline and a 24.5% base rate — a ~2.9x gain in how many flagged pages actually needed attention. The output is a ranked, decision-support queue for editors, not proof that refreshing a page recovers its traffic.

## 1. Problem framing

**Unit of analysis:** one content asset (`content_id`), aggregated over its trailing 90-day performance window.
**Output:** a refresh-urgency score per page, converted into a rank and one of four reason codes.
**Action a human takes:** a content editor opens the ranked queue and works top-down, refreshing pages under the recommended reason code (e.g. rewrite title/meta for `STALE_LOW_CTR`).
**Cost of a wrong call:** a false positive wastes a writer's hours refreshing a page that didn't need it. A false negative lets a genuinely decaying page keep losing rank and traffic until a competitor takes it, unnoticed.
**Why ML helps:** editors can't manually check thousands of pages daily, and hardcoded rules (e.g. "impressions dropped >20%") miss multi-signal, non-linear interactions — a page can have stable impressions but quietly dropping scroll depth or CTR. A model can weigh all these signals jointly and rank the entire inventory automatically.

## 2. Data safety

**Used:** `fact_content_daily_performance`, month=2026-03. Full release: 331,437 unique content items, 9,841,378 daily rows. Working sample after filtering (`impressions_90d > 0`, `content_age_days >= 90`): 30,000 assets, 32 clients.

**Deliberately excluded:**
- `trend_direction`, `trend_pct`, `is_declining_label` — these define the target proxy; including them as features would be direct label leakage. Confirmed absent from the feature matrix by explicit check.
- `client_id`, `content_id` — used only to group the train/test split, never as predictive features.
- Long-tail, low-volume queries — redacted by Google Search Console's privacy thresholding before this data ever arrives; can't be recovered.

**Leakage risks considered:** the most consequential one was client identity crossing between train and test under a naive random split — clients share domain authority and baseline traffic, so a model can partly memorize "this looks like client X's page" rather than learn transferable decay signal. A grouped split by `client_id` closes this. A synthetic leaked-feature "trap" experiment (adding a column derived from the target) was also run and correctly produced an unrealistic AUC ≈ 1.0, confirming the leakage check itself works.

**Confirmation:** no client names, URLs, or raw query text appear anywhere in `work/` — every identifier is a hashed `content_id` / `client_id`.

## 3. Baseline

**The rule:** flag a page if it's older than 180 days *and* has under 2% CTR despite meaningful search views. Outputs one of four reason codes (`STALE_LOW_CTR`, `STALE_CONTENT`, `MONITOR`, `HEALTHY`), so every pick is auditable in plain language — no black box.

**Why it's a fair comparison:** it's evaluated on the exact same held-out, client-grouped test set and the same Precision@K metrics as the model, with no advantages given to either side.

**Its numbers:** Precision@20 = 20.0%, Precision@50 = 34.0%, PR-AUC = 0.242 (base rate 24.5%). Note the rule scores *below* the 24.5% base rate at K=20 — its strict cutoff means its very top picks aren't necessarily its most confident ones. Reported as-is rather than smoothed over.

## 4. Model / analysis

**Method:** Logistic Regression first (sanity check — interpretable coefficients catch a model leaning on a spurious signal early), then Random Forest (captures non-linear, multi-feature interactions the rule and linear model both miss). Random Forest fits the lane because refresh-worthiness plausibly depends on interactions between age, volume, and engagement rather than any single threshold.

**Feature list (5):** `impressions_90d`, `clicks_90d`, `ctr_decimal`, `avg_position_clean`, `sessions_90d` — all knowable at decision time.
**Left out on purpose:** `trend_direction`/`trend_pct` (label-derived, leakage), `client_id`/`content_id` (grouping only), scroll-depth and engagement-rate fields explored in early framing but not carried into the final 5-feature model.

**Target, one sentence:** `target_needs_refresh` is a constructed binary proxy — 1 if a page shows established search demand combined with a measured downward performance trend, 0 otherwise; it is a proxy for decay, not a hand-audited editorial label.

## 5. Evaluation

**Split:** `GroupShuffleSplit` grouped by `client_id`, 80/20, `random_state=42`. Grouped rather than time-aware, because the risk in this data is client-identity leakage (shared domain authority/traffic baseline), not a temporal ordering effect — the sample is a single-month snapshot. Train: 23,837 rows (35.5% positive). Test: 6,163 rows (**base rate 24.5%**).

**Metrics vs. base rate:**

| Method | Precision@20 | Precision@50 | PR-AUC |
|---|---|---|---|
| Base rate (random) | 24.5% | 24.5% | 0.245 |
| Rule baseline | 20.0% | 34.0% | 0.242 |
| Logistic Regression | 40.0% | 38.0% | 0.258 |
| **Random Forest** | **70.0%** | **68.0%** | **0.583** |

Against a 24.5% base rate, the Random Forest's 70% Precision@20 is a genuine ~2.9x lift, not an artifact of an easy majority class. Its PR-AUC (0.583) is likewise well above the base-rate floor (0.245).

**The split mattered more than the model:** the same Random Forest scored 95% Precision@20 under a naive row-level random split — a 25-point leakage artifact from client identity crossing train/test. The honest, client-grouped number is the one reported above.

**Error analysis:**
- *False positives:* older, high-impression pages the model flags as decaying but are actually stable. Without a time-series trend feature (withheld to avoid leakage), the model reasonably but incorrectly reads "old + high volume" as decay.
- *False negatives:* pages genuinely declining but with relatively low raw impression volume or an unexpectedly high CTR, which reads as healthy to the model.

## 6. Interpretation

Permutation importance on the honest split: `impressions_90d` dominates (importance 0.224), with `ctr_decimal`, `sessions_90d`, and `clicks_90d` trailing roughly an order of magnitude lower, and `avg_position_clean` further behind. No feature crossed the 0.40 leakage-flag threshold — consistent with the explicit leakage audit passing.

In plain words: the model is mostly a function of *how much search visibility a page already has*, modulated by how well it's converting that visibility into clicks. Position matters less than expected, which is a mild negative result against the intuition that ranking position would dominate — instead, raw impression volume carries most of the signal.

## 7. Recommendation

Four archetypes map model output to editor action:

| Archetype | Reason code | Action |
|---|---|---|
| High-value decay | `RC_FULL_REFRESH` | Full content & title refresh, top priority |
| Stale, low CTR | `STALE_LOW_CTR` | Rewrite title & meta first |
| Stale content | `STALE_CONTENT` | Update facts & sources, lighter touch |
| Monitor | `MONITOR` | Watch next cycle, no action yet |

Tomorrow, an editor opens the ranked queue, works top-down under their weekly capacity, and acts on the reason code rather than a bare score. 15,086 of the 30,000 pages are flagged for **mandatory human sign-off** before any action (high-impression/high-priority cases); the remaining 14,914 fall into standard monitoring. The system never auto-deletes content or auto-publishes AI-generated rewrites.

**Confidence and limits, stated plainly:** this is directional and decision-support — a prioritization aid under fixed editorial capacity, not proof that acting on a flag recovers traffic. It should not override editorial judgment, only order the queue it works from.

## 8. Reproducibility

**Re-run from a fresh clone:**
```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute work/notebooks/w01_research_question.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w02_ml_task_framing.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w03_data_contract.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w04_baseline_score.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w06_validation_audit.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w07_action_playbook.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb
```
**Seed:** `random_state=42` used consistently for the `GroupShuffleSplit` and the Random Forest.
**Environment:** pin `pandas`, `numpy`, `scikit-learn`, `duckdb` versions in `requirements.txt` — run `pip freeze > requirements.txt` in your working environment and commit it; note here any version that materially changed a result (e.g. a `scikit-learn` major-version bump changing default RF parameters).
**Sealed/holdout evaluation:** the client-grouped test set in `w05_model.ipynb`/`w06_validation_audit.ipynb` functions as the holdout — it is built by the `GroupShuffleSplit` cell in those notebooks and its resulting metrics are written to `work/metrics/playbook_metrics.json`. Both the split-building cell and that metrics file are committed, so "evaluated once, honestly" is checkable directly rather than taken on faith. *(If you additionally run a fully sealed, touch-once holdout, commit that script and its output metrics file the same way and reference them here by path.)*

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** — [flyrank.ai](https://flyrank.ai).
