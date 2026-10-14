# HIT140 Darwin Group 56 – FIFA World Cup 2026, Objective 2: Linear Regression 2.1 & 2.2

Predicting World Cup 2026 results using **only pre-match information**.

| Task   | Target                             | Rows             | Recommended model                                | CV RMSE | CV R² |
| ------ | ---------------------------------- | ---------------- | ------------------------------------------------ | ------- | ------ |
| LR 2.1 | Goal difference (team 1 − team 2) | 104 matches      | Ridge + transformed variables + both team orders | 1.554   | 0.38   |
| LR 2.2 | Goals scored by a team             | 208 team-matches | Lasso + transformed variables                    | 1.213   | 0.23   |

## Reproduce everything

```bash
pip install -r requirements.txt   # exact versions used for the report (Python 3.13)
python run_all.py          # parse files -> datasets -> stats -> models -> Week 9 assumptions -> class workflow -> figures
```

## Reproducibility

All randomness uses fixed seeds (CV splits `SEED = 2026`, model `random_state`s, the 60/40 class split `random_state=0`),
so every run of `run_all.py` reproduces the same numbers and the same three figures. With the pinned versions in `requirements.txt` the figures are byte-for-byte identical to those in the report (checked with two clean runs from the raw files).

## Competing algorithms (18 per task incl. baseline, identical match-level splits, nested CV tuning)

- **Linear regression family:** OLS (raw), OLS + Yeo-Johnson, OLS + drop collinear variable, OLS + engineered, forward-selected OLS, Ridge, Lasso,
  Elastic Net, Huber, symmetric-augmented OLS/Ridge (2.1), Poisson GLM and log-target OLS (2.2)
- **Non-linear regression:** polynomial (degree 2) + Ridge, SVR (RBF kernel), k-NN, decision tree
- **Ensemble regression:** random forest, gradient boosting
- **Recommendation rule (fixed in advance):** best linear-family model unless another model is
  significantly better (Nadeau–Bengio corrected resampled t-test, p < .05)

## Data collection (manual, no scraping)

Every input table was collected by hand by the group from FBref (a recommended source)
and FIFA's official ranking page. FBref tables were exported with
*Share & Export -> Get table as CSV (for Excel)*, pasted into Excel and saved as CSV UTF-8.

| Folder / file                                        | What                                                     | Files              | Used for                                            |
| ---------------------------------------------------- | -------------------------------------------------------- | ------------------ | --------------------------------------------------- |
| `data/raw/fbref_wc2026.csv`                        | FBref World Cup 2026 Scores & Fixtures                   | 1 (104 matches)    | targets, stage, host venue, tournament-so-far stats |
| `data/raw/fbref_teams/team_<Team>_<2025/2026>.csv` | FBref national-team Scores & Fixtures, both seasons      | 96 (48 teams × 2) | form (last 10 matches), rest days                   |
| `data/raw/fbref_worldcups/wc_<year>.csv`           | FBref World Cup Scores & Fixtures 1998–2022             | 7 (448 matches)    | World Cup experience                                |
| `data/raw/fifa/fifa_ranking_2026-06-11.csv`        | FIFA/Coca-Cola Men's World Ranking, 11 June 2026 release | 1 (211 teams)      | team strength                                       |

FBref splits a national team's matches over two season pages ("2026" = the World Cup cycle
incl. qualifiers and the finals; "2025" = other matches, e.g. autumn 2025 friendlies), so both
are needed to obtain each team's true last 10 matches.

**Checks** (`src/parse_fbref.py`, `src/check_team_files.py`):

- every team file's World Cup 2026 rows match that team's fixtures (catches mis-named files);
- all 208 World Cup 2026 team-results in the 96 team files agree with the fixtures table;
- every team has ≥ 10 matches before every World Cup match; no imputation is used.
- rest-day outliers are capped with the IQR method (Q1 = 5, Q3 = 7 → fences 2–10 days; 7 values capped).
- Known limitation: Cabo Verde, Iran, Qatar and Uzbekistan have no FBref-listed matches between
  Oct 2025 and late May 2026, so their last-10 window spans a longer period.

## Repository layout

```
data/raw/                 the manually collected files above
data/processed/           tidy tables + lr21_matches.csv (104 x 8 X), lr22_team_matches.csv (208 x 8 X)
src/parse_fbref.py        reads every export (CSV/XLSX, Excel line-quoting) into tidy tables + cross-checks
src/check_team_files.py   per-team quality report for the national-team files
src/names.py              one canonical name per country across FBref and FIFA spellings
src/build_datasets.py     pre-match feature construction (no leakage)
src/descriptives.py       levels of measurement, descriptive statistics, IQR outliers, 95% CIs (mean, proportion)
                          + 4-step hypothesis tests (alpha = .05)
src/modelling.py          EDA statistics, competing models, nested repeated CV, temporal hold-out, inference
src/assumptions.py        Week 9 assumptions tested one at a time (outliers/IQR, linearity, collinearity, Yeo-Johnson, rescaling)
src/class_workflow.py     the Week 8-9 practical workflow (60/40 split, MAE/MSE/RMSE/NRMSE/R2 vs mean baseline, statsmodels summary)
src/figures.py            report figures
outputs/                  CV fold scores, summaries, coefficients, results.json
                          (modelling_log.txt, assumptions_log.txt = console output of the last full run)
figures/                  fig1_eda.png, fig2_models.png, fig3_interpretation.png
```

## Variables (exactly 8 per task; 3 shared ≤ 4 allowed)

**LR 2.1** (all team 1 − team 2): `fifa_pts_diff`, `form_gf_diff`, `form_ga_diff`, `form_win_diff`,
`host_diff`, `wc_exp_diff`, `rest_diff`, `tourn_gd_diff`

**LR 2.2**: `fifa_pts_diff`, `wc_exp_diff`, `rest_diff` (shared) + `team_attack_form`,
`opp_defence_form`, `is_host`, `knockout`, `team_tourn_gf_rate`

Every variable uses only information available before kick-off: the FIFA ranking published on
11 June 2026 (no update was issued during the tournament), matches dated strictly before the match,
and World Cup matches 1998–2022. Scores include extra time; penalty shoot-outs are excluded.

## Team contributions & decisions log

| Member                   | Student ID | Data collection                                                                                                | Code owned                                                                                       | Report section                                      | Other                                                                                                              |
| ------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Ashok Tamang (Member 1)  | S406128    | FBref files for 12 teams; World Cup 2026 fixtures; World Cups 1998–2022; FIFA ranking; compiled all raw files | `src/names.py`, `src/parse_fbref.py`, `src/check_team_files.py`, `src/build_datasets.py` | 1 Introduction, Table 1, 2 Methodology              | GitHub repo, project board and Issues; final clean run of`run_all.py` from the GitHub repo; final submission zip |
| Rajesh Basnet (Member 2) | S404205    | FBref files for 12 teams                                                                                       | `src/descriptives.py`, `src/assumptions.py`                                                  | 3 Exploratory data analysis (Tables 2–4, Figure 1) | Turnitin report; AI Usage Declaration signatures                                                                   |
| Ayun Neupane (Member 3)  | S406923    | FBref files for 12 teams                                                                                       | `src/modelling.py`                                                                             | 4 Linear Regression 2.1 (Figure 2)                  | –                                                                                                                 |
| Aryan Karki (Member 4)   | S407507    | FBref files for 12 teams                                                                                       | `src/class_workflow.py`, `src/figures.py`, `run_all.py`                                    | 5 Linear Regression 2.2 (Figure 3), 6 Conclusion    | –                                                                                                                 |

### Key analysis decisions

| Decision                                                      | Reason                                                               | Evidence                                         |
| ------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------ |
| Use only FBref and FIFA data, collected by hand               | Recommended sources in the brief; no third-party copies, no scraping | `data/raw/`                                    |
| Keep both FBref season pages (2025 + 2026) for each team      | Needed to get every team's true last 10 matches                      | `src/check_team_files.py`                      |
| Exactly 8 variables per model, 3 shared                       | Brief: exactly 8, at most 4 shared                                   | Report Table 1                                   |
| Cap rest days at the IQR fences (2–10 days)                  | Week 6/9 outlier rule; biggest single improvement                    | `outputs/assumptions.csv`                      |
| 20 × 5-fold cross-validation instead of one 60/40 split      | One 60/40 split gave RMSE anywhere from 1.20 to 2.06                 | `outputs/class_workflow.json`                  |
| Recommendation rule fixed before seeing results               | Avoids picking the winner after the fact                             | `src/modelling.py` (`recommend`)             |
| Keep weak but sensible variables; let Ridge/Lasso shrink them | Collinearity (WC experience vs FIFA points, r = 0.82)                | Report Table 4,`outputs/lasso_stability_*.csv` |

Task progress is tracked in this repository's GitHub Issues (#1–#11) and project board.
