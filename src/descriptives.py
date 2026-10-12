"""
Step 2b - Descriptive statistics (Week 2) and formal hypothesis tests (Weeks 3-4).

Outputs
  outputs/descriptives.csv     level of measurement, n, mean, median, mode, SD, IQR, min, max,
                               skewness and IQR-rule outlier count per variable
  outputs/hypothesis_tests.csv H0 / H1, test, statistic, p-value, decision (alpha = .05)
  outputs/confidence_intervals.csv  95% CIs for a mean and a proportion (Week 3 practical methods)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.stats.proportion as stm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data" / "processed", ROOT / "outputs"
ALPHA = 0.05

d21 = pd.read_csv(P / "lr21_matches.csv")
d22 = pd.read_csv(P / "lr22_team_matches.csv", parse_dates=["date"])
fifa = pd.read_csv(P / "fifa_ranking.csv").set_index("team").points
past = pd.read_csv(P / "worldcups_1998_2022.csv")
tm = pd.read_csv(P / "team_matches.csv", parse_dates=["date"])
teams = sorted(set(d22.team))

# team-level rest days before each World Cup match (raw, before IQR capping)
rest = []
for r in d22.itertuples():
    prev = tm[(tm.team == r.team) & (tm.date < r.date)].date.max()
    rest.append((r.date - prev).days)
wc_matches = pd.concat([past.home, past.away]).value_counts()

# (name, values, Week 2 level of measurement)
VARS = [
    ("Goal difference (LR 2.1 response)", d21.goal_diff, "Interval, discrete"),
    ("Goals scored by a team (LR 2.2 response)", d22.goals, "Ratio, discrete (count)"),
    ("FIFA ranking points, 11 Jun 2026 (per team)", fifa[teams], "Interval, continuous"),
    ("Goals scored per match, last 10", d22.team_attack_form, "Ratio, continuous"),
    ("Goals conceded per match, last 10", d22.opp_defence_form, "Ratio, continuous"),
    ("World Cup matches 1998-2022 (per team)", pd.Series([wc_matches.get(t, 0) for t in teams]),
     "Ratio, discrete (count)"),
    ("Rest days before a match (raw)", pd.Series(rest), "Ratio, discrete"),
    ("Host at home (1/0)", d22.is_host, "Nominal (binary)"),
]


def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75])
    return int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())


desc = pd.DataFrame([{
    "variable": name, "level": lvl, "n": len(s), "mean": s.mean(), "median": s.median(),
    "mode": s.mode().iloc[0] if "discrete" in lvl or "Nominal" in lvl else np.nan, "sd": s.std(), "iqr": s.quantile(.75) - s.quantile(.25),
    "min": s.min(), "max": s.max(), "skew": stats.skew(s),
    "iqr_outliers": iqr_outliers(s) if "Nominal" not in lvl else np.nan}
    for name, s, lvl in VARS])
desc.to_csv(OUT / "descriptives.csv", index=False)
print(desc.round(2).to_string(index=False))

# ---------------------------------------------------------------- hypothesis tests
tests = []


def add(question, h0, h1, test, stat, p):
    tests.append({"question": question, "H0": h0, "H1": h1, "test": test,
                  "statistic": stat, "p": p,
                  "decision": "Reject H0" if p < ALPHA else "Fail to reject H0"})


# 1. Does the higher-ranked team win by more than zero goals on average?
fav = np.where(d21.fifa_pts_diff > 0, d21.goal_diff, -d21.goal_diff)
t = stats.ttest_1samp(fav, 0, alternative="greater")
add("Higher-ranked team's winning margin", "mean margin = 0", "mean margin > 0",
    "one-sample t-test", f"t({len(fav)-1}) = {t.statistic:.2f}", t.pvalue)

# 2. Is the FIFA points gap correlated with goal difference?
r, p = stats.pearsonr(d21.fifa_pts_diff, d21.goal_diff)
add("FIFA points gap vs goal difference", "ρ = 0", "ρ ≠ 0", "Pearson correlation test",
    f"r = {r:.2f}", p)

# 3. Do knockout matches produce fewer goals per team than group matches?
g, k = d22[d22.knockout == 0].goals, d22[d22.knockout == 1].goals
t = stats.ttest_ind(g, k, equal_var=False)
add("Goals: group vs knockout", "μ_group = μ_knockout", "μ_group ≠ μ_knockout",
    "Welch two-sample t-test", f"t = {t.statistic:.2f}", t.pvalue)

# 4. Do host nations playing at home score more?
h, nh = d22[d22.is_host == 1].goals, d22[d22.is_host == 0].goals
t = stats.ttest_ind(h, nh, equal_var=False)
add("Goals: host at home vs others", "μ_host = μ_other", "μ_host ≠ μ_other",
    "Welch two-sample t-test", f"t = {t.statistic:.2f}", t.pvalue)

# 5. Are team goals Poisson distributed?
lam = d22.goals.mean()
obs = np.bincount(np.minimum(d22.goals, 4), minlength=5)
exp = np.array([stats.poisson.pmf(i, lam) for i in range(4)] + [stats.poisson.sf(3, lam)]) * len(d22)
c = stats.chisquare(obs, exp, ddof=1)
add("Team goals follow a Poisson distribution", "goals ~ Poisson", "goals not Poisson",
    "χ² goodness-of-fit", f"χ²(3) = {c.statistic:.2f}", c.pvalue)

ht = pd.DataFrame(tests)
ht.to_csv(OUT / "hypothesis_tests.csv", index=False)
print(ht.to_string(index=False))

# ---------------------------------------------------------------- confidence intervals (Week 3)
# 95% CI of a mean (z interval, as in ci_mean.py) and of a proportion (normal method, as in ci_prop.py)
z = stats.norm.ppf(1 - ALPHA / 2)
se = fav.std(ddof=1) / np.sqrt(len(fav))
wins = int((fav > 0).sum())
p_low, p_upp = stm.proportion_confint(wins, len(fav), alpha=ALPHA, method="normal")
ci = pd.DataFrame([
    {"quantity": "Higher-ranked team's mean winning margin (goals)", "n": len(fav),
     "estimate": fav.mean(), "ci_low": fav.mean() - z * se, "ci_upp": fav.mean() + z * se},
    {"quantity": "Proportion of matches won by the higher-ranked team", "n": len(fav),
     "estimate": wins / len(fav), "ci_low": p_low, "ci_upp": p_upp},
])
ci.to_csv(OUT / "confidence_intervals.csv", index=False)
print(ci.round(3).to_string(index=False))
