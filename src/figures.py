"""Step 4 - Report figures (static, print-ready, colour-blind-safe palette)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT / "data" / "processed", ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 7,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": .6,
    "legend.frameon": False, "legend.fontsize": 6.5, "savefig.dpi": 300,
})

d21 = pd.read_csv(DATA / "lr21_matches.csv")
d22 = pd.read_csv(DATA / "lr22_team_matches.csv")
res = json.loads((OUT / "results.json").read_text())
cor = pd.read_csv(OUT / "eda_correlations.csv", dtype={"task": str})

LABEL = {"fifa_pts_diff": "FIFA points diff.", "form_gf_diff": "Form: goals for diff.",
         "form_ga_diff": "Form: goals against diff.", "form_win_diff": "Form: win-rate diff.",
         "host_diff": "Host nation diff.", "wc_exp_diff": "WC experience diff.",
         "rest_diff": "Rest days diff.", "tourn_gd_diff": "Tournament GD so far diff.",
         "team_attack_form": "Attack form × opp. defence",
         "opp_defence_form": "Opponent goals-against form", "is_host": "Is host nation",
         "knockout": "Knockout stage", "team_tourn_gf_rate": "Tournament scoring rate"}

# ===================================================================== Fig 1 EDA
fig, ax = plt.subplots(2, 2, figsize=(6.9, 4.5), constrained_layout=True)

a = ax[0, 0]
rng = np.random.default_rng(1)
for st, c in [("Group", BLUE), ("Knockout", ORANGE)]:
    s = d21[np.where(d21.match_id <= 72, "Group", "Knockout") == st]
    a.scatter(s.fifa_pts_diff, s.goal_diff + rng.uniform(-.15, .15, len(s)), s=10, color=c,
              alpha=.8, edgecolor="white", linewidth=.4, label=f"{st} (n={len(s)})")
sl = stats.linregress(d21.fifa_pts_diff, d21.goal_diff)
xs = np.linspace(d21.fifa_pts_diff.min(), d21.fifa_pts_diff.max(), 100)
n = len(d21); xm = d21.fifa_pts_diff.mean(); sxx = ((d21.fifa_pts_diff - xm) ** 2).sum()
resid_se = np.sqrt(((d21.goal_diff - (sl.intercept + sl.slope * d21.fifa_pts_diff)) ** 2).sum() / (n - 2))
band = stats.t.ppf(.975, n - 2) * resid_se * np.sqrt(1 / n + (xs - xm) ** 2 / sxx)
yh = sl.intercept + sl.slope * xs
a.fill_between(xs, yh - band, yh + band, color=INK, alpha=.08, linewidth=0)
a.plot(xs, yh, color=INK, lw=1.2)
a.text(.02, .95, f"r = {sl.rvalue:.2f}, p < .001\n+100 FIFA pts ≈ +{sl.slope*100:.2f} goals",
       transform=a.transAxes, va="top", fontsize=6.5, color=INK)
a.set(xlabel="FIFA ranking points difference, 11 Jun 2026 (team 1 − team 2)", ylabel="Goal difference",
      title="(a) FIFA points gap vs goal difference")
a.legend(loc="lower right")

from matplotlib.colors import LinearSegmentedColormap
DIVERGE = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f1f0ec", "#e34948"])
SHORT = {"fifa_pts_diff": "FIFA pts", "form_gf_diff": "Form GF", "form_ga_diff": "Form GA",
         "form_win_diff": "Win rate", "host_diff": "Host", "wc_exp_diff": "WC exp.",
         "rest_diff": "Rest", "tourn_gd_diff": "Tourn. GD", "goal_diff": "GOAL DIFF."}

a = ax[0, 1]
cols21 = ["goal_diff", "fifa_pts_diff", "form_gf_diff", "form_ga_diff", "form_win_diff",
          "host_diff", "wc_exp_diff", "rest_diff", "tourn_gd_diff"]
cm = d21[cols21].corr().values
mask = np.triu(np.ones_like(cm, dtype=bool), k=0)
a.imshow(np.ma.array(cm, mask=mask), cmap=DIVERGE, vmin=-1, vmax=1)
for i in range(len(cols21)):
    for j in range(i):
        a.text(j, i, f"{cm[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center",
               va="center", fontsize=5.3, color=INK if abs(cm[i, j]) < .6 else "white")
a.set_xticks(range(len(cols21) - 1)); a.set_yticks(range(1, len(cols21)))
a.set_xticklabels([SHORT[c] for c in cols21[:-1]], rotation=45, ha="right", fontsize=5.8)
a.set_yticklabels([SHORT[c] for c in cols21[1:]], fontsize=5.8)
a.set_xlim(-.5, len(cols21) - 1.5); a.set_ylim(len(cols21) - .5, .5)
a.grid(False)
for sp in a.spines.values():
    sp.set_visible(False)
a.set_title("(b) Correlation matrix, LR 2.1")

a = ax[1, 0]
lam = d22.goals.mean(); k = np.arange(0, 8)
obs = np.bincount(d22.goals, minlength=8)[:8]
a.bar(k, obs, color=BLUE, width=.7, label="Observed", zorder=2)
a.plot(k, stats.poisson.pmf(k, lam) * len(d22), "o-", color=ORANGE, ms=4, lw=1.2,
       label=f"Poisson(λ={lam:.2f})", zorder=3)
e = res["eda"]
a.text(.98, .62, f"mean {e['goals_mean_ci'][0]:.2f} [95% CI {e['goals_mean_ci'][1]:.2f}–"
       f"{e['goals_mean_ci'][2]:.2f}]\nmedian 1, skew 1.14\n"
       f"χ² goodness-of-fit p = {e['poisson_gof_p']:.2f}", transform=a.transAxes, ha="right",
       va="top", fontsize=6.5, color=INK)
a.set(xlabel="Goals scored by a team in a match", ylabel="Team-matches",
      title="(c) Distribution of team goals (LR 2.2 response)")
a.legend(loc="upper right")

a = ax[1, 1]
ht = pd.read_csv(OUT / "hypothesis_tests.csv")
groups = [d22[d22.knockout == 0].goals, d22[d22.knockout == 1].goals,
          d22[d22.is_host == 0].goals, d22[d22.is_host == 1].goals]
pos = [0, 1, 2.6, 3.6]
bp = a.boxplot(groups, positions=pos, widths=.55, patch_artist=True, showmeans=True,
               medianprops=dict(color=INK, lw=1.1),
               meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=INK, ms=3.5),
               flierprops=dict(marker="o", ms=2.5, markerfacecolor=GREY, markeredgecolor="none"),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED), boxprops=dict(lw=0))
for patch, col in zip(bp["boxes"], [BLUE, ORANGE, BLUE, ORANGE]):
    patch.set_facecolor(col); patch.set_alpha(.75)
a.set_xticks(pos)
a.set_xticklabels([f"Group\n(n={len(groups[0])})", f"Knockout\n(n={len(groups[1])})",
                   f"Other\n(n={len(groups[2])})", f"Host at home\n(n={len(groups[3])})"], fontsize=6)
p_stage = ht.loc[ht.question.str.contains("knockout"), "p"].iloc[0]
p_host = ht.loc[ht.question.str.contains("host"), "p"].iloc[0]
a.text(.5, 6.9, f"Welch t-test p = {p_stage:.2f}", ha="center", fontsize=6.2, color=INK)
a.text(3.1, 6.9, f"Welch t-test p = {p_host:.2f}", ha="center", fontsize=6.2, color=INK)
a.set_ylim(-.4, 7.6)
a.set(ylabel="Goals scored by a team", title="(d) Goals by stage and host status (◇ = mean)")
a.grid(axis="x", visible=False)
fig.savefig(FIG / "fig1_eda.png")
plt.close(fig)

# ===================================================================== Fig 2 models
VIOLET = "#4a3aa7"
FAM_COL = {"Linear": BLUE, "Non-linear": AQUA, "Ensemble": VIOLET, "Baseline": GREY}
fig, ax = plt.subplots(1, 2, figsize=(6.9, 3.2), constrained_layout=True)
for a, task, title in [(ax[0], "2.1", "(a) LR 2.1: goal difference"),
                       (ax[1], "2.2", "(b) LR 2.2: team goals")]:
    s = pd.DataFrame(res[task]["summary"]).sort_values("rmse", ascending=False)
    y = np.arange(len(s))
    best = res[task]["best"]
    col = [FAM_COL[f] for f in s.family]
    a.errorbar(s.rmse, y, xerr=s.rmse_ci95, fmt="none", ecolor=col, elinewidth=1.2)
    a.scatter(s.rmse, y, c=col, s=22, zorder=3)
    a.scatter(s.rmse_knockout, y, facecolor="white", edgecolor=INK, s=16, lw=.9, zorder=3,
              marker="D")
    def disp(m):
        return (m.replace("OLS + eng + symmetric", "OLS + transformed + both orders")
                 .replace("Ridge + eng + symmetric", "Ridge + transformed + both orders")
                 .replace("OLS log(1+y) + eng", "OLS on log(1 + goals)")
                 .replace("OLS + engineered", "OLS + transformed").replace(" (nested CV)", "")
                 .replace("Forward-selected OLS", "Forward selection OLS")
                 .replace("OLS (8 raw)", "Plain OLS (untransformed)"))
    a.set_yticks(y); a.set_yticklabels([disp(m) for m in s.model], fontsize=6.2)
    for lbl, m in zip(a.get_yticklabels(), s.model):
        if m == best:
            lbl.set_fontweight("bold"); lbl.set_color(INK)
            lbl.set_text("★ " + m)
    a.set_yticklabels([("★ " + disp(m)) if m == best else disp(m) for m in s.model], fontsize=6.2)
    for lbl, m in zip(a.get_yticklabels(), s.model):
        if m == best:
            lbl.set_fontweight("bold"); lbl.set_color(INK)
    a.set(xlabel="RMSE (goals) - lower is better", title=title)
    a.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="none", color=FAM_COL[k], label=k + (" regression" if k != "Baseline" else " (mean)"))
     for k in ["Linear", "Non-linear", "Ensemble", "Baseline"]]
h.append(Line2D([], [], marker="D", ls="none", markerfacecolor="white", markeredgecolor=INK,
                label="Knockout hold-out RMSE"))
fig.legend(handles=h, loc="outside lower center", ncol=5, fontsize=6.3)
fig.savefig(FIG / "fig2_models.png")
plt.close(fig)

# ===================================================================== Fig 3 interpretation + assumptions
import statsmodels.api as sm
from modelling import X21, X22, engineer_21, engineer_22

fig = plt.figure(figsize=(6.9, 4.0), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
for a, task, col, title in [(ax[0], "2.1", BLUE, "(a) LR 2.1: effect of each variable"),
                            (ax[1], "2.2", AQUA, "(b) LR 2.2: effect of each variable")]:
    c = pd.DataFrame(res[task]["coef_eng"]).rename(columns={"index": "f"}).sort_values("coef")
    y = np.arange(len(c))
    sig = c.p < .05
    a.errorbar(c.coef, y, xerr=[c.coef - c.lo, c.hi - c.coef], fmt="none", ecolor=col, elinewidth=1)
    a.scatter(c.coef[sig], y[sig], color=col, s=20, zorder=3, label="p < .05")
    a.scatter(c.coef[~sig], y[~sig], facecolor="white", edgecolor=col, s=20, zorder=3,
              label="not significant")
    a.axvline(0, color=GREY, lw=.8)
    a.set_yticks(y); a.set_yticklabels([LABEL[f] for f in c.f], fontsize=6)
    a.set(xlabel="Change in goals per +1 SD (95% CI)", title=title)
    a.grid(axis="y", visible=False)
ax[0].legend(loc="lower right", fontsize=6)

fits = {}
for task, d, X, yv, eng in [("2.1", d21, X21, "goal_diff", engineer_21),
                            ("2.2", d22, X22, "goals", engineer_22)]:
    Z = eng(d[X]); Z = (Z - Z.mean()) / Z.std(ddof=0)
    fits[task] = sm.OLS(d[yv].values.astype(float), sm.add_constant(Z)).fit()

a = ax[2]
for task, col, mk in [("2.1", BLUE, "o"), ("2.2", AQUA, "s")]:
    f = fits[task]
    a.scatter(f.fittedvalues, f.resid, s=8, color=col, alpha=.75, edgecolor="white", lw=.3,
              marker=mk, label=f"LR {task}  (Breusch–Pagan p = {res[task]['diag_eng']['breusch_pagan_p']:.2f})")
a.axhline(0, color=INK, lw=.8, ls="--")
a.set(xlabel="Fitted value (goals)", ylabel="Residual (goals)",
      title="(c) Residuals vs fitted values")
a.legend(loc="upper left", fontsize=5.8, frameon=True, facecolor="white", edgecolor="none", framealpha=.9)

a = ax[3]
for task, col, mk in [("2.1", BLUE, "o"), ("2.2", AQUA, "s")]:
    r = np.sort(fits[task].resid / fits[task].resid.std())
    q = stats.norm.ppf((np.arange(1, len(r) + 1) - .5) / len(r))
    a.scatter(q, r, s=8, color=col, alpha=.75, edgecolor="white", lw=.3, marker=mk,
              label=f"LR {task}  (Shapiro–Wilk p = {res[task]['diag_eng']['shapiro_resid_p']:.3f})"
              .replace("p = 0.000", "p < .001"))
a.plot([-3, 3], [-3, 3], color=INK, lw=.8, ls="--")
a.set(xlabel="Theoretical normal quantile", ylabel="Standardised residual",
      title="(d) Normal Q-Q plot of residuals")
a.legend(loc="upper left", fontsize=5.8, frameon=True, facecolor="white", edgecolor="none", framealpha=.9)
fig.savefig(FIG / "fig3_interpretation.png")
plt.close(fig)
print("figures written")
