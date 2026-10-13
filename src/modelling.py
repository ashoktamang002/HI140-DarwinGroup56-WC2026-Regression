"""
Step 2 - EDA, inferential statistics, competing regression models and
cross-validation for Linear Regression 2.1 and 2.2.

Design choices
* All models are evaluated on IDENTICAL resampling splits so that their errors
  can be compared with a paired test (Nadeau & Bengio corrected resampled t-test).
* Splits are made at MATCH level for both tasks; in 2.2 the two rows of one match
  never sit on opposite sides of a split (avoids leakage between paired rows).
* Hyper-parameters (alpha, l1_ratio, feature subsets) are tuned INSIDE each
  training fold (nested CV) so test folds stay untouched.
* A temporal hold-out (train on 72 group matches -> test on 32 knockouts)
  checks that the models generalise forward in time, not just across folds.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import (ElasticNetCV, HuberRegressor, LassoCV,
                                  LinearRegression, PoissonRegressor, RidgeCV)
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (FunctionTransformer, PolynomialFeatures, PowerTransformer,
                                   StandardScaler)
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 2026
FROM_CACHE = False
N_SPLITS, N_REPEATS = 5, 20

X21 = ["fifa_pts_diff", "form_gf_diff", "form_ga_diff", "form_win_diff",
       "host_diff", "wc_exp_diff", "rest_diff", "tourn_gd_diff"]
X22 = ["fifa_pts_diff", "wc_exp_diff", "rest_diff", "team_attack_form",
       "opp_defence_form", "is_host", "knockout", "team_tourn_gf_rate"]
assert len(set(X21) & set(X22)) <= 4, "brief: max 4 shared variables"


# ============================================================ feature engineering
def fifa_expectancy(d):
    """Expected result implied by a FIFA points gap, using FIFA's own ranking formula
    We = 1 / (10^(-dr/600) + 1), rescaled to [-1, 1]: a bounded, S-shaped transform."""
    return 2 / (1 + 10 ** (-d / 600)) - 1


def engineer_21(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    Z["fifa_pts_diff"] = fifa_expectancy(X["fifa_pts_diff"])           # non-linear strength (FIFA formula)
    Z["wc_exp_diff"] = np.sign(X.wc_exp_diff) * np.log1p(X.wc_exp_diff.abs())
    return Z


def engineer_22(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    Z["fifa_pts_diff"] = fifa_expectancy(X["fifa_pts_diff"])
    Z["wc_exp_diff"] = np.sign(X.wc_exp_diff) * np.log1p(X.wc_exp_diff.abs())
    # attack x defence matchup (scoring form of team times leakiness of opponent)
    Z["team_attack_form"] = np.log(X.team_attack_form * X.opp_defence_form + 0.1)
    Z["team_tourn_gf_rate"] = np.log1p(X.team_tourn_gf_rate)
    return Z


class LogTargetOLS(BaseEstimator, RegressorMixin):
    """OLS on log(1+y) with Duan smearing back-transform (count target)."""
    def fit(self, X, y):
        self.m_ = make_pipeline(StandardScaler(), LinearRegression()).fit(X, np.log1p(y))
        self.smear_ = np.mean(np.exp(np.log1p(y) - self.m_.predict(X)))
        return self

    def predict(self, X):
        return np.exp(self.m_.predict(X)) * self.smear_ - 1


class SymmetricAugment(BaseEstimator, RegressorMixin):
    """2.1 only: team1/team2 order is arbitrary at neutral venues. Train on both
    orientations (negate X and y) so the model cannot learn a spurious 'listed
    first' effect; host_diff keeps genuine home advantage."""
    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        Xa = pd.concat([X, -X]); ya = np.concatenate([y, -y])
        self.m_ = clone(self.base).fit(Xa, ya)
        return self

    def predict(self, X):
        return self.m_.predict(X)


def sc(model):
    return make_pipeline(StandardScaler(), model)


def inner_cv():
    return 5


def tuned(est, grid):
    """Hyper-parameters chosen by inner 5-fold CV inside each training fold."""
    return GridSearchCV(est, grid, cv=inner_cv(), scoring="neg_root_mean_squared_error")


def competitors(eng):
    """Non-linear and ensemble regressors from scikit-learn, compared against the
    linear regression family. Each gets the same engineered inputs and is tuned
    with nested CV so the comparison is fair."""
    return {
        # ---- non-linear regression
        "Polynomial (deg 2) + Ridge": make_pipeline(
            eng, StandardScaler(), PolynomialFeatures(2, include_bias=False),
            StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
        "SVR, RBF kernel (tuned)": tuned(
            make_pipeline(eng, StandardScaler(), SVR(kernel="rbf")),
            {"svr__C": [0.3, 1, 3], "svr__epsilon": [0.2, 0.5]}),
        "k-NN regression (tuned)": tuned(
            make_pipeline(eng, StandardScaler(), KNeighborsRegressor(weights="distance")),
            {"kneighborsregressor__n_neighbors": [5, 10, 20, 30]}),
        "Decision Tree (tuned)": tuned(
            DecisionTreeRegressor(random_state=SEED, min_samples_leaf=5),
            {"max_depth": [2, 3, 4]}),
        # ---- ensemble regression
        "Random Forest (tuned)": tuned(
            RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=2),
            {"min_samples_leaf": [5, 15]}),
        "Gradient Boosting (tuned)": tuned(
            GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, subsample=0.8,
                                      random_state=SEED),
            {"max_depth": [1, 2]}),
    }


FAMILY = {"Baseline (mean)": "Baseline", "Polynomial (deg 2) + Ridge": "Non-linear",
          "SVR, RBF kernel (tuned)": "Non-linear", "k-NN regression (tuned)": "Non-linear",
          "Decision Tree (tuned)": "Non-linear", "Random Forest (tuned)": "Ensemble",
          "Gradient Boosting (tuned)": "Ensemble"}


def family(name):
    return FAMILY.get(name, "Linear")


# continuous explanatory variables (binary/ordinal ones are not power-transformed)
CONT21 = ["fifa_pts_diff", "form_gf_diff", "form_ga_diff", "form_win_diff", "wc_exp_diff",
          "tourn_gd_diff"]
CONT22 = ["fifa_pts_diff", "wc_exp_diff", "team_attack_form", "opp_defence_form",
          "team_tourn_gf_rate"]


def week9_variants(eng, cont):
    """Two fixes taught in Week 9: Yeo-Johnson power transform (Gaussian-like inputs)
    and removing one variable of the most correlated pair (collinearity)."""
    return {
        "OLS + Yeo-Johnson": make_pipeline(eng, ColumnTransformer(
            [("yj", PowerTransformer(), cont)], remainder=StandardScaler()), LinearRegression()),
        "OLS + drop collinear (WC exp.)": make_pipeline(
            eng, FunctionTransformer(lambda Z: Z.drop(columns="wc_exp_diff")),
            StandardScaler(), LinearRegression()),
    }


def models_21():
    eng = FunctionTransformer(engineer_21)
    return {
        "Baseline (mean)": DummyRegressor(),
        "OLS (8 raw)": sc(LinearRegression()),
        "OLS + engineered": make_pipeline(eng, StandardScaler(), LinearRegression()),
        "OLS + eng + symmetric": SymmetricAugment(
            make_pipeline(eng, StandardScaler(), LinearRegression())),
        "Forward-selected OLS": make_pipeline(eng, StandardScaler(), SequentialFeatureSelector(
            LinearRegression(), n_features_to_select="auto", tol=1e-3,
            scoring="neg_root_mean_squared_error", cv=inner_cv()), LinearRegression()),
        "Ridge (nested CV)": make_pipeline(eng, StandardScaler(),
                                           RidgeCV(alphas=np.logspace(-2, 3, 40))),
        "Lasso (nested CV)": make_pipeline(eng, StandardScaler(),
                                           LassoCV(cv=inner_cv(), random_state=SEED)),
        "Elastic Net (nested CV)": make_pipeline(eng, StandardScaler(), ElasticNetCV(
            l1_ratio=[.1, .3, .5, .7, .9], cv=inner_cv(), random_state=SEED)),
        "Ridge + eng + symmetric": SymmetricAugment(make_pipeline(
            eng, StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40)))),
        "Huber (robust)": make_pipeline(eng, StandardScaler(), HuberRegressor(max_iter=1000)),
        **week9_variants(FunctionTransformer(engineer_21), CONT21),
        **competitors(FunctionTransformer(engineer_21)),
    }


def models_22():
    eng = FunctionTransformer(engineer_22)
    return {
        "Baseline (mean)": DummyRegressor(),
        "OLS (8 raw)": sc(LinearRegression()),
        "OLS + engineered": make_pipeline(eng, StandardScaler(), LinearRegression()),
        "OLS log(1+y) + eng": make_pipeline(eng, LogTargetOLS()),
        "Forward-selected OLS": make_pipeline(eng, StandardScaler(), SequentialFeatureSelector(
            LinearRegression(), n_features_to_select="auto", tol=1e-3,
            scoring="neg_root_mean_squared_error", cv=inner_cv()), LinearRegression()),
        "Ridge (nested CV)": make_pipeline(eng, StandardScaler(),
                                           RidgeCV(alphas=np.logspace(-2, 3, 40))),
        "Lasso (nested CV)": make_pipeline(eng, StandardScaler(),
                                           LassoCV(cv=inner_cv(), random_state=SEED)),
        "Elastic Net (nested CV)": make_pipeline(eng, StandardScaler(), ElasticNetCV(
            l1_ratio=[.1, .3, .5, .7, .9], cv=inner_cv(), random_state=SEED)),
        "Huber (robust)": make_pipeline(eng, StandardScaler(), HuberRegressor(max_iter=1000)),
        "Poisson GLM (log link)": make_pipeline(eng, StandardScaler(),
                                                PoissonRegressor(alpha=1e-3, max_iter=1000)),
        **week9_variants(FunctionTransformer(engineer_22), CONT22),
        **competitors(FunctionTransformer(engineer_22)),
    }


# ============================================================ resampling
def stage_of(d: pd.DataFrame) -> np.ndarray:
    """Matches are numbered by date: 1-72 = group stage, 73-104 = knockouts."""
    return np.where(d.match_id.values <= 72, "Group", "Knockout")


def match_level_splits(match_ids: np.ndarray):
    """Repeated K-fold over MATCHES; returns row-index splits."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(match_ids)
    splits = []
    for _ in range(N_REPEATS):
        perm = rng.permutation(uniq)
        for k in range(N_SPLITS):
            test_m = perm[k::N_SPLITS]
            te = np.where(np.isin(match_ids, test_m))[0]
            tr = np.where(~np.isin(match_ids, test_m))[0]
            splits.append((tr, te))
    return splits


def cross_validate(models, X, y, splits):
    fold_rows, oof = [], {}
    import time
    for i, (name, mdl) in enumerate(models.items(), 1):
        t0 = time.time()
        print(f"  [{i}/{len(models)}] {name} ...", end="", flush=True)
        preds = np.zeros((N_REPEATS, len(y)))
        for f, (tr, te) in enumerate(splits):
            m = clone(mdl).fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            preds[f // N_SPLITS, te] = p
            err = y[te] - p
            fold_rows.append({"model": name, "fold": f,
                              "rmse": np.sqrt(np.mean(err ** 2)),
                              "mae": np.mean(np.abs(err)),
                              "r2": 1 - np.sum(err ** 2) / np.sum((y[te] - y[tr].mean()) ** 2)})
        oof[name] = preds.mean(0)
        print(f" done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(fold_rows), oof


def corrected_paired_t(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for CV score differences."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), k - 1)
    return t, p


def summarise(folds, best_ref, n_train, n_test):
    g = folds.groupby("model")
    s = g.agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
              mae=("mae", "mean"), r2=("r2", "mean")).reset_index()
    k = folds.fold.nunique()
    s["rmse_ci95"] = 1.96 * s.rmse_sd / np.sqrt(k)
    base = folds[folds.model == "Baseline (mean)"].sort_values("fold").rmse.values
    s["p_vs_baseline"] = [corrected_paired_t(
        folds[folds.model == m].sort_values("fold").rmse.values, base, n_train, n_test)[1]
        for m in s.model]
    s["p_vs_OLSraw"] = [corrected_paired_t(
        folds[folds.model == m].sort_values("fold").rmse.values,
        folds[folds.model == "OLS (8 raw)"].sort_values("fold").rmse.values,
        n_train, n_test)[1] for m in s.model]
    s["family"] = s.model.map(family)
    return s.sort_values("rmse").reset_index(drop=True)


def recommend(folds, summ, n_train, n_test, alpha=0.05):
    """Recommendation rule (decided before looking at results): pick the best model
    of the linear regression family unless some competitor is SIGNIFICANTLY better
    (corrected resampled t-test, p < alpha). Linear models are interpretable and are
    the task's subject, so a non-significant RMSE edge does not justify losing that."""
    overall = summ.iloc[0].model
    lin = summ[summ.family == "Linear"].iloc[0].model
    if lin == overall:
        return lin, overall, np.nan
    _, p = corrected_paired_t(folds[folds.model == lin].sort_values("fold").rmse.values,
                              folds[folds.model == overall].sort_values("fold").rmse.values,
                              n_train, n_test)
    return (lin if p >= alpha else overall), overall, p


def temporal_holdout(models, X, y, stage):
    tr, te = np.where(stage == "Group")[0], np.where(stage == "Knockout")[0]
    rows = []
    for name, mdl in models.items():
        p = clone(mdl).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        rows.append({"model": name, "rmse_knockout": np.sqrt(np.mean((y[te] - p) ** 2)),
                     "mae_knockout": np.mean(np.abs(y[te] - p))})
    return pd.DataFrame(rows)


# ============================================================ inference
def ols_inference(Xs: pd.DataFrame, y, groups=None):
    """Standardised-coefficient OLS with robust SEs (HC3, or cluster by match)."""
    Z = (Xs - Xs.mean()) / Xs.std(ddof=0)
    Zc = sm.add_constant(Z)
    fit = sm.OLS(y, Zc).fit()
    rob = (fit.get_robustcov_results("cluster", groups=groups) if groups is not None
           else fit.get_robustcov_results("HC3"))
    ci = rob.conf_int()
    coef = pd.DataFrame({"coef": rob.params, "lo": ci[:, 0], "hi": ci[:, 1],
                         "p": rob.pvalues}, index=Zc.columns).drop("const")
    vif = pd.Series([variance_inflation_factor(Zc.values, i + 1) for i in range(Z.shape[1])],
                    index=Z.columns)
    coef["vif"] = vif
    bp = het_breuschpagan(fit.resid, Zc)[1]
    sw = stats.shapiro(fit.resid).pvalue
    diag = {"r2": fit.rsquared, "adj_r2": fit.rsquared_adj, "F_p": fit.f_pvalue,
            "aic": fit.aic, "breusch_pagan_p": bp, "shapiro_resid_p": sw,
            "n": int(fit.nobs)}
    return coef, diag, fit


def mean_ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), stats.sem(x)
    h = se * stats.t.ppf(.975, len(x) - 1)
    return m, m - h, m + h


def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


# ============================================================ main
def run():
    d21 = pd.read_csv(DATA / "lr21_matches.csv")
    d22 = pd.read_csv(DATA / "lr22_team_matches.csv")
    res = {}

    # ------------------------------------------------ descriptive + inferential EDA
    eda = {}
    m, lo, hi = mean_ci(d21.goal_diff)
    eda["gd_mean_ci"] = [m, lo, hi]
    eda["gd_skew"] = float(stats.skew(d21.goal_diff))
    m, lo, hi = mean_ci(d22.goals)
    eda["goals_mean_ci"] = [m, lo, hi]
    eda["goals_var"] = float(d22.goals.var())
    eda["goals_dispersion"] = float(d22.goals.var() / d22.goals.mean())
    # chi-square goodness of fit of goals to Poisson
    lam = d22.goals.mean()
    obs = np.bincount(np.minimum(d22.goals, 4), minlength=5)
    exp = np.array([stats.poisson.pmf(k, lam) for k in range(4)] +
                   [stats.poisson.sf(3, lam)]) * len(d22)
    eda["poisson_gof_p"] = float(stats.chisquare(obs, exp, ddof=1).pvalue)
    # Two-sample (Welch) t-tests
    g = d22[stage_of(d22) == "Group"].goals; k = d22[stage_of(d22) == "Knockout"].goals
    eda["ttest_goals_group_vs_ko"] = [g.mean(), k.mean(),
                                      float(stats.ttest_ind(g, k, equal_var=False).pvalue)]
    fav = d21.fifa_pts_diff > 0
    absgd_fav = np.where(fav, d21.goal_diff, -d21.goal_diff)
    eda["fav_margin_ci"] = list(mean_ci(absgd_fav))
    eda["ttest_fav_margin_gt0_p"] = float(stats.ttest_1samp(absgd_fav, 0,
                                                            alternative="greater").pvalue)
    h = d22[d22.is_host == 1].goals; nh = d22[d22.is_host == 0].goals
    eda["ttest_host"] = [h.mean(), nh.mean(),
                         float(stats.ttest_ind(h, nh, equal_var=False).pvalue), len(h)]
    # correlations with target (+95% CI)
    cor = []
    for task, d, X, yv in [("2.1", d21, X21, "goal_diff"), ("2.2", d22, X22, "goals")]:
        for c in X:
            r, p = stats.pearsonr(d[c], d[yv])
            lo_, hi_ = fisher_ci(r, len(d))
            cor.append({"task": task, "feature": c, "r": r, "lo": lo_, "hi": hi_, "p": p})
    cor = pd.DataFrame(cor)
    cor.to_csv(OUT / "eda_correlations.csv", index=False)
    d21[X21 + ["goal_diff"]].describe().T.to_csv(OUT / "desc_21.csv")
    d22[X22 + ["goals"]].describe().T.to_csv(OUT / "desc_22.csv")
    res["eda"] = eda

    # ------------------------------------------------ modelling 2.1
    for task, d, X, yv, mods in [("2.1", d21, X21, "goal_diff", models_21()),
                                 ("2.2", d22, X22, "goals", models_22())]:
        Xd, y = d[X], d[yv].values.astype(float)
        print(f"\nLR {task}: fitting {len(mods)} models x 100 cross-validation splits", flush=True)
        splits = match_level_splits(d.match_id.values)
        if FROM_CACHE and (OUT / f"cv_folds_{task}.csv").exists():
            folds = pd.read_csv(OUT / f"cv_folds_{task}.csv")
            o = pd.read_csv(OUT / f"oof_{task}.csv")
            oof = {m: o[m].values for m in mods}
        else:
            folds, oof = cross_validate(mods, Xd, y, splits)
        n_te = np.mean([len(te) for _, te in splits]); n_tr = len(y) - n_te
        summ = summarise(folds, None, n_tr, n_te)
        # metrics taught in Weeks 8-9: MSE and RMSE normalised by the range of y
        summ["mse"] = summ.model.map(folds.assign(mse=folds.rmse ** 2).groupby("model").mse.mean())
        summ["nrmse"] = summ.rmse / (y.max() - y.min())
        best, overall_best, p_lin_vs_overall = recommend(folds, summ, n_tr, n_te)
        hold = temporal_holdout(mods, Xd, y, stage_of(d))
        summ = summ.merge(hold, on="model")
        ref = folds[folds.model == best].sort_values("fold").rmse.values
        summ["p_vs_best"] = [corrected_paired_t(
            folds[folds.model == m].sort_values("fold").rmse.values, ref, n_tr, n_te)[1]
            if m != best else np.nan for m in summ.model]
        wr = folds.pivot(index="fold", columns="model", values="rmse")
        summ["best_wins_pct"] = [np.mean(wr[best] < wr[m]) * 100 if m != best else np.nan
                                 for m in summ.model]
        folds.to_csv(OUT / f"cv_folds_{task}.csv", index=False)
        summ.to_csv(OUT / f"cv_summary_{task}.csv", index=False)
        pd.DataFrame(oof).assign(match_id=d.match_id, y=y).to_csv(
            OUT / f"oof_{task}.csv", index=False)
        print(f"\n===== LR {task} =====\n", summ.round(3).to_string())

        eng = engineer_21 if task == "2.1" else engineer_22
        coef_raw, diag_raw, _ = ols_inference(Xd, y, d.match_id.values if task == "2.2" else None)
        coef_eng, diag_eng, fit = ols_inference(eng(Xd), y,
                                                d.match_id.values if task == "2.2" else None)
        coef_raw.to_csv(OUT / f"ols_coef_raw_{task}.csv")
        coef_eng.to_csv(OUT / f"ols_coef_eng_{task}.csv")
        print(coef_eng.round(3).to_string()); print(diag_raw, "\n", diag_eng)

        # lasso path selection frequency across folds (feature stability)
        sel = np.zeros(len(X))
        for tr, _ in splits[:N_SPLITS * 5]:
            l = make_pipeline(FunctionTransformer(eng), StandardScaler(),
                              LassoCV(cv=5, random_state=SEED)).fit(Xd.iloc[tr], y[tr])
            sel += np.abs(l[-1].coef_) > 1e-6
        stab = pd.Series(sel / (N_SPLITS * 5), index=X)
        stab.to_csv(OUT / f"lasso_stability_{task}.csv")

        res[task] = {"best": best, "overall_lowest_cv_rmse": overall_best,
                     "p_recommended_vs_lowest": p_lin_vs_overall, "summary": summ.to_dict("records"),
                     "diag_raw": diag_raw, "diag_eng": diag_eng,
                     "coef_eng": coef_eng.reset_index().to_dict("records"),
                     "lasso_stability": stab.to_dict()}

    # ------------------------------------------------ cross-model consistency
    # Does modelling goals per team (2.2) then differencing beat modelling GD directly?
    o22 = pd.read_csv(OUT / "oof_2.2.csv"); o21 = pd.read_csv(OUT / "oof_2.1.csv")
    best22 = res["2.2"]["best"]; best21 = res["2.1"]["best"]
    t1 = o22.iloc[0::2].reset_index(drop=True); t2 = o22.iloc[1::2].reset_index(drop=True)
    gd_from_22 = t1[best22] - t2[best22]
    gd_true = o21.y.values
    res["cross"] = {
        "rmse_gd_direct": float(np.sqrt(np.mean((gd_true - o21[best21]) ** 2))),
        "rmse_gd_from_22": float(np.sqrt(np.mean((gd_true - gd_from_22) ** 2))),
        "outcome_acc_direct": float(np.mean(
            np.sign(np.where(np.abs(o21[best21]) < .5, 0, o21[best21])) == np.sign(gd_true))),
        "outcome_acc_from_22": float(np.mean(
            np.sign(np.where(np.abs(gd_from_22) < .5, 0, gd_from_22)) == np.sign(gd_true))),
        "outcome_acc_fifa_only": float(np.mean(np.sign(d21.fifa_pts_diff) == np.sign(gd_true))),
        "base_rate_draw": float(np.mean(gd_true == 0)),
    }
    print(res["cross"])
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    import sys
    FROM_CACHE = "--from-cache" in sys.argv
    run()
