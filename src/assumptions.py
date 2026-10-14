"""
Step 3 - Model optimisation guided by the Week 9 assumptions of linear regression.

Starting from plain OLS on the 8 explanatory variables (rest days NOT outlier-capped),
each taught fix is added ON ITS OWN, then the helpful ones are combined. All runs
use the same 100 cross-validation splits as modelling.py, so RMSEs are comparable.

  1. Remove noise          - cap rest-day outliers with the IQR method (Week 6)
  2. Linearity             - non-linear transforms: FIFA points -> expected result,
                             signed log of WC experience (LR 2.2 also log terms)
  3. Remove collinearity   - drop WC experience (r = 0.82 with FIFA points)
  4. Gaussian distribution - Yeo-Johnson power transform of continuous variables
  5. Rescale inputs        - z-score standardisation (changes interpretability, not RMSE)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from modelling import (CONT21, CONT22, DATA, OUT, X21, X22, cross_validate, engineer_21,
                       engineer_22, match_level_splits)

unc = pd.read_csv(DATA / "rest_diff_uncapped.csv")


def ols(*steps, scale=True):
    return make_pipeline(*[FunctionTransformer(f) for f in steps],
                         *([StandardScaler()] if scale else []), LinearRegression())


def yj(cont, *steps):
    return make_pipeline(*[FunctionTransformer(f) for f in steps], ColumnTransformer(
        [("yj", PowerTransformer(), cont)], remainder=StandardScaler()), LinearRegression())


rows = []
for task, file, X, yv, eng, cont in [("2.1", "lr21_matches.csv", X21, "goal_diff", engineer_21, CONT21),
                                     ("2.2", "lr22_team_matches.csv", X22, "goals", engineer_22, CONT22)]:
    d = pd.read_csv(DATA / file)
    key = ["match_id"] if task == "2.1" else ["match_id", "team"]
    u = (unc.drop_duplicates("match_id") if task == "2.1" else unc)
    d = d.merge(u[key + ["rest_diff_uncapped"]], on=key, how="left")
    capped = d[X].copy()
    uncapped = capped.assign(rest_diff=d.rest_diff_uncapped.values)
    y = d[yv].values.astype(float)
    splits = match_level_splits(d.match_id.values)
    drop = lambda Z: Z.drop(columns="wc_exp_diff")
    exps = {  # name -> (model, uses capped rest days?)
        "0. Plain OLS (raw variables)": (ols(), False),
        "1. Remove noise: IQR outlier cap": (ols(), True),
        "2. Linearity: non-linear transforms": (ols(eng), False),
        "3. Collinearity: drop WC experience": (ols(drop), False),
        "4. Gaussian: Yeo-Johnson": (yj(cont), False),
        "5. Rescale: z-scores (same fit)": (ols(scale=True), False),
        "1+2 combined (used in the final models)": (ols(eng), True),
        "1+2+3 combined": (ols(eng, drop), True),
        "1+2+4 combined": (yj(cont, eng), True),
    }
    for name, (model, use_capped) in exps.items():
        folds, _ = cross_validate({name: model}, capped if use_capped else uncapped, y, splits)
        rows.append({"task": task, "experiment": name, "cv_rmse": folds.rmse.mean(),
                     "cv_r2": folds.r2.mean()})
res = pd.DataFrame(rows)
res["change_vs_plain"] = res.cv_rmse - res.groupby("task").cv_rmse.transform("first")
res.to_csv(OUT / "assumptions.csv", index=False)
print(res.round(3).to_string(index=False))
