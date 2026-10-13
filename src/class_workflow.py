"""
The core workflow exactly as taught in the Week 8-9 practicals
(simple_linreg.py / multiple_linreg.py / multiple_linreg_collinearity.py),
applied to both World Cup datasets:

  1. separate explanatory variables (x) from the response variable (y)
  2. split 60% training / 40% test (train_test_split, random_state=0)
  3. fit LinearRegression, predict the test set
  4. MAE, MSE, RMSE, normalised RMSE and R^2, compared with a mean-only baseline
  5. statsmodels OLS summary (coefficients, p-values, R^2, adjusted R^2)

Step 6 repeats the 60/40 split with 100 different random seeds to show how much a
single split's result depends on luck - the reason the main analysis (modelling.py)
averages over 100 cross-validation splits instead.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "processed", ROOT / "outputs"

X21 = ["fifa_pts_diff", "form_gf_diff", "form_ga_diff", "form_win_diff",
       "host_diff", "wc_exp_diff", "rest_diff", "tourn_gd_diff"]
X22 = ["fifa_pts_diff", "wc_exp_diff", "rest_diff", "team_attack_form",
       "opp_defence_form", "is_host", "knockout", "team_tourn_gf_rate"]


def evaluate(y_test, y_pred, y_all):
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return {"MAE": metrics.mean_absolute_error(y_test, y_pred),
            "MSE": metrics.mean_squared_error(y_test, y_pred),
            "RMSE": rmse,
            "NRMSE": rmse / (y_all.max() - y_all.min()),
            "R2": metrics.r2_score(y_test, y_pred)}


results, report = {}, []
for task, file, cols, target in [("LR 2.1", "lr21_matches.csv", X21, "goal_diff"),
                                 ("LR 2.2", "lr22_team_matches.csv", X22, "goals")]:
    df = pd.read_csv(DATA / file)

    # 1. separate explanatory variables (x) from the response variable (y)
    x = df[cols].values
    y = df[target].values

    # 2. split dataset into 60% training and 40% test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    # 3. build, train and use the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. performance vs a baseline that always predicts the training mean
    y_pred_base = [np.mean(y_train)] * len(y_test)
    lr_metrics = evaluate(y_test, y_pred, y)
    base_metrics = evaluate(y_test, y_pred_base, y)

    # 5. statsmodels OLS on all rows: coefficients, p-values, R^2 and adjusted R^2
    ols = sm.OLS(df[target], sm.add_constant(df[cols])).fit()
    report.append(f"\n{'=' * 30} {task} {'=' * 30}\n{ols.summary()}")

    # 6. how much does a single 60/40 split depend on the random seed?
    rmses = []
    for seed in range(100):
        a, b, c, d = train_test_split(x, y, test_size=0.4, random_state=seed)
        rmses.append(math.sqrt(metrics.mean_squared_error(d, LinearRegression().fit(a, c).predict(b))))

    results[task] = {"linear_regression": lr_metrics, "baseline": base_metrics,
                     "ols_r2": ols.rsquared, "ols_adj_r2": ols.rsquared_adj,
                     "single_split_rmse_min": min(rmses), "single_split_rmse_max": max(rmses),
                     "single_split_rmse_mean": float(np.mean(rmses))}
    print(f"\n##### {task}: 60/40 split (random_state=0) #####")
    print(pd.DataFrame({"Linear regression": lr_metrics, "Baseline (mean)": base_metrics}).round(3))
    print(f"OLS R^2 = {ols.rsquared:.3f}, adjusted R^2 = {ols.rsquared_adj:.3f}")
    print(f"RMSE over 100 different 60/40 splits: {min(rmses):.2f} to {max(rmses):.2f}")

(OUT / "class_workflow.json").write_text(json.dumps(results, indent=2))
(OUT / "ols_summaries.txt").write_text("\n".join(report))
