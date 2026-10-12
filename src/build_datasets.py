"""
Step 1 - Build the two modelling datasets for Objective 2 from the manually
collected FBref and FIFA files (parsed by parse_fbref.py).

Every explanatory variable uses ONLY information available before kick-off:
  * FIFA ranking points  - the 11 June 2026 release (published before the opening
                           match; no ranking update was issued during the tournament)
  * form, rest days      - the team's internationals dated strictly before the match
                           (FBref national-team Scores & Fixtures, seasons 2025 + 2026)
  * World Cup experience - World Cup matches played 1998-2022 (FBref)
  * tournament-so-far    - earlier World Cup 2026 matches only
  * host, stage          - known from the fixture list in advance

Outputs
  data/processed/lr21_matches.csv       104 rows x (ids + 8 X + target)
  data/processed/lr22_team_matches.csv  208 rows x (ids + 8 X + target)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
HOSTS = {"United States", "Mexico", "Canada"}
FORM_N = 10      # matches used for rolling form
# Rest-day outliers are capped with the IQR method (Week 6): values outside
# [Q1 - 1.5*IQR, Q3 + 1.5*IQR] of all team rest days are set to the fence.


def main() -> None:
    wc = pd.read_csv(P / "fbref_wc2026_clean.csv", parse_dates=["date"])
    tm = pd.read_csv(P / "team_matches.csv", parse_dates=["date"])
    past = pd.read_csv(P / "worldcups_1998_2022.csv")
    fifa = pd.read_csv(P / "fifa_ranking.csv").set_index("team").points

    wc = wc.sort_values("date", kind="stable").reset_index(drop=True)
    wc["match_id"] = np.arange(1, 105)
    wc["stage"] = np.where(wc["round"] == "Group stage", "Group", "Knockout")
    assert (wc.stage == "Group").sum() == 72

    wc_exp = pd.concat([past.home, past.away]).value_counts()     # matches 1998-2022

    def team_state(team: str, date: pd.Timestamp) -> dict:
        hist = tm[(tm.team == team) & (tm.date < date)].sort_values("date")
        last = hist.tail(FORM_N)
        tourn = wc[(wc.date < date) & ((wc.home == team) | (wc.away == team))]
        gf = np.where(tourn.home == team, tourn.home_goals, tourn.away_goals).sum()
        ga = np.where(tourn.home == team, tourn.away_goals, tourn.home_goals).sum()
        rest = (date - hist.date.iloc[-1]).days
        return {"form_gf": last.gf.mean(), "form_ga": last.ga.mean(),
                "form_win": (last.result == "W").mean(), "n_form": len(last),
                "rest": rest, "wc_exp": int(wc_exp.get(team, 0)),
                "fifa": float(fifa[team]), "t_played": len(tourn), "t_gf": gf, "t_ga": ga}

    def host(team, row):
        return int(team in HOSTS and team == row.venue_country)

    states = {(r.match_id, side): team_state(team, r.date)
              for r in wc.itertuples() for side, team in (("h", r.home), ("a", r.away))}
    raw_rest = pd.Series([s["rest"] for s in states.values()])
    q1, q3 = raw_rest.quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n_out = int(((raw_rest < lo) | (raw_rest > hi)).sum())
    print(f"Rest days: Q1={q1:.0f}, Q3={q3:.0f}, IQR fences [{lo:.0f}, {hi:.0f}] -> {n_out} outliers capped")
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.Series({"q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi, "n_outliers": n_out,
               "max_raw": raw_rest.max()}).to_json(ROOT / "outputs" / "rest_iqr.json")
    uncapped = []
    for s_ in states.values():
        s_["rest_raw"] = s_["rest"]
        s_["rest"] = float(np.clip(s_["rest"], lo, hi))

    m, t = [], []
    for r in wc.itertuples():
        a, b = states[(r.match_id, "h")], states[(r.match_id, "a")]
        uncapped.append({"match_id": r.match_id, "team": r.home, "rest_diff_uncapped": a["rest_raw"] - b["rest_raw"]})
        uncapped.append({"match_id": r.match_id, "team": r.away, "rest_diff_uncapped": b["rest_raw"] - a["rest_raw"]})
        assert a["n_form"] == FORM_N and b["n_form"] == FORM_N, (r.home, r.away)
        m.append({
            "match_id": r.match_id, "date": r.date.date(),
            "team1": r.home, "team2": r.away,
            # ---- 8 explanatory variables (team1 minus team2, all pre-match)
            "fifa_pts_diff": a["fifa"] - b["fifa"],
            "form_gf_diff": a["form_gf"] - b["form_gf"],
            "form_ga_diff": a["form_ga"] - b["form_ga"],
            "form_win_diff": a["form_win"] - b["form_win"],
            "host_diff": host(r.home, r) - host(r.away, r),
            "wc_exp_diff": a["wc_exp"] - b["wc_exp"],
            "rest_diff": a["rest"] - b["rest"],
            "tourn_gd_diff": (a["t_gf"] - a["t_ga"]) - (b["t_gf"] - b["t_ga"]),
            # ---- target
            "goal_diff": r.home_goals - r.away_goals,
        })
        for team, opp, s, o, goals in [(r.home, r.away, a, b, r.home_goals),
                                       (r.away, r.home, b, a, r.away_goals)]:
            t.append({
                "match_id": r.match_id, "date": r.date.date(),
                "team": team, "opponent": opp,
                # ---- 8 explanatory variables (pre-match)
                "fifa_pts_diff": s["fifa"] - o["fifa"],          # shared with 2.1
                "wc_exp_diff": s["wc_exp"] - o["wc_exp"],        # shared with 2.1
                "rest_diff": s["rest"] - o["rest"],              # shared with 2.1
                "team_attack_form": s["form_gf"],
                "opp_defence_form": o["form_ga"],
                "is_host": host(team, r),
                "knockout": int(r.stage == "Knockout"),
                # scoring rate so far; before a team's first match, pre-tournament form
                "team_tourn_gf_rate": s["t_gf"] / s["t_played"] if s["t_played"] else s["form_gf"],
                # ---- target
                "goals": goals,
            })
    d21, d22 = pd.DataFrame(m), pd.DataFrame(t)
    # ID fields + exactly 8 explanatory variables + 1 response variable
    assert d21.shape == (104, 13) and d22.shape == (208, 13)
    assert not d21.isna().any().any() and not d22.isna().any().any()
    d21.to_csv(P / "lr21_matches.csv", index=False)
    # uncapped rest differences, kept only to show the effect of outlier capping (assumptions.py)
    pd.DataFrame(uncapped).to_csv(P / "rest_diff_uncapped.csv", index=False)
    d22.to_csv(P / "lr22_team_matches.csv", index=False)
    print(d21.describe().T.round(2).to_string())
    print(d22.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
