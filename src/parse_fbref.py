"""
Step 0 - Read every manually collected file into tidy tables.

Manual collection (see README "Data collection"):
  data/raw/fbref_wc2026.csv            FBref World Cup 2026 Scores & Fixtures (104 matches)
  data/raw/fbref_worldcups/wc_YYYY.csv  FBref World Cup Scores & Fixtures, 1998-2022
  data/raw/fbref_teams/team_<T>_<S>.csv FBref national-team Scores & Fixtures, seasons 2025 & 2026
  data/raw/fifa/fifa_ranking_2026-06-11.csv  FIFA/Coca-Cola Men's World Ranking, 11 June 2026

Each FBref table was exported with Share & Export -> Get table as CSV, pasted
into Excel and saved as CSV UTF-8 (Excel wraps each line in quotes; handled here).

Outputs (data/processed/):
  fbref_wc2026_clean.csv   104 tidy matches (targets, stage, venue country)
  worldcups_1998_2022.csv  448 tidy matches
  team_matches.csv         every international in the team files, one row per team-match
  fifa_ranking.csv         canonical team names + points
"""
import csv
import io
import re
from pathlib import Path

import pandas as pd

from names import canonical

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SCORE_RE = re.compile(r"^(?:\((\d+)\)\s*)?(\d+)\s*[–-]\s*(\d+)(?:\s*\((\d+)\))?$")

# The 16 World Cup 2026 stadiums by host country (used for the host-nation variable)
MEXICO = {"Estadio Banorte", "Estadio Akron", "Estadio BBVA"}
CANADA = {"BMO Field", "BC Place Stadium"}


def read_export(path: Path) -> pd.DataFrame:
    """Read an FBref export (CSV, possibly line-quoted by Excel, or XLSX)."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, dtype=str).fillna("")
        text = "\n".join(raw.iloc[:, 0]) if (raw.iloc[:, 1:] == "").all().all() else None
        if text is None:
            return pd.read_excel(path, dtype=str).fillna("")
    else:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
            line = line[1:-1].replace('""', '"')
        lines.append(line)
    rows = [r for r in csv.reader(io.StringIO("\n".join(lines))) if any(c.strip() for c in r)]
    header = rows[0]
    body = [(r + [""] * len(header))[:len(header)] for r in rows[1:]]
    return pd.DataFrame(body, columns=header)


def read_fixtures(path: Path) -> pd.DataFrame:
    """FBref competition 'Scores & Fixtures' table -> one tidy row per match."""
    df = read_export(path)
    df = df[df.Score.str.strip() != ""]
    parsed = df.Score.str.strip().str.extract(SCORE_RE)
    venue = df.Venue.str.replace(r"\s*\(Neutral Site\)", "", regex=True).str.strip()
    return pd.DataFrame({
        "date": pd.to_datetime(df.Date),
        "round": df.Round.str.strip(),
        "home": df.Home.str.rsplit(" ", n=1).str[0].map(canonical),   # "Mexico mx"
        "away": df.Away.str.split(" ", n=1).str[1].map(canonical),    # "za South Africa"
        "home_goals": parsed[1].astype(int), "away_goals": parsed[2].astype(int),
        "home_pens": pd.to_numeric(parsed[0]), "away_pens": pd.to_numeric(parsed[3]),
        "venue": venue,
        "venue_country": venue.map(lambda v: "Mexico" if v in MEXICO else
                                   "Canada" if v in CANADA else "United States"),
    }).reset_index(drop=True)


def read_team_files(folder: Path) -> pd.DataFrame:
    """All national-team match logs -> one row per (team, match), de-duplicated."""
    frames = []
    for p in sorted(folder.glob("team_*")):
        m = re.match(r"team_(.+)_(\d{4})$", p.stem)
        df = read_export(p)
        df = df[df.Date.str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
        gf = df.GF.str.extract(r"^(\d+)")[0]
        ga = df.GA.str.extract(r"^(\d+)")[0]
        ok = gf.notna() & ga.notna()                        # drop unplayed fixtures
        frames.append(pd.DataFrame({
            "team": canonical(m.group(1).replace("_", " ")), "season_file": m.group(2),
            "date": pd.to_datetime(df.Date[ok]), "comp": df.Comp[ok],
            "opponent": df.Opponent[ok].str.split(" ", n=1).str[1].map(canonical),
            "gf": gf[ok].astype(int), "ga": ga[ok].astype(int),
            "result": df.Result[ok].str.strip(),
        }))
    tm = pd.concat(frames, ignore_index=True)
    tm = tm.sort_values(["team", "date", "season_file"]).drop_duplicates(
        ["team", "date", "opponent"], keep="last")
    return tm.reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wc26 = read_fixtures(RAW / "fbref_wc2026.csv")
    assert len(wc26) == 104, len(wc26)
    wc26.to_csv(OUT / "fbref_wc2026_clean.csv", index=False)

    past = pd.concat([read_fixtures(p).assign(edition=int(p.stem[-4:]))
                      for p in sorted((RAW / "fbref_worldcups").glob("wc_*.csv"))])
    assert len(past) == 7 * 64, len(past)
    past.to_csv(OUT / "worldcups_1998_2022.csv", index=False)

    tm = read_team_files(RAW / "fbref_teams")
    teams26 = set(wc26.home) | set(wc26.away)
    assert set(tm.team) == teams26, set(tm.team) ^ teams26
    tm.to_csv(OUT / "team_matches.csv", index=False)

    fifa = pd.read_csv(RAW / "fifa" / "fifa_ranking_2026-06-11.csv")
    fifa["team"] = fifa.team.map(canonical)
    missing = teams26 - set(fifa.team)
    assert not missing, missing
    fifa.to_csv(OUT / "fifa_ranking.csv", index=False)

    # consistency check: World Cup 2026 results in the team files vs the fixtures table
    rows = tm[(tm.comp == "World Cup") & (tm.date >= "2026-06-11")]
    merged = rows.merge(pd.concat([
        wc26.rename(columns={"home": "team", "away": "opponent", "home_goals": "f_gf",
                             "away_goals": "f_ga"}),
        wc26.rename(columns={"away": "team", "home": "opponent", "away_goals": "f_gf",
                             "home_goals": "f_ga"})])[["date", "team", "opponent", "f_gf", "f_ga"]],
        on=["date", "team", "opponent"], how="left")
    agree = ((merged.gf == merged.f_gf) & (merged.ga == merged.f_ga)).sum()
    print(f"World Cup 2026: {len(wc26)} matches | past World Cups: {len(past)} matches | "
          f"team-file rows: {len(tm)} for {tm.team.nunique()} teams | FIFA teams: {len(fifa)}")
    print(f"Cross-check: {agree}/{len(merged)} World Cup 2026 team-match results in the team "
          f"files agree with the fixtures table (expected 208/208)")


if __name__ == "__main__":
    main()
