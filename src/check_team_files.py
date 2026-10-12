"""
Quality check for the manually exported FBref national-team files
(team_<Team>_<season>.csv, seasons 2025 and 2026).

For every team it reports:
  * rows per file and date range (catches empty / wrong-season exports)
  * whether the 2025 and 2026 files are identical (same page exported twice)
  * IDENTITY: the World Cup 2026 rows in the file must match that team's
    fixtures in fbref_wc2026.csv (catches a file saved under the wrong name)
  * matches available before 11 June 2026, and the date span of the last 10
    (catches missing seasons, i.e. gaps in recent form)

Usage: python check_team_files.py <folder with team files>
"""
import csv
import io
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WC_START = pd.Timestamp("2026-06-11")
FBREF_WC = ROOT / "data" / "processed" / "fbref_wc2026_clean.csv"

# file-name team -> name used in fbref_wc2026_clean.csv
ALIASES = {"Cote dIvoire": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast",
           "Côte d’Ivoire": "Ivory Coast", "Curacao": "Curaçao", "Curaçao": "Curaçao",
           "Turkiye": "Turkey", "Türkiye": "Turkey", "Korea Republic": "South Korea",
           "Korea_Republic": "South Korea", "Czechia": "Czech Republic", "USA": "United States",
           "Iran": "Iran", "IR Iran": "Iran", "Congo DR": "DR Congo", "Cabo Verde": "Cape Verde",
           "Bosnia and Herzegovina": "Bosnia and Herzegovina"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def read_table(path: Path) -> pd.DataFrame:
    """Read an FBref export saved as CSV (possibly line-quoted by Excel) or XLSX."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, dtype=str).fillna("")
        if raw.shape[1] == 1 or (raw.iloc[:, 1:] == "").all().all():
            text = "\n".join(raw.iloc[:, 0])          # pasted CSV text in column A
        else:
            return pd.read_excel(path, dtype=str)
    else:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
            line = line[1:-1].replace('""', '"')
        lines.append(line)
    rows = [r for r in csv.reader(io.StringIO("\n".join(lines))) if any(c.strip() for c in r)]
    header, body = rows[0], rows[1:]
    body = [r + [""] * (len(header) - len(r)) for r in body]
    df = pd.DataFrame([r[:len(header)] for r in body], columns=header)
    return df[df.get("Date", pd.Series(dtype=str)).str.match(r"\d{4}-\d{2}-\d{2}", na=False)]


def team_of(path: Path) -> tuple[str, str]:
    m = re.match(r"team_(.+)_(\d{4})$", path.stem)
    return (m.group(1).replace("_", " ").strip(), m.group(2)) if m else (None, None)


def main(folder: Path) -> pd.DataFrame:
    wc = pd.read_csv(FBREF_WC, parse_dates=["date"])
    wc_names = set(wc.home) | set(wc.away)
    files = sorted(p for p in folder.rglob("team_*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    by_team: dict[str, dict[str, Path]] = {}
    for p in files:
        t, y = team_of(p)
        if t:
            by_team.setdefault(t, {})[y] = p

    report = []
    for team, seasons in sorted(by_team.items()):
        team_nfc = unicodedata.normalize("NFC", team)
        alias_by_norm = {norm(k): v for k, v in ALIASES.items()}
        canon = ALIASES.get(team_nfc, alias_by_norm.get(norm(team_nfc), team_nfc))
        if canon not in wc_names:  # fuzzy match on normalised name
            cand = [n for n in wc_names if norm(n) == norm(canon)]
            canon = cand[0] if cand else canon
        issues, info = [], {}
        frames = []
        for y in ("2025", "2026"):
            if y not in seasons:
                issues.append(f"missing {y} file")
                continue
            try:
                df = read_table(seasons[y])
            except Exception as e:  # noqa: BLE001
                issues.append(f"{y} unreadable ({e})")
                continue
            df = df.assign(Date=pd.to_datetime(df.Date), season=y)
            frames.append(df)
            info[f"rows_{y}"] = len(df)
            info[f"range_{y}"] = f"{df.Date.min().date()}..{df.Date.max().date()}" if len(df) else "-"
            if seasons[y].suffix.lower() != ".csv":
                info[f"note_{y}"] = "xlsx"
        if len(frames) == 2 and frames[0].drop(columns="season").reset_index(drop=True).equals(
                frames[1].drop(columns="season").reset_index(drop=True)):
            issues.append("2025 and 2026 files are identical (same season exported twice?)")
        if not frames:
            report.append({"team": team, "status": "FAIL", "issues": "; ".join(issues)})
            continue
        allm = (pd.concat(frames).drop_duplicates(["Date", "Opponent"]).sort_values("Date"))

        # identity check against the World Cup 2026 fixtures
        exp = wc[(wc.home == canon) | (wc.away == canon)]
        got = allm[(allm.Comp.str.contains("World Cup", na=False)) &
                   (~allm.Comp.str.contains("qual", case=False, na=False)) &
                   (allm.Date >= WC_START)]
        info["wc_rows"] = f"{len(got)}/{len(exp)}"
        if len(exp) == 0:
            issues.append(f"team name '{team}' not found among World Cup teams")
        elif len(got) != len(exp) or set(got.Date.dt.date) != set(exp.date.dt.date):
            issues.append("World Cup matches don't match this team (wrong team or season?)")

        pre = allm[allm.Date < WC_START]
        last10 = pre.tail(10)
        info["pre_wc_matches"] = len(pre)
        if len(last10):
            info["last10_span"] = f"{last10.Date.min().date()}..{last10.Date.max().date()}"
            gaps = last10.Date.diff().dt.days.dropna()
            if len(last10) < 10:
                issues.append(f"only {len(last10)} matches before the World Cup")
            if (gaps > 200).any():
                issues.append("gap >200 days inside the last 10 (a season may be missing)")
            if last10.Date.max() < pd.Timestamp("2026-03-01"):
                issues.append("no 2026 warm-up matches found")
        report.append({"team": team, **info,
                       "status": "OK" if not issues else "CHECK", "issues": "; ".join(issues)})
    rep = pd.DataFrame(report)
    print(rep.to_string(index=False))
    return rep


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "fbref_teams")
