#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end builder for a Premier League 1X2 dataset (paper-style).

Steps:
  1) Download dataset with kagglehub (Chris Lekas – Top-5 leagues)
  2) Locate main CSV
  3) Filter Premier League rows
  4) Keep core 1X2 columns (date, teams, goals, market odds)
  5) Build pre-match features (market implied/fair probs, Elo, form if present)
  6) Add temporal context (rest days, congestion, season halves, round)
  7) Save final CSV ready for modeling

Output:
  premier_dataset_final.csv  (in current working directory)

Usage:
  python make_premier_dataset.py --prefer_odds Avg --out premier_dataset_final.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------- Odds sets -------------------------------------
ODDS_SETS = {
    "Avg":  ("AvgH","AvgD","AvgA"),
    "PS":   ("PSH","PSD","PSA"),
    "B365": ("B365H","B365D","B365A"),
    "WH":   ("WHH","WHD","WHA"),
    "Max":  ("MaxH","MaxD","MaxA"),
}

def pick_csv(download_dir: Path) -> Path:
    csvs = list(download_dir.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"[ERROR] No CSV files under {download_dir}")
    # Prefer the largest file (usually the merged matches file)
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0]

def pick_odds_columns(df: pd.DataFrame, prefer: str = "Avg"):
    # Try the preferred set first, then fall back through others
    order = [prefer] + [k for k in ODDS_SETS if k != prefer]
    cols = set(df.columns)
    for key in order:
        h,d,a = ODDS_SETS[key]
        if h in cols and d in cols and a in cols:
            return key, (h,d,a)
    # Case-insensitive fallback
    lowmap = {c.lower().strip(): c for c in df.columns}
    for key in order:
        h,d,a = ODDS_SETS[key]
        if h.lower() in lowmap and d.lower() in lowmap and a.lower() in lowmap:
            return key, (lowmap[h.lower()], lowmap[d.lower()], lowmap[a.lower()])
    raise SystemExit("[ERROR] Could not find any known 1X2 odds columns (Avg*, PS*, B365*, WH*, Max*).")

def compute_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_home_market"] = 1.0 / out["odd_home"]
    out["p_draw_market"] = 1.0 / out["odd_draw"]
    out["p_away_market"] = 1.0 / out["odd_away"]
    out["overround_wdl"] = out["p_home_market"] + out["p_draw_market"] + out["p_away_market"]
    # Margin-adjusted "fair" probs
    out["p_home_fair"] = out["p_home_market"] / out["overround_wdl"]
    out["p_draw_fair"] = out["p_draw_market"] / out["overround_wdl"]
    out["p_away_fair"] = out["p_away_market"] / out["overround_wdl"]
    return out

def infer_result_simple(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: "H" if r["home_goals"] > r["away_goals"] else ("A" if r["home_goals"] < r["away_goals"] else "D"), axis=1)

# ------------------------- Temporal context ----------------------------------
def dense_round_index(df):
    # Dense rank of dates per season (1..max)
    df = df.copy()
    df["date_only"] = df["date"].dt.normalize()
    round_map = (df.drop_duplicates(["season","date_only"])
                   .sort_values(["season","date_only"])
                   .assign(round=lambda x: x.groupby("season").cumcount()+1)
                   [["season","date_only","round"]])
    df = df.merge(round_map, on=["season","date_only"], how="left")
    df = df.drop(columns=["date_only"])
    return df

def team_temporal_features(frame, prefix):
    # frame: columns ['row_id','date','team']
    g = frame.sort_values(["team","date"]).copy()
    g["prev_date"] = g.groupby("team")["date"].shift(1)
    g[f"rest_days_{prefix}"] = (g["date"] - g["prev_date"]).dt.days

    # --- Rolling count de partidos en los 14 días previos (sin incluir el actual) ---
    # Hacemos el rolling por equipo y luego merge por (team, date) para evitar MultiIndex no único.
    def rolling_14d_count(x):
        # x: subset de un solo equipo
        x = x.sort_values("date").copy()
        y = x[["team","date"]].copy()
        y = y.set_index("date")
        y["cnt"] = 1
        rc = y["cnt"].rolling("14D", closed="left").sum().reset_index()
        rc["team"] = x["team"].iloc[0]
        rc.rename(columns={"cnt": f"matches_last14_{prefix}"}, inplace=True)
        return rc

    rc = (g.groupby("team", group_keys=False)
          .apply(rolling_14d_count))  # devuelve columnas: ['date','matches_last14_*','team']

    # Merge back por (team, date)
    g = g.merge(rc, on=["team","date"], how="left")

    # Mantén sólo lo necesario para regresar al dataset original
    keep = ["row_id", f"rest_days_{prefix}", f"matches_last14_{prefix}"]
    return g[keep]


def add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=False).rename(columns={"index":"row_id"})
    out = dense_round_index(out)
    out["max_round"] = out.groupby("season")["round"].transform("max")
    out["is_first_half"] = out["round"] <= (out["max_round"]/2.0)
    out["season_half"] = np.where(out["is_first_half"], "H1", "H2")

    season_start = out.groupby("season")["date"].transform("min")
    out["days_since_season_start"] = (out["date"] - season_start).dt.days

    home_long = out[["row_id","date","home_team"]].rename(columns={"home_team":"team"}).copy()
    away_long = out[["row_id","date","away_team"]].rename(columns={"away_team":"team"}).copy()

    home_feats = team_temporal_features(home_long, "home")
    away_feats = team_temporal_features(away_long, "away")

    out = out.merge(home_feats, on="row_id", how="left").merge(away_feats, on="row_id", how="left")

    # Fill initial NaNs
    out["rest_days_home"] = out["rest_days_home"].fillna(7)
    out["rest_days_away"] = out["rest_days_away"].fillna(7)
    out["matches_last14_home"] = out["matches_last14_home"].fillna(0)
    out["matches_last14_away"] = out["matches_last14_away"].fillna(0)

    return out.drop(columns=["row_id"])

# ------------------------------ Main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer_odds", choices=list(ODDS_SETS.keys()), default="Avg",
                    help="Preferred 1X2 odds set (default: Avg)")
    ap.add_argument("--out", default="premier_dataset_final.csv", help="Output CSV filename")
    args = ap.parse_args()

    try:
        import kagglehub
    except Exception:
        print("[ERROR] kagglehub not found. Install with: pip install kagglehub", file=sys.stderr)
        raise

    print("Downloading dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("chrislekas/european-football-dataset-europes-top-5-leagues")
    dataset_dir = Path(dataset_path)
    print("Dataset dir:", dataset_dir)

    csv_path = pick_csv(dataset_dir)
    print("Using CSV:", csv_path)

    df = pd.read_csv(csv_path)

    # Check core columns
    required = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing required columns: {missing}")

    # Filter Premier League
    work = df.copy()
    if "League" in work.columns:
        mask = work["League"].astype(str).str.contains("premier", case=False, na=False) |                work.get("Div", pd.Series([""]*len(work))).astype(str).isin(["E0","EPL"])
    else:
        fuzzy_cols = [c for c in ["Div","League","Country","Competition"] if c in work.columns]
        fuzzy_mask = pd.Series(False, index=work.index)
        for c in fuzzy_cols:
            fuzzy_mask = fuzzy_mask | work[c].astype(str).str.contains("premier|england", case=False, na=False)
        mask = fuzzy_mask | work.get("Div", pd.Series([""]*len(work))).astype(str).isin(["E0","EPL"])
    work = work.loc[mask].copy()

    # Pick odds
    chosen_tag, (h_col,d_col,a_col) = pick_odds_columns(work, prefer=args.prefer_odds)

    # Base frame
    base = work[["Date","Season","HomeTeam","AwayTeam","FTHG","FTAG", h_col, d_col, a_col]].copy()
    base.columns = ["date","season","home_team","away_team","home_goals","away_goals","odd_home","odd_draw","odd_away"]

    # Result
    if "FTR" in work.columns:
        res = work["FTR"].astype(str).str.upper().str[0].map({"H":"H","D":"D","A":"A"})
        base["result"] = res.fillna(infer_result_simple(base))
    else:
        base["result"] = infer_result_simple(base)

    # Types
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    for c in ["home_goals","away_goals","odd_home","odd_draw","odd_away"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    base = base.dropna(subset=["date","home_team","away_team","home_goals","away_goals","odd_home","odd_draw","odd_away"])

    # Optional pre-match features from dataset (if present)
    # Elo
    if "home_elo_before" in work.columns:
        base["home_elo_before"] = work.loc[base.index, "home_elo_before"].values
    if "away_elo_before" in work.columns:
        base["away_elo_before"] = work.loc[base.index, "away_elo_before"].values
    if "elo_diff" in work.columns:
        base["elo_diff"] = work.loc[base.index, "elo_diff"].values
    if "home_elo_trend_5" in work.columns:
        base["home_elo_trend_5"] = work.loc[base.index, "home_elo_trend_5"].values
    if "elo_change_home" in work.columns:
        base["elo_change_home"] = work.loc[base.index, "elo_change_home"].values
    if "elo_change_away" in work.columns:
        base["elo_change_away"] = work.loc[base.index, "elo_change_away"].values

    # Form / rolling
    optional_map = [
        ("Home_Points_last5","home_points_last5"),
        ("Away_Points_last5","away_points_last5"),
        ("form_diff_points_last5","form_diff_points_last5"),
        ("home_goals_scored_rolling_mean_5","home_goals_scored_rm5"),
        ("away_goals_conceded_rolling_mean_5","away_goals_conceded_rm5"),
        ("home_form_ratio","home_form_ratio"),
        ("away_form_ratio","away_form_ratio"),
        ("home_gd_roll5","home_gd_roll5"),
        ("away_gd_roll5","away_gd_roll5"),
        ("avg_goal_diff_last5","avg_goal_diff_last5"),
        ("Home_CleanSheets_last5","home_cleansheets_last5"),
        ("Away_CleanSheets_last5","away_cleansheets_last5"),
        ("Home_BTTS_last5","home_btts_last5"),
    ]
    for src, dst in optional_map:
        if src in work.columns:
            base[dst] = work.loc[base.index, src].values

    # Market probs
    base = compute_market_probs(base)

    # Temporal context
    final = add_temporal_context(base)

    # Save
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / args.out
    final.to_csv(out_path, index=False, encoding="utf-8")

    print("✅ Saved:", out_path.resolve())
    print("Rows:", len(final))
    print("Chosen odds set:", chosen_tag)
    print("Columns:", list(final.columns))

if __name__ == "__main__":
    main()
