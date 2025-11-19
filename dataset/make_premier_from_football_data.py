#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Premier League (E0) 1X2 dataset using *real* bookmaker odds from Football-Data.co.uk.

Key features
------------
- Pulls one or many CSVs (by season code or direct URLs) from Football-Data.
- Robust odds selection: prefers AvgH/AvgD/AvgA (market average) with fallbacks (PS, B365, WH, Max).
- Cleans dates (UK dd/mm/yyyy) and optionally merges `Date` + `Time` into UTC-naive datetimes.
- Computes market implied probabilities and fair (margin-corrected) probabilities.
- Adds temporal context (round index per season date, season halves, days since season start).
- Adds rest days and "matches in last 14 days" for each team, home and away (no leakage).
- Saves a tidy CSV ready for modeling.

Usage examples
--------------
# 1) One season (by code '2425' -> 2024/2025)
python make_premier_from_football_data.py --seasons 2425 --out premier_dataset_final.csv

# 2) Multiple seasons (2019/2020 through 2024/2025)
python make_premier_from_football_data.py --seasons 1920 2021 2122 2223 2324 2425 --out premier_2019_2025.csv

# 3) Direct URLs (mixed allowed)
python make_premier_from_football_data.py --urls https://www.football-data.co.uk/mmz4281/2425/E0.csv https://www.football-data.co.uk/mmz4281/2324/E0.csv

Notes
-----
- This script only uses Football-Data (free, redistributable). It does NOT require Kaggle.
- Football-Data snapshots are "opening/closing/average" style and not tick-by-tick. Treat them as pre-match market info, but they may not include exact timestamps per snapshot.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import io
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests

BASE = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# Preferred odds set order
ODDS_SETS = [
    ("AvgH","AvgD","AvgA"),   # market average
    ("PSH","PSD","PSA"),      # Pinnacle (often present)
    ("B365H","B365D","B365A"),
    ("WHH","WHD","WHA"),
    ("MaxH","MaxD","MaxA"),   # max across books (can be noisy)
]

@dataclass
class Args:
    seasons: List[str]
    urls: List[str]
    out: str
    prefer_time: bool
    prefer_odds: Optional[str]

def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Premier League dataset builder (Football-Data)")
    ap.add_argument("--seasons", nargs="*", default=[], help="Season codes like 2425 (-> 2024/2025).")
    ap.add_argument("--urls", nargs="*", default=[], help="Direct Football-Data URLs to CSVs.")
    ap.add_argument("--out", default="premier_dataset_final.csv", help="Output CSV filename.")
    ap.add_argument("--prefer_time", action="store_true", help="If set, combine Date+Time when Time exists.")
    ap.add_argument("--prefer_odds", choices=["Avg","PS","B365","WH","Max"], default=None,
                    help="Force one odds set (fallbacks still used if missing).")
    ns = ap.parse_args()

    return Args(
        seasons=ns.seasons,
        urls=ns.urls,
        out=ns.out,
        prefer_time=ns.prefer_time,
        prefer_odds=ns.prefer_odds,
    )

def build_urls(args: Args) -> List[str]:
    urls = list(args.urls) if args.urls else []
    for s in args.seasons:
        s = str(s).strip()
        if len(s) != 4 or not s.isdigit():
            print(f"[WARN] Ignoring invalid season code: {s}", file=sys.stderr)
            continue
        urls.append(BASE.format(season=s))
    return urls

def fetch_csv(url: str, retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            # Football-Data CSVs can be latin-1 or utf-8; let pandas infer
            data = r.content
            return pd.read_csv(io.BytesIO(data))
        except Exception as e:
            last_err = e
            print(f"[WARN] Fetch failed ({i+1}/{retries}) for {url}: {e}", file=sys.stderr)
            time.sleep(1 + i)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def best_odds_columns(cols: List[str], prefer: Optional[str]) -> Tuple[str,str,str,str]:
    """
    Returns: (tag, home_col, draw_col, away_col)
    tag ∈ {"Avg","PS","B365","WH","Max"} based on actual chosen set.
    """
    cols_set = set(cols)
    # Custom order if --prefer_odds
    order_map = {"Avg":0,"PS":1,"B365":2,"WH":3,"Max":4}
    ordered = sorted(ODDS_SETS, key=lambda t: order_map.get(prefer_from_tuple(t), 99)) if prefer else ODDS_SETS

    for h,d,a in ordered:
        if h in cols_set and d in cols_set and a in cols_set:
            return (prefer_from_tuple((h,d,a)), h, d, a)

    # last resort: case-insensitive match
    low = {c.lower(): c for c in cols}
    def find_exact(name: str) -> Optional[str]:
        return low.get(name.lower())

    for h,d,a in ordered:
        H, D, A = find_exact(h), find_exact(d), find_exact(a)
        if H and D and A:
            return (prefer_from_tuple((h,d,a)), H, D, A)

    raise ValueError("No known odds columns found (tried Avg, PS, B365, WH, Max).")

def prefer_from_tuple(tup: Tuple[str,str,str]) -> str:
    key = tup[0][:3]  # "Avg", "PSH" -> "PSH"[:3] == "PSH" but we map
    if key.lower().startswith("avg"): return "Avg"
    if key.upper().startswith("PS"): return "PS"
    if key.upper().startswith("B36"): return "B365"
    if key.upper().startswith("WH"): return "WH"
    if key.lower().startswith("max"): return "Max"
    return "Avg"

def parse_date(df: pd.DataFrame, use_time: bool) -> pd.Series:
    """
    Football-Data dates are dd/mm/yy or dd/mm/yyyy; Time is local (UK). We keep naive timestamps.
    """
    date_raw = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if use_time and "Time" in df.columns:
        tm = pd.to_datetime(df["Time"].astype(str), format="%H:%M", errors="coerce").dt.time
        # Merge date + time
        out = pd.to_datetime(date_raw.dt.date.astype(str) + " " + df["Time"].astype(str), errors="coerce", dayfirst=True)
        # Fallback to date-only if merge failed
        out = out.fillna(date_raw)
        return out
    return date_raw

def normalize_teams(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def infer_result(r) -> str:
    if r["home_goals"] > r["away_goals"]:
        return "H"
    if r["home_goals"] < r["away_goals"]:
        return "A"
    return "D"

def compute_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_home_market"] = 1.0 / out["odd_home"]
    out["p_draw_market"] = 1.0 / out["odd_draw"]
    out["p_away_market"] = 1.0 / out["odd_away"]
    out["overround_wdl"] = out["p_home_market"] + out["p_draw_market"] + out["p_away_market"]
    # "Fair" probs (normalize by overround)
    out["p_home_fair"] = out["p_home_market"] / out["overround_wdl"]
    out["p_draw_fair"] = out["p_draw_market"] / out["overround_wdl"]
    out["p_away_fair"] = out["p_away_market"] / out["overround_wdl"]
    return out

def dense_round_index(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["date_only"] = g["date"].dt.normalize()
    round_map = (g.drop_duplicates(["season","date_only"])
                   .sort_values(["season","date_only"])
                   .assign(round=lambda x: x.groupby("season").cumcount()+1)[["season","date_only","round"]])
    g = g.merge(round_map, on=["season","date_only"], how="left")
    g.drop(columns=["date_only"], inplace=True)
    return g

def team_temporal_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # frame: ['row_id','date','team']
    g = frame.sort_values(["team","date"]).copy()
    g["prev_date"] = g.groupby("team")["date"].shift(1)
    g[f"rest_days_{prefix}"] = (g["date"] - g["prev_date"]).dt.days

    def rolling_14d_count(sub: pd.DataFrame) -> pd.DataFrame:
        s = sub.sort_values("date")[["team","date"]].copy()
        s = s.set_index("date")
        s["one"] = 1
        rc = s["one"].rolling("14D", closed="left").sum().reset_index()
        rc["team"] = sub["team"].iloc[0]
        rc.rename(columns={"one": f"matches_last14_{prefix}"}, inplace=True)
        return rc

    rc = (g.groupby("team", group_keys=False)
            .apply(rolling_14d_count))

    g = g.merge(rc, on=["team","date"], how="left")
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

    home_long = out[["row_id","date","home_team"]].rename(columns={"home_team":"team"})
    away_long = out[["row_id","date","away_team"]].rename(columns={"away_team":"team"})

    home_feats = team_temporal_features(home_long, "home")
    away_feats = team_temporal_features(away_long, "away")

    out = out.merge(home_feats, on="row_id", how="left").merge(away_feats, on="row_id", how="left")

    # Fill initial NaNs with neutral defaults
    out["rest_days_home"] = out["rest_days_home"].fillna(7)
    out["rest_days_away"] = out["rest_days_away"].fillna(7)
    out["matches_last14_home"] = out["matches_last14_home"].fillna(0)
    out["matches_last14_away"] = out["matches_last14_away"].fillna(0)

    return out.drop(columns=["row_id"])

def season_code_to_label(code: str) -> str:
    """
    '2425' -> '2024/2025'
    '1920' -> '2019/2020'
    """
    if len(code) != 4 or not code.isdigit():
        return code
    y1 = int(code[:2])
    y2 = int(code[2:])
    y1 += 2000 if y1 < 50 else 1900
    y2 += 2000 if y2 < 50 else 1900
    return f"{y1}/{y2}"

def process_csv(df: pd.DataFrame, season_hint: Optional[str], prefer_time: bool, prefer_odds: Optional[str]) -> pd.DataFrame:
    # Ensure required columns
    need = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # Focus E0 (English Premier League). Most E0 files already are EPL, but keep filter for safety.
    if "Div" in df.columns:
        df = df[df["Div"].astype(str).str.upper().eq("E0")].copy()

    tag, h_col, d_col, a_col = best_odds_columns(list(df.columns), prefer_odds)

    base = df[["Date","HomeTeam","AwayTeam","FTHG","FTAG", h_col, d_col, a_col]].copy()
    base.columns = ["date_raw","home_team","away_team","home_goals","away_goals","odd_home","odd_draw","odd_away"]

    # Season label
    if "Season" in df.columns and df["Season"].notna().any():
        season = df["Season"]
        # Normalize and forward-fill a single string value if needed
        if season.dtype != object:
            season = season.astype(str)
        season = season.replace({"nan": np.nan}).ffill().bfill().astype(str)
        base["season"] = season.values[:len(base)]
    else:
        base["season"] = season_code_to_label(season_hint) if season_hint else ""

    # Parse date
    # Optionally combine Date + Time
    if prefer_time and "Time" in df.columns:
        tmp = df[["Date","Time"]].copy()
        tmp.columns = ["Date","Time"]
        base["date"] = parse_date(tmp, use_time=True)
    else:
        tmp = df[["Date"]].copy()
        base["date"] = parse_date(tmp, use_time=False)

    # Clean types
    base["home_team"] = normalize_teams(base["home_team"])
    base["away_team"] = normalize_teams(base["away_team"])

    for c in ["home_goals","away_goals","odd_home","odd_draw","odd_away"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # Drop rows missing core fields
    base = base.dropna(subset=["date","home_team","away_team","home_goals","away_goals","odd_home","odd_draw","odd_away"])

    # Result
    if "FTR" in df.columns:
        ftr = df["FTR"].astype(str).str.upper().str[0].map({"H":"H","D":"D","A":"A"})
        base["result"] = ftr.values[:len(base)]
    else:
        base["result"] = base.apply(infer_result, axis=1)

    # Probabilities
    base = compute_market_probs(base)

    # Temporal context (per season)
    base["season"] = base["season"].replace("", np.nan)
    # If some rows lack season, infer from year of date
    if base["season"].isna().any():
        inferred = base["date"].dt.year.astype(str) + "/" + (base["date"].dt.year.add(1)).astype(str)
        base["season"] = base["season"].fillna(inferred)

    final = add_temporal_context(base)

    # Select canonical columns (align with your modeling pipeline)
    ordered_cols = [
        "date","season","home_team","away_team",
        "home_goals","away_goals",
        "odd_home","odd_draw","odd_away",
        "result",
        "p_home_market","p_draw_market","p_away_market",
        "overround_wdl","p_home_fair","p_draw_fair","p_away_fair",
        "round","max_round","is_first_half","season_half","days_since_season_start",
        "rest_days_home","matches_last14_home","rest_days_away","matches_last14_away",
    ]

    # Clip overround to sensible bounds to avoid extreme floating artifacts
    final["overround_wdl"] = final["overround_wdl"].clip(lower=1.0, upper=1.25)

    # Keep only columns present
    existing = [c for c in ordered_cols if c in final.columns]
    final = final[existing].sort_values(["date","home_team","away_team"]).reset_index(drop=True)

    final.attrs["odds_tag"] = tag
    return final

def _build_from_args(args: Args) -> pd.DataFrame:
    urls = build_urls(args)
    if not urls:
        print("[ERROR] Provide --seasons and/or --urls (Football-Data).", file=sys.stderr)
        sys.exit(2)

    frames = []
    for u in urls:
        # Season hint from URL if possible
        season_hint = None
        try:
            # .../mmz4281/2425/E0.csv -> "2425"
            parts = u.strip("/").split("/")
            season_hint = parts[-2] if len(parts) >= 2 and parts[-1].lower().endswith(".csv") else None
        except Exception:
            season_hint = None

        print(f"[INFO] Fetching {u}")
        df = fetch_csv(u)
        part = process_csv(df, season_hint=season_hint, prefer_time=args.prefer_time, prefer_odds=args.prefer_odds)
        frames.append(part)
        print(f"[OK] Rows: {len(part)} | Chosen odds set: {part.attrs.get('odds_tag','?')}")

    if not frames:
        print("[ERROR] No data frames processed.", file=sys.stderr)
        sys.exit(3)

    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date","home_team","away_team"])
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved: {out_path}")
    print(f"Rows: {len(all_df)}")
    print("Columns:", list(all_df.columns))
    return all_df


def build_premier_dataset(
        seasons: List[str],
        urls: Optional[List[str]] = None,
        out: str = "premier_dataset_final.csv",
        prefer_time: bool = False,
        prefer_odds: Optional[str] = None,
) -> pd.DataFrame:
    if urls is None:
        urls = []

    args = Args(
        seasons=seasons,
        urls=urls,
        out=out,
        prefer_time=prefer_time,
        prefer_odds=prefer_odds,
    )
    return _build_from_args(args)


def main():
    args = parse_args()
    _build_from_args(args)


if __name__ == "__main__":
    main()
