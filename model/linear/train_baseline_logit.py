#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline limpio (sin fuga): Logistic Regression 1X2 con split temporal H1->H2.

Usa SOLO features 100% pre-match:
- Mercado: odd_*, p_*_market, p_*_fair, overround_wdl
- Elo previo: home_elo_before, away_elo_before, elo_diff, home_elo_trend_5
- Temporal (Premier): rest_days_*, matches_last14_*, days_since_season_start, round, max_round, is_first_half

Ejemplo:
  python train_baseline_logit_clean.py \
    --csv ../../dataset/data/premier_dataset_final.csv \
    --season latest \
    --ev_threshold 1.02 \
    --kelly_frac 0.25 \
    --outdir ../../dataset/outputs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

TARGET_MAP = {"H":0, "D":1, "A":2}
REV_TARGET_MAP = {v:k for k,v in TARGET_MAP.items()}

# ---------------------- Feature whitelist (pre-match seguras) -----------------

SAFE_COLS_EXACT = {
    # Mercado
    "odd_home","odd_draw","odd_away",
    "p_home_market","p_draw_market","p_away_market",
    "p_home_fair","p_draw_fair","p_away_fair",
    "overround_wdl",
    # Elo PRE partido
    "home_elo_before","away_elo_before","elo_diff","home_elo_trend_5",
    # Temporal (premier-only)
    "rest_days_home","rest_days_away",
    "matches_last14_home","matches_last14_away",
    "days_since_season_start","round","max_round","is_first_half",
}

ID_COLS = {"date","season","home_team","away_team"}
LEAK_COLS = {"home_goals","away_goals","result"}

def pick_features(df: pd.DataFrame):
    cols = [c for c in SAFE_COLS_EXACT if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return cols

# ---------------------- Bankroll simulation -----------------------------------

def kelly_fraction(p: float, o: float, frac: float = 0.25) -> float:
    raw = (o*p - 1) / (o - 1)
    raw = max(0.0, min(raw, 1.0))
    return frac * raw

def simulate_bankroll(test_df: pd.DataFrame, ev_threshold: float = 1.02, kelly_frac: float = 0.25, bankroll0: float = 1000.0):
    odds_cols = {"H":"odd_home", "D":"odd_draw", "A":"odd_away"}
    proba_cols = {"H":"p_H", "D":"p_D", "A":"p_A"}
    bankroll = bankroll0
    records = []
    for _, r in test_df.iterrows():
        evs = {k: r[proba_cols[k]] * r[odds_cols[k]] for k in ["H","D","A"]}
        pick = max(evs, key=evs.get)
        best_ev = evs[pick]
        if best_ev <= ev_threshold:
            records.append((r["date"], r["home_team"], r["away_team"], pick, best_ev, 0.0, bankroll, False))
            continue
        o = r[odds_cols[pick]]
        p = r[proba_cols[pick]]
        f = kelly_fraction(p, o, frac=kelly_frac)
        stake = bankroll * f
        win = (r["result"] == pick)
        bankroll = bankroll - stake + (stake*o if win else 0.0)
        records.append((r["date"], r["home_team"], r["away_team"], pick, best_ev, stake, bankroll, win))
    hist = pd.DataFrame(records, columns=["date","home_team","away_team","pick","best_ev","stake","bankroll_after","win"])
    roi = (hist["bankroll_after"].iloc[-1] - bankroll0) / bankroll0 if len(hist) else 0.0
    return hist, {"n_bets": int((hist["stake"] > 0).sum()), "final_bankroll": float(hist["bankroll_after"].iloc[-1] if len(hist) else bankroll0), "roi": roi}

# ---------------------- Temporal split ----------------------------------------

def pick_season(df: pd.DataFrame, season: str):
    if season == "latest":
        last_date_per_season = df.groupby("season")["date"].max()
        season = last_date_per_season.sort_values().index[-1]
    return season

def temporal_split(df: pd.DataFrame, season: str):
    season = pick_season(df, season)
    s_df = df[df["season"] == season].copy()
    train = s_df[s_df["is_first_half"] == True].copy()
    test  = s_df[s_df["is_first_half"] == False].copy()
    return season, train, test

# ---------------------- Main --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to premier_dataset_final.csv")
    ap.add_argument("--season", default="latest", help="Season to evaluate (e.g., '2020/2021' or 'latest')")
    ap.add_argument("--ev_threshold", type=float, default=1.02, help="EV threshold to place a bet")
    ap.add_argument("--kelly_frac", type=float, default=0.25, help="Fractional Kelly (0-1)")
    ap.add_argument("--outdir", default="outputs", help="Directory to write artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])

    needed = {"season","result","odd_home","odd_draw","odd_away","is_first_half"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    feat_cols = pick_features(df)
    if not feat_cols:
        raise SystemExit("No safe feature columns found. Check your input CSV.")

    season, train, test = temporal_split(df, args.season)

    med = train[feat_cols].median()
    trainX = train[feat_cols].fillna(med)
    testX  = test[feat_cols].fillna(med)
    trainY = train["result"].map(TARGET_MAP).values
    testY  = test["result"].map(TARGET_MAP).values

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=300, random_state=42))  # multinomial por defecto en sklearn>=1.5
    ])
    pipe.fit(trainX, trainY)
    proba = pipe.predict_proba(testX)
    preds = proba.argmax(1)

    acc = accuracy_score(testY, preds)
    ll  = log_loss(testY, proba, labels=[0,1,2])

    pred_df = test.copy()
    pred_df[["p_H","p_D","p_A"]] = proba
    pred_df["pred"] = [REV_TARGET_MAP[i] for i in preds]

    # bankroll
    hist, sim = simulate_bankroll(pred_df, ev_threshold=args.ev_threshold, kelly_frac=args.kelly_frac, bankroll0=1000.0)

    # outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    season_safe = str(season).replace("/", "-").replace("\\", "-").replace(" ", "_")

    preds_path = outdir / f"predictions_clean_{season_safe}.csv"
    hist_path  = outdir / f"bankroll_clean_{season_safe}.csv"
    meta_path  = outdir / f"metrics_clean_{season_safe}.txt"

    pred_df.to_csv(preds_path, index=False)
    hist.to_csv(hist_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Season: {season}\n")
        f.write(f"SAFE Features: {len(feat_cols)} -> {feat_cols}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"LogLoss:  {ll:.4f}\n")
        f.write(f"Bets:     {sim['n_bets']}\n")
        f.write(f"Final bankroll: {sim['final_bankroll']:.2f}\n")
        f.write(f"ROI:      {sim['roi']:.2%}\n")

    print("=== Baseline Logistic CLEAN (Multinomial) ===")
    print(f"Season: {season}")
    print(f"SAFE features used ({len(feat_cols)}): {feat_cols}")
    print(f"Accuracy: {acc:.4f} | LogLoss: {ll:.4f}")
    print("=== Betting Simulation (Kelly frac={:.2f}, EV>{:.2f}) ===".format(args.kelly_frac, args.ev_threshold))
    print(f"Bets: {sim['n_bets']} | Final bankroll: {sim['final_bankroll']:.2f} | ROI: {sim['roi']:.2%}")
    print("Artifacts:\n - {}\n - {}\n - {}".format(preds_path, hist_path, meta_path))

if __name__ == "__main__":
    main()
