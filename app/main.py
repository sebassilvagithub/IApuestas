#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path

from dataset.make_premier_from_football_data import build_premier_dataset


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Front principal: construye el dataset de Premier usando el módulo dataset."
    )

    parser.add_argument(
        "--seasons",
        nargs="*",
        default=[],
        help="Season codes like 2425 (-> 2024/2025).",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="Direct Football-Data URLs to CSVs.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Nombre del archivo dentro de data/input. "
            "Si no se pasa, se usa dataset-football-YYYYMMDD.csv."
        ),
    )
    parser.add_argument(
        "--prefer_time",
        action="store_true",
        help="If set, combine Date+Time when Time exists.",
    )
    parser.add_argument(
        "--prefer_odds",
        choices=["Avg", "PS", "B365", "WH", "Max"],
        default=None,
        help="Force one odds set (fallbacks still used if missing).",
    )
    return parser


def resolve_output_path(ns: Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "input"
    data_dir.mkdir(parents=True, exist_ok=True)

    if ns.out:
        out_name = ns.out
    else:
        today_str = date.today().strftime("%Y%m%d")
        out_name = f"dataset-football-{today_str}.csv"

    return data_dir / out_name


def run(ns: Namespace) -> None:
    out_path = resolve_output_path(ns)

    build_premier_dataset(
        seasons=ns.seasons,
        urls=ns.urls,
        out=str(out_path),
        prefer_time=ns.prefer_time,
        prefer_odds=ns.prefer_odds,
    )

    print(f"\n✅ Dataset generado en: {out_path}")


def main() -> None:
    parser = create_parser()
    ns = parser.parse_args()
    run(ns)

if __name__ == "__main__": main()
