#!/usr/bin/env python3
import argparse
import os

import pandas as pd

from generate_factors import build_factor_table


DATASETS = {
    "ETTh1": {"path": "ETT-small/ETTh1.csv", "target": "OT", "points_per_day": 24},
    "ETTh2": {"path": "ETT-small/ETTh2.csv", "target": "OT", "points_per_day": 24},
    "ETTm1": {"path": "ETT-small/ETTm1.csv", "target": "OT", "points_per_day": 96},
    "ETTm2": {"path": "ETT-small/ETTm2.csv", "target": "OT", "points_per_day": 96},
    "Solar1": {"path": "../Solar/Site_1_50MW.csv", "target": "Power", "points_per_day": 96},
    "Solar2": {"path": "../Solar/Site_2_130MW.csv", "target": "Power", "points_per_day": 96},
    "Solar3": {"path": "../Solar/Site_3_30MW.csv", "target": "Power", "points_per_day": 96},
    "Solar4": {"path": "../Solar/Site_4_130MW.csv", "target": "Power", "points_per_day": 96},
    "Solar5": {"path": "../Solar/Site_5_110MW.csv", "target": "Power", "points_per_day": 96},
    "Solar6": {"path": "../Solar/Site_6_35MW.csv", "target": "Power", "points_per_day": 96},
    "Solar7": {"path": "../Solar/Site_7_30MW.csv", "target": "Power", "points_per_day": 96},
    "Solar8": {"path": "../Solar/Site_8_30MW.csv", "target": "Power", "points_per_day": 96},
    "Weather": {"path": "../Weather/weather.csv", "target": "OT", "points_per_day": 144},
    "Exchange": {"path": "../Exchange/exchange_rate.csv", "target": "OT", "points_per_day": 1},
    "ILI": {"path": "../ILI/national_illness.csv", "target": "OT", "points_per_day": 1},
}


def main():
    parser = argparse.ArgumentParser(description="Generate Phase-4 factor CSV files for all ETT and Solar sites")
    parser.add_argument("--dataset", choices=DATASETS.keys(), action="append")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_days", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--method", choices=["sum", "mean"], default="sum")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    selected = args.dataset or list(DATASETS.keys())
    all_frames = []

    for dataset in selected:
        meta = DATASETS[dataset]
        csv_path = os.path.normpath(os.path.join(root, meta["path"]))
        factors = build_factor_table(
            dataset=dataset,
            csv_path=csv_path,
            target=meta["target"],
            points_per_day=meta["points_per_day"],
            reference_days=args.reference_days,
            alpha=args.alpha,
            beta=args.beta,
            method=args.method,
        )
        output_path = os.path.join(args.output_dir, f"{dataset}_factors.csv")
        factors.to_csv(output_path, index=False)
        all_frames.append(factors)
        print(f"Wrote {output_path}: {len(factors)} rows")

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(args.output_dir, "all_factors.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}: {len(combined)} rows")


if __name__ == "__main__":
    main()
