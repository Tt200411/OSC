#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_path: str
    data_path: str
    target: str
    freq: str
    seq_len: int
    label_len: int
    horizons: tuple[int, ...]
    alternate_paths: tuple[str, ...] = ()


DATASET_SPECS = {
    "Weather": DatasetSpec(
        name="Weather",
        root_path="../Weather",
        data_path="weather.csv",
        target="OT",
        freq="t",
        seq_len=96,
        label_len=48,
        horizons=(96, 192, 336),
        alternate_paths=(
            "../dataset/weather/weather.csv",
            "../weather/weather.csv",
            "./dataset/weather/weather.csv",
            "./Weather/weather.csv",
        ),
    ),
    "Exchange": DatasetSpec(
        name="Exchange",
        root_path="../Exchange",
        data_path="exchange_rate.csv",
        target="OT",
        freq="d",
        seq_len=96,
        label_len=48,
        horizons=(96, 192, 336),
        alternate_paths=(
            "../dataset/exchange_rate/exchange_rate.csv",
            "../exchange_rate/exchange_rate.csv",
            "./dataset/exchange_rate/exchange_rate.csv",
            "./Exchange/exchange_rate.csv",
        ),
    ),
    "ILI": DatasetSpec(
        name="ILI",
        root_path="../ILI",
        data_path="national_illness.csv",
        target="OT",
        freq="w",
        seq_len=36,
        label_len=18,
        horizons=(24, 36, 48),
        alternate_paths=(
            "../dataset/illness/national_illness.csv",
            "../illness/national_illness.csv",
            "./dataset/illness/national_illness.csv",
            "./ILI/national_illness.csv",
        ),
    ),
}


def markdown_table(frame):
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = ["" if pd.isna(row[col]) else str(row[col]) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def resolve_csv(project_root, spec, explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.append(os.path.join(spec.root_path, spec.data_path))
    candidates.extend(spec.alternate_paths)

    checked = []
    for candidate in candidates:
        path = candidate if os.path.isabs(candidate) else os.path.join(project_root, candidate)
        path = os.path.abspath(path)
        checked.append(path)
        if os.path.exists(path):
            return path, checked
    return "", checked


def infer_frequency(dates):
    parsed = pd.to_datetime(dates, errors="coerce")
    parsed = parsed.dropna().sort_values()
    if len(parsed) < 3:
        return {"median_delta": "", "mode_delta": "", "mode_delta_rate": np.nan}
    deltas = parsed.diff().dropna()
    if deltas.empty:
        return {"median_delta": "", "mode_delta": "", "mode_delta_rate": np.nan}
    mode_delta = deltas.mode().iloc[0]
    return {
        "median_delta": str(deltas.median()),
        "mode_delta": str(mode_delta),
        "mode_delta_rate": float((deltas == mode_delta).mean()),
    }


def inspect_csv(path, spec):
    frame = pd.read_csv(path)
    columns = frame.columns.tolist()
    numeric_cols = [col for col in columns if col != "date" and pd.api.types.is_numeric_dtype(frame[col])]
    target_exists = spec.target in columns
    date_exists = "date" in columns
    target_nan = int(frame[spec.target].isna().sum()) if target_exists else -1
    total_nan = int(frame.isna().sum().sum())
    freq = infer_frequency(frame["date"]) if date_exists else {"median_delta": "", "mode_delta": "", "mode_delta_rate": np.nan}
    reasons = []
    if not date_exists:
        reasons.append("missing date column")
    if not target_exists:
        reasons.append(f"missing target {spec.target}")
    if len(numeric_cols) == 0:
        reasons.append("no numeric feature columns")
    if target_nan > 0:
        reasons.append(f"{spec.target} has {target_nan} NaN")
    if total_nan > 0:
        reasons.append(f"file has {total_nan} total NaN")
    if len(frame) < spec.seq_len + max(spec.horizons) + 2:
        reasons.append("too few rows for largest smoke horizon")
    return {
        "rows": int(len(frame)),
        "columns": int(len(columns)),
        "numeric_columns": int(len(numeric_cols)),
        "has_date": bool(date_exists),
        "has_target": bool(target_exists),
        "target": spec.target,
        "target_nan": target_nan,
        "total_nan": total_nan,
        "median_delta": freq["median_delta"],
        "mode_delta": freq["mode_delta"],
        "mode_delta_rate": freq["mode_delta_rate"],
        "schema_ok": not reasons,
        "schema_reason": "; ".join(reasons),
    }


def loader_smoke(path, spec, lee_root):
    root_path = os.path.dirname(path)
    data_path = os.path.basename(path)
    sys.path.insert(0, os.path.abspath(lee_root))
    from data.data_loader import Dataset_Custom

    rows = []
    for pred_len in spec.horizons:
        size = [spec.seq_len, spec.label_len, pred_len]
        for flag in ["train", "val", "test"]:
            row = {"pred_len": pred_len, "flag": flag}
            try:
                dataset = Dataset_Custom(
                    root_path=root_path,
                    data_path=data_path,
                    flag=flag,
                    size=size,
                    features="M",
                    target=spec.target,
                    inverse=False,
                    timeenc=1,
                    freq=spec.freq,
                    cols=None,
                )
                length = len(dataset)
                row["loader_ok"] = bool(length > 0)
                row["length"] = int(length)
                if length > 0:
                    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
                    row["seq_x_shape"] = "x".join(map(str, seq_x.shape))
                    row["seq_y_shape"] = "x".join(map(str, seq_y.shape))
                    row["seq_x_mark_shape"] = "x".join(map(str, seq_x_mark.shape))
                    row["seq_y_mark_shape"] = "x".join(map(str, seq_y_mark.shape))
                else:
                    row["reason"] = "non-positive dataset length"
            except Exception as exc:
                row["loader_ok"] = False
                row["length"] = 0
                row["reason"] = repr(exc)
            rows.append(row)
    return rows


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    project_root = os.path.abspath(args.project_root)
    dataset_names = args.datasets or list(DATASET_SPECS)
    explicit = {
        "Weather": args.weather_csv,
        "Exchange": args.exchange_csv,
        "ILI": args.ili_csv,
    }

    schema_rows = []
    loader_rows = []
    path_rows = []
    for name in dataset_names:
        spec = DATASET_SPECS[name]
        path, checked = resolve_csv(project_root, spec, explicit_path=explicit.get(name))
        path_rows.append({"dataset": name, "resolved_path": path, "checked_paths": " ; ".join(checked)})
        if not path:
            schema_rows.append(
                {
                    "dataset": name,
                    "path": "",
                    "exists": False,
                    "schema_ok": False,
                    "schema_reason": "missing dataset file",
                }
            )
            continue
        row = {"dataset": name, "path": path, "exists": True, "freq": spec.freq, "horizons": ",".join(map(str, spec.horizons))}
        row.update(inspect_csv(path, spec))
        schema_rows.append(row)
        if args.loader_smoke and row["schema_ok"]:
            for smoke in loader_smoke(path, spec, args.lee_root):
                loader_rows.append({"dataset": name, "path": path, **smoke})

    schema = pd.DataFrame(schema_rows)
    loaders = pd.DataFrame(loader_rows)
    paths = pd.DataFrame(path_rows)
    schema_path = os.path.join(args.output_dir, "phase5_new_dataset_schema.csv")
    loader_path = os.path.join(args.output_dir, "phase5_new_dataset_loader_smoke.csv")
    path_path = os.path.join(args.output_dir, "phase5_new_dataset_path_candidates.csv")
    report_path = os.path.join(args.output_dir, "phase5_new_dataset_preflight.md")
    json_path = os.path.join(args.output_dir, "phase5_new_dataset_preflight.json")
    schema.to_csv(schema_path, index=False)
    loaders.to_csv(loader_path, index=False)
    paths.to_csv(path_path, index=False)
    payload = {
        "schema_path": schema_path,
        "loader_path": loader_path,
        "path_path": path_path,
        "report_path": report_path,
        "datasets": dataset_names,
        "all_schema_ok": bool(schema["schema_ok"].fillna(False).all()) if not schema.empty else False,
        "all_loader_ok": bool(loaders["loader_ok"].fillna(False).all()) if not loaders.empty else None,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Phase-5 New Dataset Preflight\n\n")
        handle.write("## Schema\n\n")
        handle.write(markdown_table(schema) + "\n\n")
        handle.write("## Loader Smoke\n\n")
        handle.write(markdown_table(loaders) + "\n\n")
        handle.write("## Path Candidates\n\n")
        handle.write(markdown_table(paths) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and (not payload["all_schema_ok"] or payload["all_loader_ok"] is False):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Phase-5 local schema and Dataset_Custom loader preflight")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--output_dir", default=".aris/phase5_new_dataset_preflight")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=None)
    parser.add_argument("--weather_csv", default=None)
    parser.add_argument("--exchange_csv", default=None)
    parser.add_argument("--ili_csv", default=None)
    parser.add_argument("--loader_smoke", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
