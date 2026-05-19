#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from phase4_common import expected_matrix, read_all_summaries, reusable_runs, validate_local_data
from phase4_report_utils import markdown_table


def write_reuse_report(path, expected, reusable, missing, data_status):
    cells = expected[["dataset", "pred_len"]].drop_duplicates().shape[0]
    reusable_cells = reusable[["dataset", "pred_len", "config_name", "seed"]].drop_duplicates().shape[0]
    missing_by_dataset = (
        missing.groupby("dataset").size().reset_index(name="missing_seed_runs").sort_values("dataset")
        if not missing.empty
        else pd.DataFrame(columns=["dataset", "missing_seed_runs"])
    )
    reusable_by_dataset = (
        reusable.groupby("dataset").size().reset_index(name="reusable_seed_runs").sort_values("dataset")
        if not reusable.empty
        else pd.DataFrame(columns=["dataset", "reusable_seed_runs"])
    )
    lines = [
        "# Phase-4 Reuse Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Coverage",
        "",
        f"- Dataset-horizon cells: {cells}",
        f"- Required seed-runs: {len(expected)}",
        f"- Reusable seed-runs: {len(reusable)}",
        f"- Missing seed-runs: {len(missing)}",
        f"- Reusable unique seed cells: {reusable_cells}",
        "",
        "## Local Data Integrity",
        "",
        markdown_table(data_status),
        "",
        "## Reusable By Dataset",
        "",
        markdown_table(reusable_by_dataset),
        "",
        "## Missing By Dataset",
        "",
        markdown_table(missing_by_dataset),
        "",
        "## Reuse Rules Applied",
        "",
        "- Matched canonical key: dataset, horizon, protocol, activation signature, seed.",
        "- Required protocol: batch_size=32, train_epochs=6, embed=timeF, attn=prob.",
        "- Rejected batch-adjusted/small-batch rows by protocol and run descriptors.",
        "- Required finite mse/mae and readable config where available.",
        "- Arrays are tracked separately through has_arrays for factor/oracle analysis.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build Phase-4 expected/reusable/missing matrices")
    parser.add_argument("--output_dir", default=".aris")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--summary_glob", action="append", default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    expected = expected_matrix()
    summaries = read_all_summaries(args.summary_glob or None)
    reusable, missing = reusable_runs(expected, summaries)
    data_status = validate_local_data(".")

    expected_path = os.path.join(args.output_dir, f"phase4_expected_matrix_{ts}.csv")
    reusable_path = os.path.join(args.output_dir, f"phase4_reusable_runs_{ts}.csv")
    missing_path = os.path.join(args.output_dir, f"phase4_missing_runs_{ts}.csv")
    data_path = os.path.join(args.output_dir, f"phase4_data_integrity_{ts}.csv")
    report_path = os.path.join(args.output_dir, f"phase4_reuse_report_{ts}.md")
    latest_path = os.path.join(args.output_dir, "phase4_latest.json")

    expected.to_csv(expected_path, index=False)
    reusable.to_csv(reusable_path, index=False)
    missing.to_csv(missing_path, index=False)
    data_status.to_csv(data_path, index=False)
    write_reuse_report(report_path, expected, reusable, missing, data_status)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": ts,
                "expected_matrix": expected_path,
                "reusable_runs": reusable_path,
                "missing_runs": missing_path,
                "data_integrity": data_path,
                "reuse_report": report_path,
                "required_seed_runs": int(len(expected)),
                "reusable_seed_runs": int(len(reusable)),
                "missing_seed_runs": int(len(missing)),
                "data_ok": bool(data_status["ok"].all()),
            },
            handle,
            indent=2,
        )
    print(json.dumps(json.load(open(latest_path, encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
