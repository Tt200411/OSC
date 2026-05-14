# Phase 1 Remote Informer Activation Experiments

This repository is prepared so local work handles code, scheduling, collection, and analysis. Training should run on remote GPU servers.

## Primary Server

- First server: `10.20.12.248`
- Remote directory: `~/project/osc_informer`
- Do not write SSH passwords to scripts, configs, logs, or result files.

## Current Mechanism Notes

- Keep the current conclusion: Lee-OC type1 is not the only plausible mechanism. On ETTh probes, cheap bounded sign-preserving activations such as `tanh` and `softsign` explain much of the gain; Lee-OC type1 should be treated as one candidate in that family, not as an independently dominant default.
- Factor explanations should not rely on "every bin beats GELU" alone. Use high-minus-low differences, spread, slope/Spearman, and aggregate bin MSE ratios to test whether the gain is stronger under specific data states.
- For Solar Site 1/5 seed `2024` (`phase1_solar_sites_20260514_234004`), bounded sign activations improved all-feature MSE across `pred_len=24` and `pred_len=96`. Power-only aggregate analysis is more nuanced: `pred_len=96` shows clearer high-volatility/high-turbulence gains, while `pred_len=24` can be flatter or stronger in low-volatility bins.
- For Solar, prefer aggregate bin metrics such as `aggregate_relative_target_mse_change` because per-sample relative ratios can be distorted when nighttime baseline target errors are near zero.
- The next specificity test is tanh-centered: compare `tanh_sin` against `tanh`, and compare Lee-OC type1 against both GELU and `tanh` using `--baseline_override lee=tanh` in analysis.

## Sync Core Files

Run from the local repository root:

```bash
scripts/sync_phase1_remote.sh USER@10.20.12.248
```

The sync sends only:

- `lee_ocil/` source and scripts, excluding checkpoints/results/logs and ETTm CSVs
- `lee_ocil/ETT-small/ETTh1.csv`
- `lee_ocil/ETT-small/ETTh2.csv`
- `Solar/PV_Solar_Station_1.csv`
- `Solar/Site_1_50MW.csv`
- `Solar/Site_5_110MW.csv`

## Remote Smoke Checks

Run on the server:

```bash
cd ~/project/osc_informer/lee_ocil
bash scripts/smoke_remote_env.sh
```

This checks Python, pip, Torch/CUDA, `nvidia-smi`, ETTh1 dataloader, and Solar dataloader.

## Minimal Matrix

Run first on `10.20.12.248`:

```bash
cd ~/project/osc_informer/lee_ocil
SERVER_ID=4090-248 SERVER_IP=10.20.12.248 SEEDS="2024" bash scripts/run_phase1_minimal.sh
```

This runs:

- ETTh1, `pred_len=24`, GELU
- ETTh1, `pred_len=24`, GELU + `0.01*sin(x)`
- ETTh1, `pred_len=24`, Lee-OC type1

## Full Matrix

After the minimal loop produces metrics/config/summary rows:

```bash
cd ~/project/osc_informer/lee_ocil
SERVER_ID=4090-248 SERVER_IP=10.20.12.248 SEEDS="2024 2025 2026" bash scripts/run_phase1_matrix.sh
```

For 5 seeds:

```bash
SEEDS="2024 2025 2026 2027 2028" bash scripts/run_phase1_matrix.sh
```

When `10.20.12.248` is in long training, sync the same directory to other 4090 servers and run the matrix with unique `SERVER_ID` and `SERVER_IP`.

## Tanh Specificity Probe

Use this after `tanh` and `softsign` have established the bounded-sign baseline:

```bash
cd ~/project/osc_informer/lee_ocil
SERVER_ID=4090-248 SERVER_IP=10.20.12.248 \
  DATASETS="ETTh1 ETTh2" PRED_LENS="24 168" SEEDS="2024 2025 2026" \
  TANH_SIN_AMPLITUDES="0.01 0.05 0.1" LEE_TYPES="1" \
  bash scripts/run_phase1_tanh_specificity_probe.sh
```

This runs:

- `tanh`
- `tanh + a*sin(x)`, with configured amplitudes
- Lee-OC type1

For Solar, first run `tanh_sin` without Lee-OC to avoid Lee's high runtime:

```bash
SERVER_ID=4090-248 SERVER_IP=10.20.12.248 \
  DATASETS="Solar1 Solar5" PRED_LENS="96" SEEDS="2024 2025 2026" \
  INCLUDE_LEE=0 TANH_SIN_AMPLITUDES="0.01 0.05 0.1" \
  bash scripts/run_phase1_tanh_specificity_probe.sh
```

## Fetch Results

Run locally:

```bash
scripts/fetch_phase1_results.sh USER@10.20.12.248
```

This fetches remote `results/` and `logs/` into `phase1_remote_results/...`.

## Local Analysis

Generate factor tables:

```bash
cd lee_ocil
python scripts/generate_factors.py --output_dir factors
```

Summarize collected experiment tables:

```bash
cd lee_ocil
python scripts/summarize_phase1_results.py ../phase1_remote_results --output_dir analysis
```

Outputs:

- `analysis/phase1_detailed_results.csv`
- `analysis/phase1_activation_summary.csv`

Link predictions to factor bins:

```bash
python lee_ocil/scripts/analyze_factor_effects.py \
  phase1_remote_results/10.20.12.248/results/summary.csv \
  --results_dir phase1_remote_results/10.20.12.248/results \
  --factors_dir lee_ocil/factors \
  --lee_root lee_ocil \
  --output_dir phase1_remote_results/10.20.12.248/analysis_factors
```

Key outputs:

- `phase1_factor_sample_results.csv`
- `phase1_factor_bin_summary.csv`
- `phase1_factor_contrast_summary.csv`

For Lee-OC specificity over the `tanh` baseline, add:

```bash
--baseline_override lee=tanh
```
