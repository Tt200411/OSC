# Phase 1 Remote Informer Activation Experiments

This repository is prepared so local work handles code, scheduling, collection, and analysis. Training should run on remote GPU servers.

## Primary Server

- First server: `10.20.12.248`
- Remote directory: `~/project/osc_informer`
- Do not write SSH passwords to scripts, configs, logs, or result files.

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
