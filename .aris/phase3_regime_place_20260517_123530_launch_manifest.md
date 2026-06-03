# Phase 3 Regime Placement Launch Manifest

Created: 2026-05-17 12:38 Asia/Shanghai

Scope: Agent A remote deployment only. No paper files changed. No SSH password recorded.

## Pre-flight

All target hosts accepted password-based SSH via transient expect environment.

| Server | GPU | Initial GPU mem | Disk available | Screen state | Decision |
| --- | --- | ---: | ---: | --- | --- |
| 10.21.53.62 | RTX 4090 | 117 MiB | 42G | none | launched; disk tight but sufficient for requested batch |
| 10.21.53.113 | RTX 4090 | 143 MiB | 397G | none | launched |
| 10.21.53.142 | RTX 4090 | 40 MiB | 1.6T | none | launched |
| 10.21.53.82 | RTX 4090 | 40 MiB | 1.2T | none | launched |

Remote directory: `~/project/osc_informer`

Sync command used per host:

```bash
bash scripts/sync_phase1_remote.sh testsv@<host> ~/project/osc_informer
```

The sync copied `lee_ocil` code plus ETTh1/ETTh2 and Solar station data, excluding remote results/logs/checkpoints and `.venv*`.

## Launched Screens

| Server | Screen | Plan | Dataset filter | Configs | Seeds | Batch | Arrays | Log |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| 10.21.53.62 | `phase3_regime_place_20260517_123530_10_21_53_62` | `ett_core` | `ETTh1` | `relu_all swish_all` | `2024 2025 2026` | 32 | yes | `~/project/osc_informer/lee_ocil/logs/phase3_regime_place_20260517_123530_10_21_53_62.launch.log` |
| 10.21.53.113 | `phase3_regime_place_20260517_123530_10_21_53_113` | `ett_core` | `ETTh2` | `relu_all swish_all` | `2024 2025 2026` | 32 | yes | `~/project/osc_informer/lee_ocil/logs/phase3_regime_place_20260517_123530_10_21_53_113.launch.log` |
| 10.21.53.142 | `phase3_regime_place_20260517_123530_10_21_53_142` | `solar_96` | `Solar1` | `relu_all swish_all` | `2024 2025 2026` | 32 | yes | `~/project/osc_informer/lee_ocil/logs/phase3_regime_place_20260517_123530_10_21_53_142.launch.log` |
| 10.21.53.82 | `phase3_regime_place_20260517_123530_10_21_53_82` | `solar_96` | `Solar5` | `relu_all swish_all` | `2024 2025 2026` | 32 | yes | `~/project/osc_informer/lee_ocil/logs/phase3_regime_place_20260517_123530_10_21_53_82.launch.log` |

## Command Template

```bash
cd ~/project/osc_informer/lee_ocil
CUDA_VISIBLE_DEVICES=0 \
SERVER_ID=<server-id> SERVER_IP=<server-ip> GPU=0 PYTHON=.venv/bin/python \
RUN_ID=<screen> RUN_TAG=<screen> \
PLAN=<plan> DATASET_FILTER=<dataset> CONFIGS="relu_all swish_all" \
SEEDS="2024 2025 2026" BATCH_SIZE=32 SAVE_ARRAYS=1 \
bash scripts/run_phase2_original_protocol_probe.sh 2>&1 | tee logs/<screen>.launch.log
```

## Post-launch Verification

All four screens were detached and still running after launch. GPU memory and utilization increased on each host, and each launcher log showed active epoch progress.
