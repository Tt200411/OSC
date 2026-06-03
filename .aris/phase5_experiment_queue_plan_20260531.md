# Phase-5 Experiment Queue Plan

Date: 2026-05-31

## Goal

Confirm the single-seed iTransformer placement/regime signal with the smallest useful multi-seed matrix, while preserving the current paper boundary. This plan does not launch jobs; it defines the next queue only.

## Claim Tested

Whether iTransformer has selected dataset-horizon cells where output placement or tanh-family FFN regimes beat the default `gelu_ffn_linear` under repeated seeds.

This is a confirmation experiment, not a broad cross-architecture claim.

## Run Policy

- Run only on cloud 4090 hosts after a fresh preflight.
- Do not use local GPU training.
- Do not use `10.20.12.248` or `10.20.12.247`.
- Do not schedule `10.21.53.62` until the NVML mismatch is fixed.
- Treat `10.21.53.82` old queue manager processes as a stale-process risk; verify or clean intentionally before launching there.
- Use transient SSH credentials only; do not write passwords into files, logs, scripts, or manifests.

## Stage ITR-P1: iTransformer Placement Confirmation

### Cells

Use the same positive-signal cells from the 2026-05-22 placement probe:

| dataset | pred_len | reason |
| --- | ---: | --- |
| ETTh2 | 96 | output tanh improved all six activations in the single-seed probe; best was `gelu_ffn_outtanh`. |
| ETTm2 | 96 | output tanh hurt all six activations; best was `tanh_sin001_ffn_linear`. |
| ETTm2 | 720 | output tanh improved all six activations; best was `tanh_ffn_outtanh`. |

### Configs

Repeat the full placement probe config set so paired output-placement conclusions remain valid:

- `gelu_ffn_linear`
- `relu_ffn_linear`
- `swish_ffn_linear`
- `tanh_ffn_linear`
- `softsign_ffn_linear`
- `tanh_sin001_ffn_linear`
- `gelu_ffn_outtanh`
- `relu_ffn_outtanh`
- `swish_ffn_outtanh`
- `tanh_ffn_outtanh`
- `softsign_ffn_outtanh`
- `tanh_sin001_ffn_outtanh`

### Seeds

- Existing seed: 2024
- New confirmation seeds: 2025, 2026

New jobs required: `3 cells x 12 configs x 2 new seeds = 72`.

### Preferred Host Allocation

Use four healthy 4090 hosts if fresh preflight remains clean:

| host | planned jobs | notes |
| --- | ---: | --- |
| `10.21.53.82` | 18 | Only after stale Phase-4 queue managers are verified harmless or intentionally handled. |
| `10.21.53.113` | 18 | Recheck small non-project GPU app before launch. |
| `10.21.53.142` | 18 | GPU looked free; CPU load by another user should not block GPU training but must be rechecked. |
| `10.21.53.162` | 18 | GPU looked free; recheck before launch. |

Fallback: if one host is unavailable, redistribute evenly across the remaining healthy hosts. Do not use `10.21.53.62` unless `nvidia-smi` is fixed.

## Success Criteria

Stage ITR-P1 is useful if:

- all 72 new rows complete with finite MSE/MAE;
- each of the three cells has 3-seed coverage for all 12 configs;
- paired output-tanh vs linear conclusions are consistent in at least two of three cells;
- best config beats `gelu_ffn_linear` by mean MSE in at least two of three cells.

If the positive signal vanishes, keep iTransformer as a weak/negative transfer boundary.

## Post-Run Required Analysis

After collection:

1. Aggregate by dataset, horizon, config, and seed.
2. Compute 3-seed mean/std MSE and MAE.
3. Compute paired output-tanh gains for each activation base.
4. Compare each config against `gelu_ffn_linear`.
5. Write a result-to-claim note before editing paper claims.

## Paper Use

Allowed if Stage ITR-P1 confirms:

- "iTransformer shows selected placement-sensitive cells, but the evidence is targeted and architecture-dependent."

Not allowed without broader balanced evidence:

- "Informer activation rankings transfer to iTransformer."
- "Output tanh is generally useful."
- "A deployable cross-architecture router is validated."

## Deferred Work

Do not run these until Stage ITR-P1 has been analyzed:

- broader iTransformer cells such as ETTh2/336, ETTh2/720, ETTm2/336, ETTh1, or ETTm1;
- balanced cross-architecture router retraining;
- PatchTST output-tanh expansion.
