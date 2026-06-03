# Phase-5 Router-Core Context Pack

Date: 2026-05-31

## Purpose

This context pack supersedes the earlier conservative “router diagnostic only” takeover boundary for Phase-5 planning. The new working hypothesis is that `factor-router` remains the core algorithm candidate, while claims stay locked to evidence:

- retrospective Informer evidence can support a router framework;
- cross-dataset generalization is not yet proven;
- oracle-supplement calibration is the planned fallback if zero-shot is weak.

## Evidence Snapshot

Phase-4 completeness:

- Required seed-runs: 972
- Reusable seed-runs: 972
- Missing seed-runs: 0
- `data_ok`: true
- Complete cells: 324/324 dataset-horizon-config cells

Current retrospective router outputs:

- `.aris/phase5_router_core_20260531/phase4_pairwise_factor_router_pairs.csv`
- `.aris/phase5_router_core_20260531/phase4_pairwise_factor_router_predictions.csv`
- `.aris/phase5_router_core_20260531/phase4_pairwise_factor_router_rules.csv`
- `.aris/phase5_router_core_20260531/phase4_pairwise_factor_router_summary.csv`
- `.aris/phase5_router_core_20260531/phase5_router_cell_predictions.csv`
- `.aris/phase5_router_core_20260531/phase5_router_cell_summary.csv`
- `.aris/phase5_router_core_20260531/phase5_router_cell_tables.md`

Best retrospective signal, Informer-only and input-factor-only:

| split | mean gain vs GELU | mean gain vs train static | mean regret vs oracle | oracle capture ratio |
| --- | ---: | ---: | ---: | ---: |
| leave-cell | 0.1237 | 0.00003 | 0.0472 | 0.6409 |
| leave-dataset | 0.1236 | -0.00010 | 0.0473 | 0.6383 |
| leave-horizon | 0.1232 | -0.00048 | 0.0477 | 0.6243 |

Interpretation for first paper version:

- Strong enough to keep router as the algorithmic spine.
- Not strong enough yet to claim zero-shot beats the strongest static baseline.
- The likely paper fork is GO vs CALIBRATE, not “abandon router”.

## Code Changes

Router analysis:

- `lee_ocil/scripts/phase4_pairwise_factor_router.py`
  - pairwise uplift + abstain cell-level outputs;
  - split summaries;
  - oracle capture ratio and regret;
  - action/config hit rate;
  - placement precision;
  - input-only factor groups by default.

Dataset support:

- `lee_ocil/phase4_informer.py`
- `lee_ocil/main_informer.py`
- `lee_ocil/exp/exp_informer.py`

New helper scripts:

- `lee_ocil/scripts/phase5_dataset_preflight.py`
- `lee_ocil/scripts/phase5_build_router_oracle_matrix.py`

## Local Checks Run

Syntax:

```bash
python -m py_compile \
  lee_ocil/scripts/phase4_pairwise_factor_router.py \
  lee_ocil/scripts/phase5_dataset_preflight.py \
  lee_ocil/scripts/phase5_build_router_oracle_matrix.py \
  lee_ocil/phase4_informer.py \
  lee_ocil/main_informer.py \
  lee_ocil/exp/exp_informer.py
```

Phase-4 integrity:

```bash
python - <<'PY'
import json
with open('.aris/phase4_latest.json','r',encoding='utf-8') as f:
    d=json.load(f)
print({k:d.get(k) for k in ['required_seed_runs','reusable_seed_runs','missing_seed_runs','data_ok']})
PY
```

Result:

```text
{'required_seed_runs': 972, 'reusable_seed_runs': 972, 'missing_seed_runs': 0, 'data_ok': True}
```

New dataset preflight:

```bash
python lee_ocil/scripts/phase5_dataset_preflight.py \
  --output_dir .aris/phase5_new_dataset_preflight_20260531 \
  --loader_smoke
```

Result:

- Weather / Exchange / ILI missing locally.
- No loader smoke rows were run because CSVs were absent.

Oracle matrix generation:

```bash
python lee_ocil/scripts/phase5_build_router_oracle_matrix.py \
  --output .aris/phase5_new_dataset_oracle_matrix_20260531.csv
```

Result:

- 54 planned oracle seed-runs.

## Dataset Status

Expected first-wave paths:

| dataset | expected path from `lee_ocil` cwd | target | freq |
| --- | --- | --- | --- |
| Weather | `../Weather/weather.csv` | `OT` | `t` |
| Exchange | `../Exchange/exchange_rate.csv` | `OT` | `d` |
| ILI | `../ILI/national_illness.csv` | `OT` | `w` |

If the cloud machines already contain these files, run preflight there and record outputs before launch. If not, sync datasets first and rerun schema checks.

## Remote Guardrails

Allowed 4090 hosts:

- `10.21.53.82`
- `10.21.53.113`
- `10.21.53.142`
- `10.21.53.162`
- `10.21.53.62` only after `nvidia-smi`/NVML is fixed

Disabled:

- `10.20.12.248`
- `10.20.12.247`

Always rerun fresh preflight before launch. Existing snapshot is not enough for a later launch.

## Next Execution Steps

1. Confirm Weather / Exchange / ILI files on cloud hosts.
2. Run `phase5_dataset_preflight.py --loader_smoke` on cloud in the project venv.
3. Build queue manifests from `.aris/phase5_new_dataset_oracle_matrix_20260531.csv`.
4. Launch only after no conflicting GPU jobs are present.
5. Analyze 1-seed oracle matrix as triage only.
6. If trends align with router predictions, add seeds `2025,2026`.
7. If zero-shot is weak, use oracle-supplement calibration and evaluate held-out split before writing claims.
