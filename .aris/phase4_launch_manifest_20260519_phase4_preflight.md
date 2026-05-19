# Phase-4 Launch Manifests

Generated: 2026-05-19

## Scope

Launch only missing seed-runs from `.aris/phase4_missing_runs_20260519_phase4_preflight.csv`.

- Missing seed-runs: 856.
- Smoke queue: 4 seed-runs.
- Post-smoke main queue: 852 seed-runs.
- Disabled hosts: `10.20.12.248`, `10.20.12.247`.
- Candidate hosts: `10.21.53.62`, `10.21.53.82`, `10.21.53.113`, `10.21.53.142`, `10.21.53.162`.

## Smoke Queue

Run first with arrays enabled, then pull summaries and regenerate the missing matrix before launching main.

| host | jobs | manifest |
| --- | ---: | --- |
| 10.21.53.82 | 1 | `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_82.json` |
| 10.21.53.113 | 1 | `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_113.json` |
| 10.21.53.142 | 1 | `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_142.json` |
| 10.21.53.162 | 1 | `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_162.json` |

Smoke rows:

- `ETTm1`, horizon 96, `gelu_all`, seed 2024.
- `ETTm2`, horizon 96, `gelu_all`, seed 2024.
- `Solar2`, horizon 96, `gelu_all`, seed 2024.
- `Solar8`, horizon 96, `gelu_all`, seed 2024.

## Main Queue

Launch only after smoke passes and the missing matrix has been regenerated.

| host | jobs | manifest |
| --- | ---: | --- |
| 10.21.53.62 | 216 | `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_62.json` |
| 10.21.53.82 | 227 | `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_82.json` |
| 10.21.53.113 | 146 | `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_113.json` |
| 10.21.53.142 | 108 | `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_142.json` |
| 10.21.53.162 | 155 | `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_162.json` |

## Resume Commands

Use a transient environment variable only. Do not write the SSH password to any file.

```bash
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh inventory
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh sync
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh launch .aris/phase4_queue_20260519_phase4_preflight_smoke

# After smoke completion and result pull/reuse scan:
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh launch .aris/phase4_queue_20260519_phase4_preflight
```

