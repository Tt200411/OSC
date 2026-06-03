# PatchTST Experiment Audit

Date: 2026-05-21

Scope:

- Stage1 run id: `patchtst_stage1_20260521_actfix`
- Stage2 run id: `patchtst_stage2_20260521_actfix`
- Local result roots:
  - `patchtst_remote_results/stage1_actfix_20260521`
  - `patchtst_remote_results/stage2_actfix_20260521`
  - `patchtst_remote_results/analysis_patchtst_transfer_actfix`

## Verdict

PASS with claim-scope qualification.

The PatchTST results are complete and internally consistent for the tested matrix. The supported claim is partial, architecture-dependent transfer; the results do not support a broad claim that the Informer activation ranking or output-tanh placement transfers to PatchTST.

## Integrity Checks

### Completion

- Stage1 queue states: `144/144` completed across four RTX 4090 hosts.
- Stage2 queue states: `54/54` completed across four RTX 4090 hosts.
- Stage1 finite rows: `144/144`.
- Stage2 finite rows: `54/54`.

### Seed Coverage

- Stage1 complete config cells: `48/48`, each with seeds `2024, 2025, 2026`.
- Stage2 complete config cells: `18/18`, each with seeds `2024, 2025, 2026`.

### Artifact Coverage

Stage2 pulled artifacts:

- `config.json`: `54/54`
- `metrics.npy`: `54/54`
- `pred.npy`: `54/54`
- `true.npy`: `54/54`

Stage1 was previously audited as complete with prediction arrays present.

### Duplicate / Protocol Audit

Combined transfer raw table:

- Rows: `198`
- Stage1 rows: `144`
- Stage2 rows: `54`
- Duplicate protocol rows after canonical dedupe: `0`
- Finite MSE/MAE rows: `198/198`
- Source-path priority check: all Stage1 rows resolve to Stage1 source paths and all Stage2 rows resolve to Stage2 source paths.

### Secret Scan

Targeted scan for the literal SSH password string returned no matches in scripts, manifests, paper notes, ARIS PatchTST manifests, or analysis outputs. Generic `password` matches exist only in expect helper code that reads `PATCHTST_SSH_PASSWORD` from the transient environment.

## Claim Impact

Supported:

- PatchTST has measurable activation sensitivity in selected dataset-horizon cells.
- Non-GELU PatchTST wins occur in `5/12` tested Stage1 cells.
- Output-tanh placement does not transfer to PatchTST; it loses `54/54` paired seed-run comparisons against the corresponding linear-head configuration.

Not supported:

- Informer activation ranking transfers to PatchTST.
- Bounded/tanh-family activations are generally better than GELU in PatchTST.
- Output tanh is a robust cross-architecture placement rule.

Recommended paper claim:

> PatchTST transfer is partial and architecture-dependent: activation choice remains relevant, but the best activation family and placement cannot be assumed to transfer from Informer.
