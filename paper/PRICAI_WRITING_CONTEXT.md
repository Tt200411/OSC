# PRICAI Writing Context

Date: 2026-05-31

## Venue Constraints

Source: https://2026.pricai.org/submission

- Submission deadline: June 13, 2026.
- Acceptance notification: August 8, 2026.
- Camera-ready deadline: August 22, 2026.
- Format: Springer Lecture Notes in Artificial Intelligence (LNAI), using the Springer LNCS/LNAI proceedings template.
- Submission length: at most 16 pages including references.
- Review mode: double anonymous. The submission must not include author names, affiliations, acknowledgments, or identifying text.
- System: EasyChair.
- AI-tool policy: LLMs cannot be listed as authors. Generative LLM use should be documented in the manuscript unless it is only copy editing.

Springer template source: https://www.springer.com/us/computer-science/lncs/conference-proceedings-guidelines

## Local Paper Skills Read

- `paper-writing`: full pipeline: plan -> figures -> writing -> compile -> improvement/audits. Defaults are ICLR, so PRICAI requires overriding the venue/page assumptions.
- `paper-plan`: builds claims-evidence matrix and paper outline; default page accounting is not PRICAI-compatible because PRICAI counts references.
- `paper-figure`: generates data plots and LaTeX tables from CSV/JSON result assets.
- `paper-write`: writes LaTeX sections; default templates are ICLR/NeurIPS/ICML/etc., not LNCS/LNAI.
- `paper-compile`: local expected compiler is `latexmk/pdflatex`, but this machine does not have those installed.
- `overleaf-sync`: local `paper/` remains the ARIS working copy; an Overleaf bridge would be a sibling `paper-overleaf/` clone. No bridge is currently configured in this repository.

## Local Compile Logic

Available:

- `tectonic 0.16.9`

Unavailable:

- `latexmk`
- `pdflatex`
- `bibtex`

Use:

```bash
cd paper
tectonic --keep-logs --keep-intermediates main.tex
```

Historical notes in `session5.md` and `paper/PAPER_WRITING_PIPELINE_REPORT.md` confirm that Tectonic is the working local compile path.

## Core Story

Primary story file: `paper/投稿故事.md`

Current safe one-sentence claim:

> Under a matched Informer official-script-style protocol with complete three-seed coverage, activation choice is a dataset- and horizon-sensitive design variable. Bounded and tanh-family activations are strong static choices on average, but the full grid rejects a universal activation or deployable factor-router claim.

## Main Evidence Assets

Informer Phase-4 full grid:

- `phase4_remote_results/analysis_current/phase4_protocol_raw_dedup.csv`
- `phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv`
- `phase4_remote_results/analysis_current/phase4_completion_matrix.csv`
- `phase4_remote_results/analysis_current/phase4_paired_vs_gelu.csv`
- `phase4_remote_results/analysis_current/phase4_claim_summary.md`

Factor diagnostics:

- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_contrast_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_loso_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_claim_summary.md`

Architecture-transfer stress tests:

- `paper/patchtst_transfer_results.md`
- `paper/itransformer_transfer_results.md`
- `patchtst_remote_results/analysis_patchtst_stage1_actfix/patchtst_claim_summary.md`
- `patchtst_remote_results/analysis_patchtst_stage2_actfix/patchtst_claim_summary.md`
- `itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_claim_summary.md`
- `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_claim_summary.md`

Router context:

- `phase4_remote_results/analysis_current/factors_router/phase4_factor_router_claim_summary.md`
- `phase4_remote_results/analysis_current/factors_router/phase4_cross_arch_router_context.md`

## Claim Boundaries

Safe to claim now:

- Informer activation-only changes matter under a matched protocol.
- The Phase-4 Informer grid is complete: 36 dataset-horizon cells, nine activation configurations, three seeds per configuration, 972 seed-runs.
- Non-GELU activations are best in 34/36 Informer dataset-horizon cells.
- `tanh_sin001_all`, `tanh_all`, and `softsign_all` are the strongest static Informer choices by mean rank and positive-cell coverage.
- Placement matters: output tanh and encoder-side tanh have specific positive cells but are not universal.
- PatchTST and iTransformer show architecture-dependent transfer, not universal transfer.
- Factor bins and router probes are diagnostic evidence, not a deployable zero-shot router.

Do not claim yet:

- A universal oscillatory activation rule.
- A high-volatility -> tanh/oscillatory selection rule.
- A deployable zero-shot activation router.
- Informer ranking transfers directly to PatchTST or iTransformer.
- Output tanh is robust across architectures.
- The paper surpasses the original Informer Table 2.

## Remaining Warnings

- The strongest paper can be written before all future experiments finish, but router claims should stay diagnostic until prospective validation is complete.
- Any new numerical claim added after this context must be checked against the CSV assets above.
- After final result freezing, rerun paper-claim audit, citation audit, and compile checks.
