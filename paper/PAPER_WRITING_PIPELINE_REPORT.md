# Paper Writing Pipeline Report

**Input**: `NARRATIVE_REPORT.md`  
**Workflow requested**: `/paper-writing "NARRATIVE_REPORT.md" -- effort: max -- assurance: submission`  
**Actual execution mode**: ARIS Workflow 3 followed manually from installed `SKILL.md` files because `/paper-writing` is not available as a callable tool in this Codex session.  
**Venue**: ICLR draft target  
**Assurance**: submission  
**Submission-ready**: no  
**Date**: 2026-05-16

## Pipeline Summary

| Phase | Status | Output |
|---|---|---|
| 0. Assurance setup | done | `paper/.aris/assurance.txt = submission` |
| 1. Paper plan | done | `PAPER_PLAN.md` |
| 2. Figures/tables | done | `figures/*.pdf`, `figures/TABLE_*.tex`, `figures/latex_includes.tex` |
| 3. LaTeX writing | done | `paper/main.tex`, `paper/sections/*.tex`, `paper/references.bib` |
| 4. Compilation | done | `paper/main.pdf` generated with Tectonic; 10 pages, 190KB |
| 4.5 Proof audit | done | `paper/PROOF_AUDIT.json` verdict `NOT_APPLICABLE` |
| 5.5 Paper claim audit | done | `paper/PAPER_CLAIM_AUDIT.json` verdict `WARN` |
| 5.8 Citation audit | done | `paper/CITATION_AUDIT.json` verdict `WARN` |
| 5.6 Kill-argument audit | done | `paper/KILL_ARGUMENT.json` verdict `FAIL` |
| 6.0 Assurance verifier | fail | `kill-argument` has blocking verdict `FAIL`; all artifacts present and fresh |

## Deliverables Produced

- `NARRATIVE_REPORT.md`
- `PAPER_PLAN.md`
- `figures/gen_phase1_figures.py`
- `figures/fig1_activation_effects.{pdf,png}`
- `figures/fig2_tanh_specificity_heatmap.{pdf,png}`
- `figures/fig3_factor_gradients.{pdf,png}`
- `figures/TABLE_dataset_coverage.tex`
- `figures/TABLE_bounded_vs_gelu.tex`
- `figures/TABLE_tanh_specificity.tex`
- `figures/TABLE_factor_gradients.tex`
- `paper/main.tex`
- `paper/sections/*.tex`
- `paper/references.bib`
- `paper/PROOF_AUDIT.{md,json}`
- `paper/PAPER_CLAIM_AUDIT.{md,json}`
- `paper/CITATION_AUDIT.{md,json}`
- `paper/KILL_ARGUMENT.{md,json}`
- `paper/.aris/audit-verifier-report.json`

## Current Scientific Claims

Supported by current data:

1. Activation-only changes in Informer materially affect forecasting error.
2. Bounded sign-preserving activations are the strongest current explanation for improvements over GELU.
3. Explicit `tanh + a sin(x)` perturbation is not globally reliable against `tanh`.
4. Lee-OC has dataset- and horizon-specific behavior; Lee3 is strong on ETTh2/168 but harmful elsewhere.
5. Input-window factors are useful for explaining conditional perturbation behavior, but factor gradients do not by themselves prove global effectiveness.

Not yet submission-safe:

1. A universal claim that oscillation improves long high-volatility sequences.
2. A universal claim that Lee1 or Lee3 is better than `tanh`.
3. A dynamic activation claim.
4. A statistical-significance claim across all selected datasets.

## DATA_NEEDED

1. Decide and freeze the de-duplication policy for repeated remote rows.
2. Regenerate result summaries and tables from unique seed/config units.
3. Add paired confidence intervals or paired tests for headline gains and win rates.
4. Add at least 3 seeds for ETTh1/336 and ETTh2/336 if the final narrative emphasizes long-horizon behavior.
5. Replace or move server-sliced factor-gradient extremes with deduplicated pooled estimates, or keep them as appendix diagnostics only.
6. Rerun claim, citation, kill-argument, compile, and verifier after the result table is frozen.

## Unresolved Risks

- Current all-server summaries include repeated run records. The paper now labels them as provenance records, but submission still requires a frozen de-duplication policy and regenerated tables.
- Some factor-gradient rows are server-slice specific. They support conditional diagnostics, not universal activation selection.
- The current PDF passes automated compile checks and the first page render was inspected, but a full human visual pass is still outstanding.
- Citation audit ran: all cited entries exist; minor weak-context risks remain for broad background wording.
- Kill-argument audit is blocking: critical unresolved issues are repeated-run de-duplication and missing uncertainty/paired tests.

## Next Steps

1. Freeze a de-duplicated results table.
2. Regenerate figures/tables and update claims from the frozen CSVs.
3. Add paired uncertainty/tests, then rerun the ARIS submission audit chain.

## Compilation Report

- Local dependencies installed: `tectonic 0.16.9` and `poppler 26.04.0`.
- Command: `cd paper && tectonic --keep-logs --keep-intermediates main.tex`.
- Output: `paper/main.pdf`, 10 pages, 191KB.
- Checks passed: no undefined references or citations, no overfull boxes, no `TODO`/`FIXME`/`[VERIFY]`/`DATA_NEEDED` markers in extracted PDF text, and all fonts are embedded with no Type 3 fonts.
- Remaining compile warnings are underfull boxes only.

## Audit Report

- `proof-checker`: `NOT_APPLICABLE`; no theorem/proof environments detected.
- `paper-claim-audit`: `WARN`; numeric claims match CSV evidence after wording fixes, but results remain Phase-1 descriptive signals.
- `citation-audit`: `WARN`; all cited entries are real, with minor weak-context background citation risks.
- `kill-argument`: `FAIL`; unresolved critical issues remain around de-duplication and missing paired uncertainty/tests.
- Verifier command: `bash /Users/tangbao/.codex/Auto-claude-code-research-in-sleep/tools/verify_paper_audits.sh paper/ --assurance submission`.
- Verifier result: fails only because `kill-argument` is a blocking `FAIL`; artifacts are present and fresh.
