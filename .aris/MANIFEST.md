# ARIS Workflow 3 Manifest

**Date**: 2026-05-15  
**Requested workflow**: `/paper-writing "NARRATIVE_REPORT.md" -- effort: max -- assurance: submission`  
**Assurance**: submission  
**Submission-ready**: no

## Inputs

- `phase1_remote_results/all_servers_current/phase1_activation_summary.csv`
- `phase1_remote_results/all_servers_current/phase1_detailed_results.csv`
- `phase1_remote_results/all_servers_current_tanh_baseline/phase1_activation_summary.csv`
- `phase1_remote_results/*/analysis_current_factors*/phase1_factor_*`
- `PHASE1_REMOTE.md`

## Generated Evidence Tables

- `.aris/dataset_coverage_current.csv`
- `.aris/claim_support_activation_summary_min3seeds.csv`
- `.aris/gelu_centered_activation_summary_current.csv`
- `.aris/tanh_centered_activation_summary_current.csv`
- `.aris/all_servers_factor_contrast_default.csv`
- `.aris/all_servers_factor_contrast_lee_tanh.csv`
- `.aris/factor_bin_high_low_default.csv`
- `.aris/factor_bin_high_low_lee_tanh.csv`
- `.aris/hypothesis_current_default/phase1_hypothesis_evidence.{csv,md}`
- `.aris/hypothesis_current_tanh_baseline/phase1_hypothesis_evidence.{csv,md}`
- `.aris/layer_activation_probe_plan_20260516.md`

## Paper Workflow Outputs

- `NARRATIVE_REPORT.md`
- `PAPER_PLAN.md`
- `figures/gen_phase1_figures.py`
- `figures/paper_plot_style.py`
- `figures/fig1_activation_effects.{pdf,png}`
- `figures/fig2_tanh_specificity_heatmap.{pdf,png}`
- `figures/fig3_factor_gradients.{pdf,png}`
- `figures/TABLE_dataset_coverage.tex`
- `figures/TABLE_bounded_vs_gelu.tex`
- `figures/TABLE_tanh_specificity.tex`
- `figures/TABLE_factor_gradients.tex`
- `figures/latex_includes.tex`
- `paper/main.tex`
- `paper/sections/0_abstract.tex`
- `paper/sections/1_introduction.tex`
- `paper/sections/2_related_work.tex`
- `paper/sections/3_method.tex`
- `paper/sections/4_experiments.tex`
- `paper/sections/5_conclusion.tex`
- `paper/sections/A_appendix.tex`
- `paper/references.bib`
- `paper/PAPER_WRITING_PIPELINE_REPORT.md`
- `paper/.aris/assurance.txt`
- `paper/.aris/audit-verifier-report.json`

## Blockers

- Local LaTeX toolchain is missing (`latexmk`, `pdflatex`, `pdfinfo`).
- Submission audits are missing, so ARIS verifier fails under `--assurance submission`.
