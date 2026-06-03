# Kill Argument Report — A Phase-1 Evidence Report on Informer Activations

**Date**: 2026-05-16
**Reviewer model**: GPT-5.5 xhigh, fresh threads
**Verdict**: FAIL (`reason_code: unresolved_critical`)

## Net Assessment

The paper has materially downgraded its title, abstract, and scope into a Phase-1 evidence report. That defuses the strongest overstated-oscillation attack, but the central statistical objection remains: precise empirical gains are reported before frozen de-duplication, effective sample sizes, confidence intervals, or paired tests. As an ICLR empirical claim, the hostile memo is still substantially sustained.

## Attack Memo

> I would reject because the paper’s central empirical evidence is explicitly not in a submission-ready statistical state. The abstract reports precise paired gains and win rates as if they are validated findings, while the paper concedes repeated remote run records and no frozen de-duplication, confidence intervals, or paired tests. The factor-conditioned story relies on server-sliced diagnostics with extreme values, and the paper itself reserves dynamic selection for future work.

## Adjudication Summary

- answered_by_current_text: 1
- partially_answered: 4
- still_unresolved: 2

Critical unresolved points: repeated run records are not independent experimental units; three-seed point estimates lack final inferential support.

## Top Action Items

1. Freeze de-duplication and regenerate all result tables from unique seed/config units.
2. Add paired uncertainty or tests for every headline gain and win rate.
3. Re-scope factor-gradient results as appendix diagnostics unless deduplicated pooled estimates with uncertainty are available.
