# Phase-4 Result-To-Claim Verdict

**Date**: 2026-05-21  
**Integrity audit**: pass, pending external reviewer  
**Verdict**: partial support for the broad idea; strong support for the narrowed claim

## Intended Claim

Activation-only changes can improve Informer forecasting under fixed architecture and training protocol, and the useful activation depends on dataset, horizon, and placement.

## What The Results Support

- The full matrix is complete: 972/972 seed-runs, 324/324 three-seed config cells.
- A non-GELU activation is best in 34/36 dataset-horizon cells.
- All 20 ETT/ETTm cells prefer non-GELU activations.
- Solar is mixed but still mostly non-GELU-best: 14/16 cells.
- The best average static ranks are `tanh_sin001_all` (2.78), `tanh_all` (3.19), and `softsign_all` (3.47).
- Placement matters: output `tanh` wins five cells but has poor mean rank, encoder-side `tanh` wins three cells, and decoder-side `tanh` wins none.

## What The Results Do Not Support

- No universal activation is supported. Best-cell wins are spread across `tanh_sin001_all`, `softsign_all`, `tanh_all`, `tanh_all_outtanh`, `enc_tanh_dec_gelu`, `swish_all`, `gelu_all`, and `relu_all`.
- No universal bounded or oscillatory win is supported. Solar3 remains GELU-best at both horizons.
- Factor analysis does not support a deployed activation router. Among 84 LOSO oracle rows, only seven are positive, all on Solar1/96.
- The work does not claim to reproduce and surpass original Informer Table 2.

## Suggested Claim Revision

Use the following claim boundary:

> Under a matched Informer official-script protocol with complete three-seed coverage, activation choice is a dataset- and horizon-sensitive design variable. Bounded and tanh-family activations are strong static choices on average, but the full grid rejects a universal activation or deployable factor-router claim.

## Next Experiments Needed

- Targeted placement ablations on the high-gain output-`tanh` ETT/ETTm cells.
- Exact-protocol follow-up for any Lee-style variants before they are promoted beyond diagnostics.
- Prospective factor-conditioned routing only after rules are learned on one split and validated on held-out datasets or future windows.

## Confidence

High for the narrowed same-protocol claim. Medium for mechanism interpretation. Low for any factor-router generalization.
