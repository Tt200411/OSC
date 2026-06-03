# PatchTST Result-to-Claim Gate

Date: 2026-05-21

## Intended Claim

Informer activation findings transfer to a second forecasting architecture, PatchTST, and can validate activation regime/placement generality.

## Verdict

`claim_supported = partial`

Confidence: high for the tested PatchTST matrix; medium for broader architecture-general statements.

## What The Results Support

1. PatchTST is activation-sensitive in selected dataset-horizon cells.
2. Non-GELU activations are best in `5/12` Stage1 PatchTST cells:
   - ETTh2/720: `softsign_ffn_linear`
   - Solar2/24: `tanh_sin001_ffn_linear`
   - Solar2/96: `softsign_ffn_linear`
   - Solar5/24: `tanh_sin001_ffn_linear`
   - Solar5/96: `softsign_ffn_linear`
3. The effect is dataset-specific:
   - Solar2 and Solar5 show bounded/tanh-family gains.
   - Solar3 remains GELU-best at both horizons.
   - ETTm2 remains GELU-best at all tested horizons.
4. Output-tanh placement is not supported in PatchTST:
   - `0/54` paired seed-run wins versus matched linear-head configs.
   - Mean degradation is approximately `5.3%--5.4%`.

## What The Results Do Not Support

1. They do not support a broad claim that the Informer activation ranking transfers to PatchTST.
2. They do not support treating bounded/tanh-family activations as generally superior PatchTST defaults.
3. They do not support output tanh as an architecture-independent placement rule.
4. They do not validate a factor-based placement or regime selector.

## Claim Revision

Use:

> PatchTST transfer is partial and architecture-dependent. Activation choice remains relevant outside Informer, but the winning activation family and placement differ by architecture, dataset, and horizon.

Avoid:

> The proposed activation family transfers to PatchTST.

Avoid:

> Output tanh is a robust placement strategy.

## Missing Evidence

1. Prospective factor-router validation remains open.
2. A broader PatchTST grid could test whether the Solar2/Solar5 signal holds on more Solar sites, but this is no longer the highest-value next experiment.
3. If the paper wants a cross-architecture positive claim, add one more architecture or expand PatchTST beyond the current 12 cells. If the paper accepts architecture-dependence, the current PatchTST evidence is enough as a stress test.

## Route

Continue to factor-router validation rather than expanding output-tanh in PatchTST.

The immediate paper story should be:

- Informer full-grid result is strong.
- PatchTST transfer is partial and prevents overclaiming.
- Factor bins are still diagnostic, not yet deployable.
