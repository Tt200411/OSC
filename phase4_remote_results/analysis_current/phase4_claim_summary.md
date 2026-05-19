# Phase-4 Claim Summary

Generated: 2026-05-19T04:44:08.855843+00:00

## Evidence Status

- Complete dataset-horizon-config cells: 37
- Incomplete dataset-horizon-config cells: 287
- Dataset-horizon cells with a non-GELU best activation: 6

## Supported

Same-protocol activation choice is dataset- and horizon-sensitive where cells have all three seeds. The claim is limited to matched Informer settings and same-environment GELU comparisons.

## Not Supported

A universal oscillatory or bounded-activation win is not supported unless the full matrix shows consistent three-seed gains. Small-batch Lee diagnostics remain excluded from main evidence.

## Dataset-Specific Signals

| dataset | pred_len | config_name | mse_mean | mse_std | seed_count | improvement_vs_gelu_mean | win_rate_vs_gelu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 168 | tanh_all | 0.9115707166666667 | 0.1307529008731196 | 3 | 0.12943737450817336 | 0.6666666666666666 |
| ETTh1 | 336 | softsign_all | 1.2189896666666666 | 0.04196436245689117 | 3 | 0.10351801874705231 | 1.0 |
| ETTh2 | 168 | tanh_all | 3.1845544333333335 | 0.28975137672867624 | 3 | 0.3229979940856428 | 1.0 |
| ETTh2 | 336 | tanh_all | 2.2126836 | 0.20943688340001138 | 3 | 0.4438925186518927 | 1.0 |
| Solar1 | 96 | enc_tanh_dec_gelu | 0.22410256333333334 | 0.0049727243209566185 | 3 | 0.024914397179880195 | 0.6666666666666666 |
| Solar5 | 96 | enc_tanh_dec_gelu | 0.18039039666666667 | 0.00258973115771373 | 3 | 0.07590397980016877 | 1.0 |

## Still Needs Verification

- Any incomplete cell must stay out of the main table until all three seeds exist or are explicitly marked invalid.
- Same-split factor oracles are diagnostic upper bounds; leave-one-seed-out oracles are the only weak generalization evidence.
