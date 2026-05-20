# Phase-4 Claim Summary

Generated: 2026-05-21T02:31:00+08:00

## Evidence Status

- Complete dataset-horizon-config cells: 324
- Incomplete dataset-horizon-config cells: 0
- Dataset-horizon cells with a non-GELU best activation: 34
- Required seed-runs: 972
- Reusable/completed seed-runs: 972
- Missing seed-runs: 0

## Supported

Same-protocol activation choice is dataset- and horizon-sensitive under the completed Phase-4 Informer matrix. Bounded and tanh-family activations are strong static choices by mean rank, and non-GELU activations are best in 34/36 dataset-horizon cells. The claim is limited to matched Informer settings and same-environment GELU comparisons.

## Not Supported

A universal oscillatory, bounded, or output-bounded activation win is not supported. Solar3 remains GELU-best at both tested horizons. Factor-bin and oracle analyses do not support a deployable activation router. Small-batch Lee diagnostics remain excluded from main evidence.

## Dataset-Specific Signals

| dataset | pred_len | config_name | mse_mean | mse_std | seed_count | improvement_vs_gelu_mean | win_rate_vs_gelu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 24 | softsign_all | 0.5324173 | 0.02563012450555015 | 3 | 0.05169489362903271 | 0.6666666666666666 |
| ETTh1 | 48 | softsign_all | 0.6459194633333333 | 0.07306488495333471 | 3 | 0.12647239089473447 | 1.0 |
| ETTh1 | 168 | tanh_all | 0.9115707166666667 | 0.1307529008731196 | 3 | 0.12943737450817336 | 0.6666666666666666 |
| ETTh1 | 336 | softsign_all | 1.2189896666666666 | 0.04196436245689117 | 3 | 0.10351801874705231 | 1.0 |
| ETTh1 | 720 | tanh_all_outtanh | 1.3278413666666666 | 0.030069313667314163 | 3 | 0.05971978912674164 | 1.0 |
| ETTh2 | 24 | tanh_sin001_all | 0.5158879133333333 | 0.02298116841491166 | 3 | 0.27759540453576026 | 1.0 |
| ETTh2 | 48 | softsign_all | 1.6064207333333334 | 0.3857662182627496 | 3 | 0.3883877885346281 | 1.0 |
| ETTh2 | 168 | tanh_sin001_all | 2.9625697333333334 | 0.5090410788583406 | 3 | 0.41347375534278247 | 1.0 |
| ETTh2 | 336 | tanh_all_outtanh | 2.053870566666667 | 0.21873389576456412 | 3 | 0.47680769802158496 | 1.0 |
| ETTh2 | 720 | tanh_all_outtanh | 3.2981987333333334 | 0.18607150045854234 | 3 | 0.18603247968351624 | 1.0 |
| ETTm1 | 24 | tanh_all | 0.31447460666666666 | 0.015218849157966372 | 3 | 0.13373821270780709 | 1.0 |
| ETTm1 | 48 | tanh_all | 0.4533515166666667 | 0.02276197124483804 | 3 | 0.10837417451301795 | 1.0 |
| ETTm1 | 96 | tanh_sin001_all | 0.5593670199999999 | 0.05013844201812819 | 3 | 0.0926210452427011 | 1.0 |
| ETTm1 | 288 | tanh_all | 0.8019380233333333 | 0.04381890085819623 | 3 | 0.07192182571291834 | 0.6666666666666666 |
| ETTm1 | 672 | softsign_all | 0.92203278 | 0.02342834254225425 | 3 | 0.07300007678794627 | 0.6666666666666666 |
| ETTm2 | 24 | tanh_all | 0.6846686166666666 | 0.19141435897843362 | 3 | 0.42463772452235443 | 1.0 |
| ETTm2 | 48 | softsign_all | 0.30922799 | 0.013584117038412917 | 3 | 0.215991629475398 | 1.0 |
| ETTm2 | 96 | tanh_sin001_all | 1.3350371 | 0.21439168432784414 | 3 | 0.16423079806949312 | 1.0 |
| ETTm2 | 288 | tanh_all_outtanh | 2.3545683 | 0.35919560612719087 | 3 | 0.5570479221985768 | 1.0 |
| ETTm2 | 672 | tanh_all_outtanh | 2.8509023666666664 | 0.1619360355080469 | 3 | 0.6273818378859305 | 1.0 |
| Solar1 | 24 | swish_all | 0.15043060333333333 | 0.00030593529272272005 | 3 | 0.017026815361270303 | 0.6666666666666666 |
| Solar1 | 96 | softsign_all | 0.22388964666666666 | 0.0009466405841888009 | 3 | 0.025846765548748676 | 1.0 |
| Solar2 | 24 | tanh_all | 0.06606729666666666 | 0.0029103276221300834 | 3 | 0.06429307389526233 | 1.0 |
| Solar2 | 96 | tanh_all | 0.10796687133333333 | 0.004506539567518005 | 3 | 0.10793094864443642 | 1.0 |
| Solar3 | 24 | gelu_all | 0.19754754 | 0.009032388668602549 | 3 | nan | nan |
| Solar3 | 96 | gelu_all | 0.3490070666666667 | 0.007134661292993066 | 3 | nan | nan |
| Solar4 | 24 | tanh_sin001_all | 0.2198203 | 0.0025476834113366503 | 3 | 0.019361038795452173 | 1.0 |
| Solar4 | 96 | enc_tanh_dec_gelu | 0.29044897333333336 | 0.004620217432560203 | 3 | 0.019375793028420896 | 0.6666666666666666 |
| Solar5 | 24 | swish_all | 0.10869422666666667 | 0.0020422638957131294 | 3 | 0.041661470190579496 | 1.0 |
| Solar5 | 96 | enc_tanh_dec_gelu | 0.18039039666666667 | 0.00258973115771373 | 3 | 0.07590397980016877 | 1.0 |
| Solar6 | 24 | tanh_sin001_all | 0.26476890999999997 | 0.0016180409457426813 | 3 | 0.01123132926671868 | 1.0 |
| Solar6 | 96 | tanh_sin001_all | 0.35758402 | 0.009418286723581968 | 3 | 0.02251340784976164 | 1.0 |
| Solar7 | 24 | relu_all | 3.280724533333333 | 1.0349107089792544 | 3 | 0.12042672087487927 | 0.6666666666666666 |
| Solar7 | 96 | enc_tanh_dec_gelu | 4.969628033333334 | 1.3531521857797162 | 3 | 0.38187427230844984 | 1.0 |
| Solar8 | 24 | tanh_sin001_all | 0.21601371 | 0.0025490222482552017 | 3 | 0.0075035929545134925 | 1.0 |
| Solar8 | 96 | softsign_all | 0.30559342 | 0.005197340363868807 | 3 | 0.02601445942756651 | 1.0 |

## Still Needs Verification

- Prospective factor-conditioned routing needs held-out validation beyond same-split diagnostics.
- Exact-protocol Lee variants would need their own complete three-seed matrix before becoming main evidence.
- The result is not an original Informer Table 2 reproduction/surpass claim.
