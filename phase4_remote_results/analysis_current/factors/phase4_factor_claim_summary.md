# Phase-4 Factor Claim Summary

- Summary-level factor-bin coverage: 36 dataset-horizon cells.
- Factor-bin rows: 783.
- Factor-conditioned contrasts with |end-start improvement| >= 0.02: 196.

Summary-level factor bins cover the full completed matrix. Sample-level oracle files in this directory remain diagnostic and are restricted to rows with prediction arrays.

## Static Activation Ranking

| config_name | mean_rank_in_cell | best_cell_count | improvement_vs_gelu_mean | positive_cell_rate |
| --- | --- | --- | --- | --- |
| tanh_sin001_all | 2.7777777777777777 | 8 | 0.11684794465819258 | 0.8333333333333334 |
| tanh_all | 3.1944444444444446 | 7 | 0.11925067580881696 | 0.8055555555555556 |
| softsign_all | 3.4722222222222223 | 8 | 0.11177396747111211 | 0.8055555555555556 |
| enc_tanh_dec_gelu | 4.833333333333333 | 3 | 0.050981801360899456 | 0.6666666666666666 |
| swish_all | 4.888888888888889 | 2 | 0.03649532228179982 | 0.7777777777777778 |

## Strongest Factor-Conditioned Contrasts

| factor | config_name | start_bin | end_bin | improvement_start | improvement_end | improvement_end_minus_start |
| --- | --- | --- | --- | --- | --- | --- |
| jump_intensity | tanh_all_outtanh | low | high | 0.01101028627517115 | -0.8984515570758204 | -0.9094618433509916 |
| turbulence_score | tanh_all_outtanh | low | high | 0.01101028627517115 | -0.854065286983663 | -0.8650755732588342 |
| skewness | tanh_all_outtanh | low | high | 0.023558131263866616 | -0.8195650938187033 | -0.84312322508257 |
| volatility | tanh_all_outtanh | low | high | 0.01101028627517115 | -0.8195650938187033 | -0.8305753800938745 |
| realized_volatility | tanh_all_outtanh | low | high | -0.11706783283201465 | -0.9038455784126732 | -0.7867777455806586 |
| downside_volatility | tanh_all_outtanh | low | high | -0.11706783283201465 | -0.9038455784126732 | -0.7867777455806586 |
| upside_volatility | tanh_all_outtanh | low | high | -0.11706783283201465 | -0.9038455784126732 | -0.7867777455806586 |
| max_abs_change | tanh_all_outtanh | low | high | -0.11706783283201465 | -0.9038455784126732 | -0.7867777455806586 |
| volatility_ratio_late_early | tanh_all_outtanh | low | high | -0.06795287641725979 | -0.854065286983663 | -0.7861124105664032 |
| range | tanh_all_outtanh | low | high | 0.01101028627517115 | -0.7287040397165776 | -0.7397143259917488 |
