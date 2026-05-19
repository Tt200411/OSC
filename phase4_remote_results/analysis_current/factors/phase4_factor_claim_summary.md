# Phase-4 Factor Claim Summary

- Array-ready seed-runs: 60
- Same-split oracle rows: 84
- LOSO oracle rows: 84

Same-split oracle results are diagnostic upper bounds. Treat only LOSO gains as weak generalization evidence.

## Strongest LOSO Signals

| dataset | pred_len | factor | loso_oracle_mse | eval_seed_count | best_static_signature | best_static_placement | best_static_mse | loso_gain_vs_best_static |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Solar1 | 96 | max_abs_change | 0.1740786491939535 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0195954499147477 |
| Solar1 | 96 | volatility | 0.1750451658247246 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0141520637962967 |
| Solar1 | 96 | volatility_shock | 0.1753172454663362 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0126197212611456 |
| Solar1 | 96 | trend_consistency | 0.1767630426784817 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0044770445351285 |
| Solar1 | 96 | range | 0.1772750232621719 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0015935887171442 |
| Solar1 | 96 | max_drawup | 0.1772750232621719 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0015935887171442 |
| Solar1 | 96 | max_drawdown | 0.1772750232621719 | 3 | act=gelu\|enc=tanh\|dec=gelu\|out=linear\|a=0\|lee=1 | encoder-only | 0.1775579776519971 | 0.0015935887171442 |
| ETTh2 | 168 | reversal_rate | 2.29435691062748 | 3 | act=swish\|enc=swish\|dec=swish\|out=linear\|a=0\|lee=1 | full-hidden | 2.29435691062748 | 0.0 |
| ETTh1 | 336 | volatility_shock | 0.5995427262456487 | 3 | act=swish\|enc=swish\|dec=swish\|out=linear\|a=0\|lee=1 | full-hidden | 0.5995427262456487 | 0.0 |
| ETTh2 | 168 | jump_intensity | 2.29435691062748 | 3 | act=swish\|enc=swish\|dec=swish\|out=linear\|a=0\|lee=1 | full-hidden | 2.29435691062748 | 0.0 |
