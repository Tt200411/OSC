# Phase-4 Solar4 NaN Fix

Generated: 2026-05-21

## Reason

The full-grid reuse scan after `phase4_full_venv` found 54 missing canonical seed-runs. All were Solar4 because the corresponding completed remote rows had non-finite `mse`/`mae`. Job logs and `metrics.npy` showed NaN metrics from the first epoch. Local data inspection found 6 missing values in `Solar/Site_4_130MW.csv`, all in the target column `Power`.

## Repair Policy

Only the six missing target values were filled. Nighttime rows with zero irradiance and zero neighboring power were filled with `0.0`. Daytime rows were filled by local linear interpolation between the immediate previous and next valid `Power` values. No other dataset or column was changed.

## Filled Rows

| csv_line | index | date | previous_power | next_power | filled_power |
| ---: | ---: | --- | ---: | ---: | ---: |
| 23897 | 23895 | 2019-09-06 21:45:00 | 0.0 | 0.0 | 0.0 |
| 45433 | 45431 | 2020-04-18 05:45:00 | 0.0 | 0.0 | 0.0 |
| 48425 | 48423 | 2020-05-19 09:45:00 | 63.374 | 80.307 | 71.8405 |
| 64455 | 64453 | 2020-11-02 09:15:00 | 58.116 | 72.774 | 65.445 |
| 64960 | 64958 | 2020-11-07 15:30:00 | 51.665 | 37.913 | 44.789 |
| 64962 | 64960 | 2020-11-07 16:00:00 | 37.913 | 20.517 | 29.215 |

## Hashes

- Before: `8cda5fc1e9c4368fbe2e611c4dc0195f3aa6831b120bddf9daaff1970e1ea5e9`
- After: `47c851e6e04642770cd94e76e5b6782bcf4203fb08cf4695a8d08fa6bbd70672`

## Verification

- `Solar/Site_4_130MW.csv` rows: `70176`
- Total NaN after repair: `0`
- `Power` NaN after repair: `0`
- Local Phase-4 data integrity after repair: `12/12` ok with `target_nan=0` and `total_nan=0`.

## Rerun Scope

Only the 54 missing canonical Solar4 seed-runs should be relaunched:

- datasets: `Solar4`
- horizons: `24`, `96`
- configs: 9 canonical Phase-4 configs
- seeds: `2024`, `2025`, `2026`
- protocol: `batch_size=32`, `train_epochs=6`
