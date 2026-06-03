# Phase-5 Dataset Sources

Generated: 2026-06-01

## Source

The Phase-5 cross-dataset expansion uses the Autoformer-format forecasting CSVs mirrored from the Hugging Face dataset `AutonLab/Timeseries-PILE`:

- Weather: `https://hf-mirror.com/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/weather.csv`
- Exchange: `https://hf-mirror.com/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/exchange_rate.csv`
- ILI: `https://hf-mirror.com/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/national_illness.csv`

Canonical dataset page: `https://huggingface.co/datasets/AutonLab/Timeseries-PILE`

## Local Files

| dataset | local path | rows | columns | target | missing values | size bytes | sha256 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| Weather | `Weather/weather.csv` | 52696 | 22 | `OT` | 0 | 7182728 | `9b2b19b13e342d9c9b9c51620d42b4be232df7fba30e7f5c4968b7c79ab138ad` |
| Exchange | `Exchange/exchange_rate.csv` | 7588 | 9 | `OT` | 0 | 630212 | `d55e7aa2641009814a18ba3279431b13f6d413b0eab195b9ff21988d8cf94e97` |
| ILI | `ILI/national_illness.csv` | 966 | 8 | `OT` | 0 | 66653 | `8594cecb98c6f2f9dc014eca11c4d5d1e38ad5ce6ed0592d4abe65b146fce395` |

## Validation Boundary

- Local validation is limited to CSV schema, missingness, inferred frequency, feature-factor generation, and dataloader shape smoke checks.
- Training is remote-only on allowed 4090 hosts.
- New dataset oracle results are initially seed-2024 triage evidence. Multi-seed claims require seed 2025/2026 completion before paper wording can treat them as stable.
