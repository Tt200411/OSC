# Phase-5 New Dataset Preflight

## Schema

| dataset | path | exists | freq | horizons | rows | columns | numeric_columns | has_date | has_target | target | target_nan | total_nan | median_delta | mode_delta | mode_delta_rate | schema_ok | schema_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | True | t | 96,192,336 | 52696 | 22 | 21 | True | True | OT | 0 | 0 | 0 days 00:10:00 | 0 days 00:10:00 | 0.9999620457348894 | True |  |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | True | d | 96,192,336 | 7588 | 9 | 8 | True | True | OT | 0 | 0 | 1 days 00:00:00 | 1 days 00:00:00 | 1.0 | True |  |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | True | w | 24,36,48 | 966 | 8 | 7 | True | True | OT | 0 | 0 | 7 days 00:00:00 | 7 days 00:00:00 | 1.0 | True |  |

## Loader Smoke

| dataset | path | pred_len | flag | loader_ok | length | seq_x_shape | seq_y_shape | seq_x_mark_shape | seq_y_mark_shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 96 | train | True | 36696 | 96x21 | 144x21 | 96x5 | 144x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 96 | val | True | 5175 | 96x21 | 144x21 | 96x5 | 144x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 96 | test | True | 10444 | 96x21 | 144x21 | 96x5 | 144x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 192 | train | True | 36600 | 96x21 | 240x21 | 96x5 | 240x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 192 | val | True | 5079 | 96x21 | 240x21 | 96x5 | 240x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 192 | test | True | 10348 | 96x21 | 240x21 | 96x5 | 240x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 336 | train | True | 36456 | 96x21 | 384x21 | 96x5 | 384x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 336 | val | True | 4935 | 96x21 | 384x21 | 96x5 | 384x5 |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | 336 | test | True | 10204 | 96x21 | 384x21 | 96x5 | 384x5 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 96 | train | True | 5120 | 96x8 | 144x8 | 96x3 | 144x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 96 | val | True | 665 | 96x8 | 144x8 | 96x3 | 144x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 96 | test | True | 1422 | 96x8 | 144x8 | 96x3 | 144x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 192 | train | True | 5024 | 96x8 | 240x8 | 96x3 | 240x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 192 | val | True | 569 | 96x8 | 240x8 | 96x3 | 240x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 192 | test | True | 1326 | 96x8 | 240x8 | 96x3 | 240x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 336 | train | True | 4880 | 96x8 | 384x8 | 96x3 | 384x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 336 | val | True | 425 | 96x8 | 384x8 | 96x3 | 384x3 |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | 336 | test | True | 1182 | 96x8 | 384x8 | 96x3 | 384x3 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 24 | train | True | 617 | 36x7 | 42x7 | 36x2 | 42x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 24 | val | True | 74 | 36x7 | 42x7 | 36x2 | 42x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 24 | test | True | 170 | 36x7 | 42x7 | 36x2 | 42x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 36 | train | True | 605 | 36x7 | 54x7 | 36x2 | 54x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 36 | val | True | 62 | 36x7 | 54x7 | 36x2 | 54x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 36 | test | True | 158 | 36x7 | 54x7 | 36x2 | 54x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 48 | train | True | 593 | 36x7 | 66x7 | 36x2 | 66x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 48 | val | True | 50 | 36x7 | 66x7 | 36x2 | 66x2 |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | 48 | test | True | 146 | 36x7 | 66x7 | 36x2 | 66x2 |

## Path Candidates

| dataset | resolved_path | checked_paths |
| --- | --- | --- |
| Weather | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv | /Users/tangbao/project/Weather/weather.csv ; /Users/tangbao/project/dataset/weather/weather.csv ; /Users/tangbao/project/weather/weather.csv ; /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/dataset/weather/weather.csv ; /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Weather/weather.csv |
| Exchange | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv | /Users/tangbao/project/Exchange/exchange_rate.csv ; /Users/tangbao/project/dataset/exchange_rate/exchange_rate.csv ; /Users/tangbao/project/exchange_rate/exchange_rate.csv ; /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/dataset/exchange_rate/exchange_rate.csv ; /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/Exchange/exchange_rate.csv |
| ILI | /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv | /Users/tangbao/project/ILI/national_illness.csv ; /Users/tangbao/project/dataset/illness/national_illness.csv ; /Users/tangbao/project/illness/national_illness.csv ; /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/dataset/illness/national_illness.csv ; /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main/ILI/national_illness.csv |
