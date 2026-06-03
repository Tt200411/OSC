# Phase-4 Reuse Report

Generated: 2026-05-20T16:33:11.726930+00:00

## Coverage

- Dataset-horizon cells: 36
- Required seed-runs: 972
- Reusable seed-runs: 918
- Missing seed-runs: 54
- Reusable unique seed cells: 918

## Local Data Integrity

| dataset | path | exists | ok | rows | columns | reason |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | lee_ocil/ETT-small/ETTh1.csv | True | True | 17420 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT |  |
| ETTh2 | lee_ocil/ETT-small/ETTh2.csv | True | True | 17420 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT |  |
| ETTm1 | lee_ocil/ETT-small/ETTm1.csv | True | True | 69680 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT |  |
| ETTm2 | lee_ocil/ETT-small/ETTm2.csv | True | True | 69680 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT |  |
| Solar1 | Solar/Site_1_50MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar2 | Solar/Site_2_130MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar3 | Solar/Site_3_30MW.csv | True | True | 20352 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar4 | Solar/Site_4_130MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar5 | Solar/Site_5_110MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar6 | Solar/Site_6_35MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar7 | Solar/Site_7_30MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |
| Solar8 | Solar/Site_8_30MW.csv | True | True | 69408 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power |  |

## Reusable By Dataset

| dataset | reusable_seed_runs |
| --- | --- |
| ETTh1 | 135 |
| ETTh2 | 135 |
| ETTm1 | 135 |
| ETTm2 | 135 |
| Solar1 | 54 |
| Solar2 | 54 |
| Solar3 | 54 |
| Solar5 | 54 |
| Solar6 | 54 |
| Solar7 | 54 |
| Solar8 | 54 |

## Missing By Dataset

| dataset | missing_seed_runs |
| --- | --- |
| Solar4 | 54 |

## Reuse Rules Applied

- Matched canonical key: dataset, horizon, protocol, activation signature, seed.
- Required protocol: batch_size=32, train_epochs=6, embed=timeF, attn=prob.
- Rejected batch-adjusted/small-batch rows by protocol and run descriptors.
- Required finite mse/mae and readable config where available.
- Arrays are tracked separately through has_arrays for factor/oracle analysis.
