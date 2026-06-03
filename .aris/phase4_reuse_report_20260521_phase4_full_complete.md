# Phase-4 Reuse Report

Generated: 2026-05-20T18:11:45.843996+00:00

## Coverage

- Dataset-horizon cells: 36
- Required seed-runs: 972
- Reusable seed-runs: 972
- Missing seed-runs: 0
- Reusable unique seed cells: 972

## Local Data Integrity

| dataset | path | exists | ok | rows | columns | target_nan | total_nan | sha256 | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | lee_ocil/ETT-small/ETTh1.csv | True | True | 17420 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT | 0 | 0 | f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066 |  |
| ETTh2 | lee_ocil/ETT-small/ETTh2.csv | True | True | 17420 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT | 0 | 0 | a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b |  |
| ETTm1 | lee_ocil/ETT-small/ETTm1.csv | True | True | 69680 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT | 0 | 0 | 6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e |  |
| ETTm2 | lee_ocil/ETT-small/ETTm2.csv | True | True | 69680 | date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT | 0 | 0 | db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd |  |
| Solar1 | Solar/Site_1_50MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 61d2519242e34fc5459d6b8fa5485cc63ad1817331b26c7d935d46ba8fd60502 |  |
| Solar2 | Solar/Site_2_130MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | cd42a3001d36f550fbaa38ceec68ce3e46be10361590a43b3debd13ea7126d34 |  |
| Solar3 | Solar/Site_3_30MW.csv | True | True | 20352 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 843e0ca8e98f3fe7be3d4d01154da31ded6ab99f84d616d3c180e733a2a2cf2a |  |
| Solar4 | Solar/Site_4_130MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 47c851e6e04642770cd94e76e5b6782bcf4203fb08cf4695a8d08fa6bbd70672 |  |
| Solar5 | Solar/Site_5_110MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 46bdff295e5f8a355f4f5d7d06de08d24be6e55beabc4b8d7dd1a528fa887863 |  |
| Solar6 | Solar/Site_6_35MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 61c10a0fbba38cbf9c7e5c2529b2c5403934a0ba90aa28dcee8ff2930c0029ce |  |
| Solar7 | Solar/Site_7_30MW.csv | True | True | 70176 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 1e9864b301eb6ef4b5655c83bfd767505566820d45ea956207dd64ac6df04575 |  |
| Solar8 | Solar/Site_8_30MW.csv | True | True | 69408 | date,Total_Irradiance,Global_Irradiance,Direct_Irradiance,Temperature,Atmosphere,Power | 0 | 0 | 333eb5182abd2182af081a7a4554c66be59609fa88b1734536b955c77bd49216 |  |

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
| Solar4 | 54 |
| Solar5 | 54 |
| Solar6 | 54 |
| Solar7 | 54 |
| Solar8 | 54 |

## Missing By Dataset

_No rows._

## Reuse Rules Applied

- Matched canonical key: dataset, horizon, protocol, activation signature, seed.
- Required protocol: batch_size=32, train_epochs=6, embed=timeF, attn=prob.
- Rejected batch-adjusted/small-batch rows by protocol and run descriptors.
- Required finite mse/mae and readable config where available.
- Arrays are tracked separately through has_arrays for factor/oracle analysis.
