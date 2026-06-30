# Loss/Outlier Correlation Summary

- Run: `pleasant-frost-1504`
- Outlier steps: `[1089, 1090, 1091]`
- Outlier split: `train`
- Event count: `300`
- Events with batch-aligned activation/attention analysis: `100`
- Events missing batch alignment for activation/attention: `200`
- Events with activation-position matches: `0`
- Events with attention-key matches: `0`
- Events with token-embedding weight matches: `2`

Batch alignment meanings:
- `exact`: matched via `per_batch`.
- `single_batch_aggregate`: matched from a report that only analyzed one batch.
- `missing_per_batch`: multi-batch outlier report without `per_batch`, so activation/attention joins are unavailable.
- `missing_checkpoint_step`: no outlier checkpoint entry for that loss event step.

## Top Matched Events

| Step | Source | Split | Rank | Loss | Batch | Ex | Pos | Input | Target | Alignment | Activation Matches | Attention Matches | Weight Matches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1091 | next_train_batch_analysis | train | 1 | 940.000 | 1092 | 149 | 573 | 337 | 9 | missing_per_batch | 0 | 0 | 1 |
| 1089 | top_token_events | train | 4 | 930.500 | 4113 | 200 | 219 | 337 | 1133 | missing_per_batch | 0 | 0 | 1 |
| 1091 | top_token_events | train | 1 | 1587.156 | 8787 | 84 | 292 | 3076 | 1096 | missing_per_batch | 0 | 0 | 0 |
| 1090 | top_token_events | train | 1 | 1507.000 | 879 | 11 | 349 | 558 | 260 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 2 | 1479.219 | 4471 | 59 | 156 | 77 | 43 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 3 | 1478.000 | 500 | 134 | 762 | 2243 | 8448 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 4 | 1476.000 | 681 | 76 | 458 | 324 | 9072 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 5 | 1441.000 | 6639 | 197 | 620 | 48880 | 30837 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 6 | 1416.000 | 4901 | 36 | 740 | 75 | 6654 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 7 | 1388.000 | 7697 | 115 | 200 | 309 | 2777 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 8 | 1373.000 | 8838 | 115 | 651 | 48880 | 27504 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 9 | 1369.000 | 1310 | 157 | 308 | 410 | 3794 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 10 | 1360.000 | 4523 | 151 | 829 | 39899 | 2478 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 11 | 1341.750 | 5538 | 198 | 849 | 47796 | 2571 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 12 | 1341.000 | 3997 | 252 | 754 | 43485 | 1133 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 13 | 1336.000 | 779 | 48 | 454 | 739 | 307 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 14 | 1333.500 | 2772 | 126 | 409 | 360 | 198 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 15 | 1328.125 | 6650 | 199 | 330 | 43485 | 867 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 16 | 1328.000 | 7884 | 83 | 733 | 1489 | 3085 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 17 | 1312.000 | 3 | 34 | 261 | 906 | 14150 | missing_per_batch | 0 | 0 | 0 |

## Activation Match Hotspots

_No matches._

## Attention Match Hotspots

_No matches._

## Embedding Weight Match Hotspots

| Item | Match Count |
|---|---:|
| token_embed_in.embedding | 2 |
