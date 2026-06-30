# Loss/Outlier Correlation Summary

- Run: `pleasant-frost-1504`
- Outlier steps: `[1089, 1090, 1091]`
- Outlier split: `train`
- Event count: `300`
- Events with batch-aligned activation/attention analysis: `150`
- Events missing batch alignment for activation/attention: `150`
- Events with activation-position matches: `0`
- Events with attention-key matches: `0`
- Events with token-embedding weight matches: `1`

Batch alignment meanings:
- `exact`: matched via `per_batch`.
- `single_batch_aggregate`: matched from a report that only analyzed one batch.
- `missing_per_batch`: multi-batch outlier report without `per_batch`, so activation/attention joins are unavailable.
- `missing_checkpoint_step`: no outlier checkpoint entry for that loss event step.

## Top Matched Events

| Step | Source | Split | Rank | Loss | Batch | Ex | Pos | Input | Target | Alignment | Activation Matches | Attention Matches | Weight Matches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1091 | next_train_batch_analysis | train | 1 | 940.000 | 1092 | 149 | 573 | 337 | 9 | missing_per_batch | 0 | 0 | 1 |
| 1090 | next_train_batch_analysis | train | 1 | 944.000 | 1091 | 3 | 262 | 282 | 19590 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 1 | 897.000 | 1091 | 45 | 623 | 874 | 1517 | exact | 0 | 0 | 0 |
| 1090 | top_token_events | train | 1 | 819.250 | 1090 | 126 | 459 | 404 | 7092 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 2 | 819.000 | 1091 | 172 | 662 | 327 | 519 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 2 | 812.500 | 1092 | 11 | 550 | 897 | 41566 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 2 | 807.000 | 1091 | 37 | 554 | 1974 | 1752 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 3 | 775.500 | 1091 | 3 | 723 | 528 | 2586 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 3 | 770.193 | 1091 | 242 | 323 | 83 | 299 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 4 | 719.018 | 1091 | 0 | 798 | 70 | 286 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 4 | 708.000 | 1091 | 69 | 669 | 11684 | 1619 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 3 | 702.282 | 1092 | 218 | 817 | 263 | 19650 | missing_per_batch | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 4 | 689.000 | 1092 | 158 | 290 | 5667 | 477 | missing_per_batch | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 5 | 685.000 | 1092 | 117 | 419 | 2850 | 11 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | train | 5 | 672.000 | 1091 | 171 | 560 | 6562 | 594 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 6 | 662.500 | 1091 | 110 | 841 | 313 | 11 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 7 | 651.250 | 1091 | 186 | 992 | 82 | 262 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 8 | 648.000 | 1091 | 149 | 410 | 1168 | 3312 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 9 | 639.000 | 1091 | 78 | 326 | 74 | 271 | exact | 0 | 0 | 0 |
| 1091 | top_token_events | train | 10 | 620.000 | 1091 | 20 | 1001 | 370 | 4093 | exact | 0 | 0 | 0 |

## Activation Match Hotspots

_No matches._

## Attention Match Hotspots

_No matches._

## Embedding Weight Match Hotspots

| Item | Match Count |
|---|---:|
| token_embed_in.embedding | 1 |
