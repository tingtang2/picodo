# Loss/Outlier Correlation Summary

- Run: `pleasant-frost-1504`
- Outlier steps: `[1090, 1091, 1092]`
- Outlier split: `train`
- Event count: `150`
- Events with batch-aligned activation/attention analysis: `150`
- Events missing batch alignment for activation/attention: `0`
- Events with activation-position matches: `0`
- Events with attention-key matches: `1`
- Events with token-embedding weight matches: `1`

Batch alignment meanings:
- `exact`: matched via `per_batch`.
- `single_batch_aggregate`: matched from a report that only analyzed one batch.
- `missing_per_batch`: multi-batch outlier report without `per_batch`, so activation/attention joins are unavailable.
- `missing_checkpoint_step`: no outlier checkpoint entry for that loss event step.

## Top Matched Events

| Step | Source | Split | Rank | Loss | Batch | Ex | Pos | Input | Target | Alignment | Activation Matches | Attention Matches | Weight Matches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1091 | next_train_batch_analysis | train | 1 | 940.000 | 1092 | 149 | 573 | 337 | 9 | exact | 0 | 0 | 1 |
| 1092 | next_train_batch_analysis | train | 37 | 81.776 | 1093 | 144 | 4 | 2747 | 261 | exact | 0 | 1 | 0 |
| 1090 | next_train_batch_analysis | train | 1 | 944.000 | 1091 | 3 | 262 | 282 | 19590 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 2 | 819.000 | 1091 | 172 | 662 | 327 | 519 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 2 | 812.500 | 1092 | 11 | 550 | 897 | 41566 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 3 | 775.500 | 1091 | 3 | 723 | 528 | 2586 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 4 | 719.018 | 1091 | 0 | 798 | 70 | 286 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 3 | 702.282 | 1092 | 218 | 817 | 263 | 19650 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 4 | 689.000 | 1092 | 158 | 290 | 5667 | 477 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 5 | 685.000 | 1092 | 117 | 419 | 2850 | 11 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 6 | 618.000 | 1092 | 91 | 477 | 3586 | 286 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 7 | 599.000 | 1092 | 195 | 829 | 1084 | 1279 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 8 | 598.500 | 1092 | 56 | 294 | 8 | 2585 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 9 | 597.500 | 1092 | 26 | 609 | 641 | 13514 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 10 | 596.000 | 1092 | 47 | 702 | 3938 | 11 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 5 | 581.500 | 1091 | 2 | 684 | 8800 | 347 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 6 | 570.000 | 1091 | 164 | 785 | 3802 | 291 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 7 | 560.000 | 1091 | 3 | 696 | 7295 | 1589 | exact | 0 | 0 | 0 |
| 1090 | next_train_batch_analysis | train | 8 | 558.500 | 1091 | 164 | 917 | 66 | 290 | exact | 0 | 0 | 0 |
| 1091 | next_train_batch_analysis | train | 11 | 542.750 | 1092 | 161 | 981 | 1243 | 661 | exact | 0 | 0 | 0 |

## Activation Match Hotspots

_No matches._

## Attention Match Hotspots

| Item | Match Count |
|---|---:|
| layer_10 | 1 |

## Embedding Weight Match Hotspots

| Item | Match Count |
|---|---:|
| token_embed_in.embedding | 1 |
