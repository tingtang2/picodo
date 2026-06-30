# Loss/Outlier Correlation Summary

- Run: `pleasant-frost-1504`
- Outlier steps: `[1089, 1090, 1091]`
- Outlier split: `train`
- Event count: `150`
- Events with batch-aligned activation/attention analysis: `0`
- Events missing batch alignment for activation/attention: `150`
- Events with activation-position matches: `0`
- Events with attention-key matches: `0`
- Events with token-embedding weight matches: `0`

Batch alignment meanings:
- `exact`: matched via `per_batch`.
- `single_batch_aggregate`: matched from a report that only analyzed one batch.
- `missing_per_batch`: multi-batch outlier report without `per_batch`, so activation/attention joins are unavailable.
- `missing_checkpoint_step`: no outlier checkpoint entry for that loss event step.

## Top Matched Events

| Step | Source | Split | Rank | Loss | Batch | Ex | Pos | Input | Target | Alignment | Activation Matches | Attention Matches | Weight Matches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1091 | top_token_events | valid | 1 | 1138.500 | 40 | 132 | 695 | 1808 | 7607 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 2 | 1088.250 | 10 | 220 | 890 | 5080 | 361 | missing_per_batch | 0 | 0 | 0 |
| 1090 | top_token_events | valid | 1 | 1076.000 | 53 | 112 | 568 | 533 | 4811 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 3 | 1073.000 | 13 | 44 | 965 | 399 | 22372 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 4 | 1071.750 | 37 | 139 | 305 | 23537 | 905 | missing_per_batch | 0 | 0 | 0 |
| 1090 | top_token_events | valid | 2 | 1041.000 | 30 | 19 | 251 | 563 | 1435 | missing_per_batch | 0 | 0 | 0 |
| 1090 | top_token_events | valid | 3 | 992.000 | 56 | 148 | 413 | 273 | 41475 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 5 | 975.000 | 54 | 158 | 470 | 5796 | 3934 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 6 | 958.000 | 46 | 145 | 254 | 19829 | 858 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 7 | 953.000 | 56 | 4 | 705 | 324 | 10332 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 8 | 934.000 | 38 | 72 | 695 | 8301 | 3171 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 9 | 933.000 | 54 | 191 | 385 | 1637 | 517 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 10 | 927.000 | 15 | 18 | 312 | 299 | 363 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 11 | 924.000 | 67 | 19 | 642 | 17723 | 1456 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 12 | 918.000 | 26 | 84 | 638 | 77 | 1759 | missing_per_batch | 0 | 0 | 0 |
| 1090 | top_token_events | valid | 4 | 912.000 | 14 | 140 | 731 | 2945 | 1134 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 13 | 911.000 | 58 | 138 | 842 | 14 | 4627 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 14 | 910.000 | 19 | 87 | 529 | 38222 | 263 | missing_per_batch | 0 | 0 | 0 |
| 1090 | top_token_events | valid | 5 | 907.000 | 27 | 57 | 651 | 54 | 4089 | missing_per_batch | 0 | 0 | 0 |
| 1091 | top_token_events | valid | 15 | 907.000 | 69 | 96 | 431 | 31826 | 17054 | missing_per_batch | 0 | 0 | 0 |

## Activation Match Hotspots

_No matches._

## Attention Match Hotspots

_No matches._

## Embedding Weight Match Hotspots

_No matches._
