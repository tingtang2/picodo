# Outlier Report Summary

- Run: `pleasant-frost-1504`
- Split: `train`
- Batches: `[1089, 1090, 1091]`
- Steps analyzed: `[1089, 1090, 1091]`
- Pre-spike baseline steps: `[1089, 1090]`
- Spike step: `1091`

| Step | Batch Loss Mean | Batch Loss Min | Batch Loss Max |
|---|---:|---:|---:|
| 1089 | 5.538 | 5.403 | 5.700 |
| 1090 | 5.610 | 5.461 | 5.763 |
| 1091 | 12.820 | 12.098 | 14.004 |

## Activations

**Top 8 Pre-spike Activations Outliers**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_7/mlp_hidden | 6.645 | 5.729 | -0.916 | 23.953 | 43.574 | 19.621 |
| layer_9/mlp_hidden | 6.204 | 5.394 | -0.810 | 18.163 | 19.376 | 1.213 |
| layer_11/mlp_hidden | 5.326 | 4.696 | -0.630 | 25.493 | 22.673 | -2.820 |
| layer_8/mlp_hidden | 5.096 | 4.107 | -0.988 | 27.512 | 36.368 | 8.855 |
| layer_10/mlp_hidden | 5.009 | 4.367 | -0.642 | 30.376 | 27.442 | -2.935 |
| layer_5/mlp_hidden | 4.551 | 4.417 | -0.134 | 73.121 | 80.666 | 7.545 |
| layer_6/mlp_hidden | 4.445 | 4.773 | 0.328 | 49.966 | 53.643 | 3.677 |
| layer_3/mlp_hidden | 3.961 | 4.143 | 0.182 | 91.727 | 84.053 | -7.674 |

**Top 8 Spike-vs-Pre Delta (Activations)**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_9/attn_input | 0.538 | 1.938 | 1.399 | 281.794 | 266.505 | -15.290 |
| layer_10/attn_input | 0.437 | 1.826 | 1.388 | 283.001 | 273.671 | -9.330 |
| layer_9/mlp_input | 0.448 | 1.789 | 1.342 | 282.736 | 272.017 | -10.719 |
| layer_8/mlp_input | 0.557 | 1.846 | 1.289 | 281.323 | 264.527 | -16.796 |
| layer_8/attn_input | 1.098 | 1.710 | 0.612 | 275.748 | 259.377 | -16.371 |
| layer_10/mlp_input | 0.406 | 1.000 | 0.594 | 283.404 | 281.011 | -2.393 |
| layer_11/attn_input | 0.405 | 0.926 | 0.521 | 283.500 | 281.608 | -1.892 |
| layer_7/mlp_input | 0.919 | 1.276 | 0.356 | 272.844 | 255.147 | -17.697 |

## Attention

**Top 8 Pre-spike Attention Outliers**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_8 | 3.180 | 3.141 | -0.039 | 72.181 | 98.083 | 25.903 |
| layer_7 | 1.046 | 1.458 | 0.412 | 25.477 | 23.717 | -1.761 |
| layer_1 | 0.778 | 0.788 | 0.010 | 16.277 | 15.496 | -0.780 |
| layer_5 | 0.763 | 0.726 | -0.037 | 29.476 | 28.452 | -1.024 |
| layer_6 | 0.663 | 0.841 | 0.178 | 27.668 | 25.481 | -2.187 |
| layer_2 | 0.477 | 0.491 | 0.013 | 52.518 | 48.761 | -3.757 |
| layer_4 | 0.474 | 0.429 | -0.045 | 9.819 | 10.272 | 0.453 |
| layer_9 | 0.418 | 0.940 | 0.522 | 147.291 | 62.947 | -84.344 |

**Top 8 Spike-vs-Pre Delta (Attention)**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_9 | 0.418 | 0.940 | 0.522 | 147.291 | 62.947 | -84.344 |
| layer_7 | 1.046 | 1.458 | 0.412 | 25.477 | 23.717 | -1.761 |
| layer_10 | 0.048 | 0.296 | 0.248 | 58.200 | 41.478 | -16.722 |
| layer_6 | 0.663 | 0.841 | 0.178 | 27.668 | 25.481 | -2.187 |
| layer_11 | 0.262 | 0.328 | 0.066 | 17.660 | 16.219 | -1.441 |
| layer_3 | 0.342 | 0.396 | 0.055 | 18.640 | 16.174 | -2.466 |
| layer_2 | 0.477 | 0.491 | 0.013 | 52.518 | 48.761 | -3.757 |
| layer_1 | 0.778 | 0.788 | 0.010 | 16.277 | 15.496 | -0.780 |

## Weights

**Top 8 Pre-spike Weights Outliers**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| blocks.5.mlp.down_proj.kernel | 4.125 | 4.114 | -0.010 | 13.157 | 13.130 | -0.027 |
| blocks.4.mlp.down_proj.kernel | 2.897 | 2.892 | -0.005 | 7.211 | 7.200 | -0.010 |
| blocks.6.mlp.down_proj.kernel | 1.600 | 1.599 | -0.002 | 17.020 | 16.974 | -0.045 |
| blocks.3.mlp.down_proj.kernel | 1.530 | 1.515 | -0.015 | 4.843 | 4.831 | -0.012 |
| blocks.8.attn.out_proj.kernel | 1.018 | 1.006 | -0.012 | 14.899 | 14.893 | -0.006 |
| blocks.11.mlp.down_proj.kernel | 0.985 | 0.989 | 0.004 | 8.255 | 8.245 | -0.010 |
| blocks.2.mlp.down_proj.kernel | 0.729 | 0.725 | -0.004 | 3.398 | 3.395 | -0.004 |
| blocks.10.attn.out_proj.kernel | 0.586 | 0.578 | -0.007 | 4.622 | 4.639 | 0.017 |

**Top 8 Spike-vs-Pre Delta (Weights)**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| blocks.10.mlp.down_proj.kernel | 0.577 | 0.596 | 0.019 | 7.624 | 7.682 | 0.058 |
| blocks.7.mlp.down_proj.kernel | 0.466 | 0.479 | 0.013 | 34.002 | 34.052 | 0.050 |
| blocks.11.mlp.down_proj.kernel | 0.985 | 0.989 | 0.004 | 8.255 | 8.245 | -0.010 |
| blocks.3.attn.out_proj.kernel | 0.205 | 0.207 | 0.002 | 2.355 | 2.360 | 0.005 |
| blocks.9.attn.out_proj.kernel | 0.446 | 0.448 | 0.002 | 5.070 | 5.107 | 0.037 |
| blocks.2.attn.out_proj.kernel | 0.028 | 0.029 | 0.001 | 0.883 | 0.882 | -0.001 |
| blocks.0.attn.out_proj.kernel | 0.017 | 0.018 | 0.000 | 0.697 | 0.695 | -0.002 |
| blocks.1.mlp.down_proj.kernel | 0.122 | 0.122 | 0.000 | 1.717 | 1.715 | -0.002 |
