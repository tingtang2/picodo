# Outlier Report Summary

- Run: `pleasant-frost-1504`
- Split: `train`
- Batches: `[1089, 1090, 1091]`
- Steps analyzed: `[1089, 1090, 1091]`
- Pre-spike baseline steps: `[1089, 1090]`
- Spike step: `1091`

| Step | Batch Loss Mean | Batch Loss Min | Batch Loss Max |
|---|---:|---:|---:|
| 1089 | 5.539 | 5.403 | 5.700 |
| 1090 | 5.610 | 5.463 | 5.763 |
| 1091 | 12.818 | 12.114 | 13.999 |

## Activations

**Top 8 Pre-spike Activations Outliers**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_7/mlp_hidden | 6.645 | 5.729 | -0.916 | 23.953 | 43.591 | 19.639 |
| layer_9/mlp_hidden | 6.204 | 5.394 | -0.810 | 18.163 | 19.364 | 1.201 |
| layer_11/mlp_hidden | 5.326 | 4.696 | -0.630 | 25.493 | 22.676 | -2.817 |
| layer_8/mlp_hidden | 5.096 | 4.107 | -0.988 | 27.509 | 36.384 | 8.876 |
| layer_10/mlp_hidden | 5.009 | 4.367 | -0.642 | 30.376 | 27.423 | -2.953 |
| layer_5/mlp_hidden | 4.551 | 4.417 | -0.134 | 73.123 | 80.668 | 7.545 |
| layer_6/mlp_hidden | 4.445 | 4.773 | 0.328 | 49.967 | 53.647 | 3.679 |
| layer_3/mlp_hidden | 3.960 | 4.142 | 0.182 | 91.732 | 84.058 | -7.675 |

**Top 8 Spike-vs-Pre Delta (Activations)**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_9/attn_input | 0.538 | 1.938 | 1.400 | 281.795 | 266.498 | -15.298 |
| layer_10/attn_input | 0.437 | 1.826 | 1.389 | 283.002 | 273.670 | -9.332 |
| layer_9/mlp_input | 0.448 | 1.789 | 1.342 | 282.736 | 272.021 | -10.716 |
| layer_8/mlp_input | 0.557 | 1.846 | 1.289 | 281.323 | 264.526 | -16.797 |
| layer_8/attn_input | 1.098 | 1.710 | 0.612 | 275.749 | 259.374 | -16.375 |
| layer_10/mlp_input | 0.406 | 0.999 | 0.594 | 283.404 | 281.016 | -2.388 |
| layer_11/attn_input | 0.405 | 0.926 | 0.521 | 283.501 | 281.609 | -1.891 |
| layer_7/mlp_input | 0.919 | 1.276 | 0.357 | 272.845 | 255.146 | -17.699 |

## Attention

**Top 8 Pre-spike Attention Outliers**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_8 | 3.181 | 3.141 | -0.040 | 72.188 | 98.031 | 25.843 |
| layer_7 | 1.047 | 1.458 | 0.412 | 25.474 | 23.719 | -1.755 |
| layer_1 | 0.778 | 0.788 | 0.010 | 16.279 | 15.498 | -0.780 |
| layer_5 | 0.764 | 0.727 | -0.037 | 29.475 | 28.453 | -1.022 |
| layer_6 | 0.663 | 0.842 | 0.179 | 27.668 | 25.482 | -2.186 |
| layer_2 | 0.477 | 0.491 | 0.014 | 52.514 | 48.766 | -3.748 |
| layer_4 | 0.474 | 0.429 | -0.045 | 9.818 | 10.271 | 0.453 |
| layer_9 | 0.418 | 0.939 | 0.521 | 147.331 | 63.189 | -84.142 |

**Top 8 Spike-vs-Pre Delta (Attention)**

| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| layer_9 | 0.418 | 0.939 | 0.521 | 147.331 | 63.189 | -84.142 |
| layer_7 | 1.047 | 1.458 | 0.412 | 25.474 | 23.719 | -1.755 |
| layer_10 | 0.048 | 0.296 | 0.247 | 58.239 | 41.444 | -16.795 |
| layer_6 | 0.663 | 0.842 | 0.179 | 27.668 | 25.482 | -2.186 |
| layer_11 | 0.262 | 0.328 | 0.066 | 17.672 | 16.250 | -1.423 |
| layer_3 | 0.341 | 0.397 | 0.055 | 18.637 | 16.175 | -2.462 |
| layer_2 | 0.477 | 0.491 | 0.014 | 52.514 | 48.766 | -3.748 |
| layer_1 | 0.778 | 0.788 | 0.010 | 16.279 | 15.498 | -0.780 |

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
