import wandb
import pandas as pd
from spike_analysis import compute_spike_score

api = wandb.Api()

entity = "sophie-qx-li-ucla"
project = "loss-spikes-edu"
run_id = "6tru6q8c" 

run = api.run(f"{entity}/{project}/{run_id}")

#pull history  - include moment1 norm stuff
history = run.history(
    keys=[c for c in run.summary.keys() if 'moment1_norm' in c] + ['_step'],
    samples=10000
)

layer_scores = {}
for col in [c for c in history.columns if 'moment1_norm' in c]:
    data = history[col].dropna().values
    score, _ = compute_spike_score(data)
    layer_scores[col] = score

print(f"{'Layer':<50} | {'Spike Score'}")
print("-" * 65)
for name, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name.replace('moment1_norm/', ''):<50} | {score:.4f}%")
