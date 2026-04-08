import wandb
import pandas as pd
from spike_analysis import compute_spike_score

import re
api = wandb.Api()

entity = "sophie-qx-li-ucla"
project = "loss-spikes-edu"
run_id = "6tru6q8c" 

run = api.run(f"{entity}/{project}/{run_id}")


which_norm = 'grad_norm'

# Use the most basic history call possible to avoid the 1-row API bug
print("🔄 Attempting unfiltered history pull...")
df = run.history(samples=10000) 

# Check the count immediately
print(f"📊 ACTUAL ROWS FETCHED: {len(df)}")

if len(df) <= 1:
    print("❌ API still returning summary only. Switching to full scan...")
    # This is the final manual override
    data = [row for row in run.scan_history()]
    df = pd.DataFrame(data)
    print(f"✅ FINAL ATTEMPT ROWS: {len(df)}")

# Now filter columns locally
grad_cols = [c for c in df.columns if 'grad_norm' in c]
layer_scores = {}

for col in grad_cols:
    series = df[col].dropna()
    if len(series) > 1:
        score, _ = compute_spike_score(series.values, threshold=3.5)
        layer_scores[col] = score

# 2. Define the categories
categories = [
    ("MLP: Up Projections (Expansion)", "mlp.up_proj"),
    ("MLP: Down Projections (Bottleneck)", "mlp.down_proj"),
    ("Attention: QKV Projections", "attn.qkv_proj"),
    ("Attention: Output Projections", "attn.out_proj"),
    ("Embeddings & Global", None) # Catch-all
]

# --- 1. OVERALL TOP 10 RANKING ---
print(f"\n{'='*75}")
print(f"{'TOP 10 MOST STABLE / UNSTABLE LAYERS (OVERALL)':^75}")
print(f"{'='*75}")
print(f"{'Rank':<5} | {'Layer Name':<50} | {'Spike Score'}")
print("-" * 75)

overall_sorted = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(overall_sorted[:10], 1):
    clean_name = name.replace(f"{which_norm}/", "")
    print(f"{i:<5} | {clean_name:<50} | {score:.4f}%")

# --- 2. CATEGORICAL BREAKDOWN ---
# Using the categories list you already have defined in your screenshot
print(f"\n{'='*75}")
print(f"{'RANKED SPIKE SCORES BY COMPONENT TYPE':^75}")
print(f"{'='*75}\n")

seen_keys = set()

for title, pattern in categories:
    print(f"--- {title} ---")
    print(f"{'Kernel Name':<55} | {'Spike Score'}")
    print("-" * 75)

    # Filter for this category
    if pattern:
        current_group = [(k, v) for k, v in layer_scores.items() if pattern in k]
    else:
        # Catch-all for global/embeddings that weren't in the above patterns
        current_group = [(k, v) for k, v in layer_scores.items() if k not in seen_keys]

    # Sort this specific group by score
    current_group.sort(key=lambda x: x[1], reverse=True)

    for name, score in current_group:
        seen_keys.add(name)
        clean_name = name.replace(f'{which_norm}/', '')
        print(f"{clean_name:<55} | {score:.4f}%")
    print()

print(f"{'='*75}")







'''

history = run.history(
    keys=[c for c in run.summary.keys() if which_norm in c] + ['_step'],
    samples=10000
)

'''


'''

layer_scores = {}
for col in [c for c in history.columns if which_norm in c]:
    data = history[col].dropna().values
    score, _ = compute_spike_score(data)
    layer_scores[col] = score

print(f"{'Layer':<50} | {'Spike Score'}")
print("-" * 65)
for name, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name.replace(f'{which_norm}/', ''):<50} | {score:.4f}%")



print(f"\n{'='*75}")
print(f"{'RANKED SPIKE SCORES BY COMPONENT TYPE':^75}")
print(f"{'='*75}\n")

# To keep track of what we've already printed
seen_keys = set()

for title, pattern in categories:
    print(f"--- {title} ---")
    print(f"{'Kernel Name':<55} | {'Spike Score'}")
    print("-" * 75)
    
    # Filter for this category
    if pattern:
        current_group = [(k, v) for k, v in layer_scores.items() if pattern in k]
    else:
        # Catch-all for global/embeddings that weren't in the above patterns
        current_group = [(k, v) for k, v in layer_scores.items() if k not in seen_keys]

    # SORT BY SPIKE SCORE DESCENDING
    current_group.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in current_group:
        seen_keys.add(name)
        clean_name = name.replace(f'{which_norm}/', '')
        print(f"{clean_name:<55} | {score:.4f}%")
    print()

print(f"{'='*75}")
'''
