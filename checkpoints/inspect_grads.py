import os
import numpy as np
from collections import defaultdict

# base directory (script runs in 'checkpoints/')

#NOTE: change "grad_saving" to your diagnostics directory 
diagnostics_dir = os.path.join("grad_saving", "top_loss_diagnostics")
step = 10000
grad_norms_path = os.path.join(diagnostics_dir, f"grad_norms_step_{step}.npy")

# load grad norms dict
grad_norms = np.load(grad_norms_path, allow_pickle=True).item()
grad_norms_clean = {k: float(v) for k, v in grad_norms.items()}

# group by block
blocks = defaultdict(list)
for k, v in grad_norms_clean.items():
    if "blocks" in k:
        block_idx = int(k.split("/[")[1].split("]")[0])
    else:
        block_idx = -1  # embeddings, token_embed_in/out etc.
    blocks[block_idx].append((k, v))

for block_idx in sorted(blocks.keys()):
    block_name = "Embeddings" if block_idx == -1 else f"Block {block_idx}"
    print(f"\n=== {block_name} ===")
    for name, val in sorted(blocks[block_idx]):
        print(f"{name}: {val:.6f}")

