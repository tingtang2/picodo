"""
Spike analysis: layer-by-layer diagnostics comparing spiking vs normal tokens.

Usage (from train.py or pdb):
    from spike_analysis import collect_spike_diagnostics, plot_spike_diagnostics

    input_data = jnp.repeat(seq[None, :], 4, axis=0)
    layer_data = collect_spike_diagnostics(model_state, model_graphdef, input_data, spike_pos, normal_pos)
    plot_spike_diagnostics(layer_data, seq[spike_pos + 1], seq[normal_pos + 1], "spike_diagnostics.png")
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax import nnx
from rope import apply_rope


@partial(jax.jit, static_argnames=('model_graphdef',))
def collect_spike_diagnostics(model_state, model_graphdef, x, spike_pos, normal_pos):
    """
    JIT-compiled forward pass that collects per-layer norms and logit lens data.
    Uses the model's own submodules (including flash attention) so results are
    consistent with training.

    Args:
        model_state: model parameters (from nnx.state)
        model_graphdef: model graph definition (from nnx.graphdef)
        x: input tokens [B, T] — pass 4 identical copies for mesh compatibility
        spike_pos: position index of the spiking token
        normal_pos: position index of a normal (low-loss) token

    Returns:
        List of tuples per layer, each containing:
        (residual_norm_spike, residual_norm_normal,
         attn_norm_spike, attn_norm_normal,
         mlp_norm_spike, mlp_norm_normal,
         early_logits_spike [V], early_logits_normal [V])
    """
    model = nnx.merge(model_graphdef, model_state)
    h = model.token_embed_in(x)  # [B, T, D]

    layer_norms = []
    layer_logits_spike = []
    layer_logits_normal = []

    for block in model.blocks:
        # attention sublayer (uses flash attn from graphdef)
        attn_out = block.attn(block.ln1(h))
        h = h + attn_out

        # MLP sublayer
        mlp_out = block.mlp(block.ln2(h))
        h = h + mlp_out

        # norms at both positions
        layer_norms.append((
            jnp.linalg.norm(h[0, spike_pos]),
            jnp.linalg.norm(h[0, normal_pos]),
            jnp.linalg.norm(attn_out[0, spike_pos]),
            jnp.linalg.norm(attn_out[0, normal_pos]),
            jnp.linalg.norm(mlp_out[0, spike_pos]),
            jnp.linalg.norm(mlp_out[0, normal_pos]),
        ))

        # logit lens: project current residual to vocab
        h_normed = model.out_ln(h)
        early_logits = model.token_embed_out.attend(h_normed)  # [B, T, V]
        layer_logits_spike.append(early_logits[0, spike_pos].astype(jnp.float32))
        layer_logits_normal.append(early_logits[0, normal_pos].astype(jnp.float32))

    return layer_norms, layer_logits_spike, layer_logits_normal


def print_spike_diagnostics(layer_norms, layer_logits_spike, layer_logits_normal,
                            target_id_spike, target_id_normal):
    """
    Print per-layer diagnostics to stdout for quick inspection.

    Args:
        layer_norms: list of 6-tuples from collect_spike_diagnostics
        layer_logits_spike: list of logit vectors [V] at spike_pos per layer
        layer_logits_normal: list of logit vectors [V] at normal_pos per layer
        target_id_spike: ground-truth next token id at spike position
        target_id_normal: ground-truth next token id at normal position
    """
    n_layers = len(layer_norms)

    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER SPIKE DIAGNOSTICS")
    print("=" * 80)

    # --- Norms table ---
    print(f"\n{'Layer':<6} | {'Resid (spike)':<14} | {'Resid (norm)':<14} | "
          f"{'Attn (spike)':<14} | {'Attn (norm)':<14} | "
          f"{'MLP (spike)':<14} | {'MLP (norm)':<14}")
    print("-" * 100)
    for i in range(n_layers):
        rs, rn, as_, an, ms, mn = [float(v) for v in layer_norms[i]]
        print(f"{i:<6} | {rs:<14.4f} | {rn:<14.4f} | "
              f"{as_:<14.4f} | {an:<14.4f} | "
              f"{ms:<14.4f} | {mn:<14.4f}")

    # --- Logit lens table ---
    print(f"\n{'Layer':<6} | {'Tgt logit (spike)':<18} | {'Tgt rank (spike)':<18} | "
          f"{'Tgt logit (norm)':<18} | {'Tgt rank (norm)':<18}")
    print("-" * 85)
    for i in range(n_layers):
        logits_s = layer_logits_spike[i]
        logits_n = layer_logits_normal[i]

        tgt_logit_s = float(logits_s[target_id_spike])
        tgt_logit_n = float(logits_n[target_id_normal])

        rank_s = int(jnp.sum(logits_s > logits_s[target_id_spike])) + 1
        rank_n = int(jnp.sum(logits_n > logits_n[target_id_normal])) + 1

        print(f"{i:<6} | {tgt_logit_s:<18.4f} | {rank_s:<18} | "
              f"{tgt_logit_n:<18.4f} | {rank_n:<18}")

    print("=" * 80)


def plot_spike_diagnostics(layer_norms, layer_logits_spike, layer_logits_normal,
                           target_id_spike, target_id_normal, save_path="spike_diagnostics.png"):
    """
    Plot 4-panel figure comparing spike vs normal token across layers.

    Args:
        layer_norms: list of 6-tuples from collect_spike_diagnostics
        layer_logits_spike: list of logit vectors [V] at spike_pos per layer
        layer_logits_normal: list of logit vectors [V] at normal_pos per layer
        target_id_spike: ground-truth next token id at spike position
        target_id_normal: ground-truth next token id at normal position
        save_path: where to save the figure
    """
    import matplotlib.pyplot as plt

    n_layers = len(layer_norms)
    layers = list(range(n_layers))

    # extract norms
    resid_spike = [float(layer_norms[i][0]) for i in layers]
    resid_normal = [float(layer_norms[i][1]) for i in layers]
    attn_spike = [float(layer_norms[i][2]) for i in layers]
    attn_normal = [float(layer_norms[i][3]) for i in layers]
    mlp_spike = [float(layer_norms[i][4]) for i in layers]
    mlp_normal = [float(layer_norms[i][5]) for i in layers]

    # extract logit lens data
    tgt_logits_spike = []
    tgt_logits_normal = []
    tgt_ranks_spike = []
    tgt_ranks_normal = []
    for i in layers:
        ls = layer_logits_spike[i]
        ln = layer_logits_normal[i]
        tgt_logits_spike.append(float(ls[target_id_spike]))
        tgt_logits_normal.append(float(ln[target_id_normal]))
        tgt_ranks_spike.append(int(jnp.sum(ls > ls[target_id_spike])) + 1)
        tgt_ranks_normal.append(int(jnp.sum(ln > ln[target_id_normal])) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) residual stream norm
    ax = axes[0, 0]
    ax.plot(layers, resid_spike, 'r-o', label='spike pos', markersize=4)
    ax.plot(layers, resid_normal, 'b-o', label='normal pos', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm')
    ax.set_title('(a) Residual Stream Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) attention vs MLP contribution norms
    ax = axes[0, 1]
    ax.plot(layers, attn_spike, 'r-o', label='attn (spike)', markersize=4)
    ax.plot(layers, attn_normal, 'b-o', label='attn (normal)', markersize=4)
    ax.plot(layers, mlp_spike, 'r--s', label='mlp (spike)', markersize=4)
    ax.plot(layers, mlp_normal, 'b--s', label='mlp (normal)', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm')
    ax.set_title('(b) Attention & MLP Contribution Norms')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) logit lens: target token rank
    ax = axes[1, 0]
    ax.plot(layers, tgt_ranks_spike, 'r-o', label='spike pos', markersize=4)
    ax.plot(layers, tgt_ranks_normal, 'b-o', label='normal pos', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Target Token Rank')
    ax.set_title('(c) Logit Lens: Target Token Rank')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) logit lens: target token logit value
    ax = axes[1, 1]
    ax.plot(layers, tgt_logits_spike, 'r-o', label='spike pos', markersize=4)
    ax.plot(layers, tgt_logits_normal, 'b-o', label='normal pos', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Target Token Logit')
    ax.set_title('(d) Logit Lens: Target Token Logit Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Spike vs Normal Token: Layer-by-Layer Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spike diagnostics plot to {save_path}")


@partial(jax.jit, static_argnames=('model_graphdef',))
def collect_attention_weights(model_state, model_graphdef, x, spike_pos, normal_pos):
    """
    JIT-compiled forward pass that collects per-layer attention weights
    at the spike and normal query positions.

    Manually applies QK-norm and RoPE to the raw Q/K from qkv_proj,
    then computes softmax(Q @ K^T / sqrt(H)) to get attention weights.

    Args:
        model_state: model parameters (from nnx.state)
        model_graphdef: model graph definition (from nnx.graphdef)
        x: input tokens [B, T] — pass 4 identical copies for mesh compatibility
        spike_pos: position index of the spiking token
        normal_pos: position index of a normal (low-loss) token

    Returns:
        attn_weights_spike: list of [N, T] arrays — attention weights at spike query pos per layer
        attn_weights_normal: list of [N, T] arrays — attention weights at normal query pos per layer
    """
    model = nnx.merge(model_graphdef, model_state)
    h = model.token_embed_in(x)  # [B, T, D]
    B, T, D = h.shape

    position = jnp.arange(T)
    attn_weights_spike = []
    attn_weights_normal = []

    for block in model.blocks:
        # get raw Q, K, V from qkv_proj
        ln_h = block.ln1(h)
        q, k, v = block.attn.qkv_proj(ln_h)  # each [B, T, N, H]

        # apply QK-norm
        q = block.attn.query_norm(q)
        k = block.attn.key_norm(k)

        # apply RoPE
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

        # compute attention weights: softmax(Q @ K^T / sqrt(H))
        H = q.shape[-1]
        # q, k are [B, T, N, H] -> transpose to [B, N, T, H] for matmul
        q_t = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)  # [B, N, T, H]
        k_t = jnp.transpose(k, (0, 2, 1, 3)).astype(jnp.float32)  # [B, N, T, H]
        scores = jnp.matmul(q_t, jnp.transpose(k_t, (0, 1, 3, 2))) / jnp.sqrt(H)  # [B, N, T, T]

        # causal mask
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None], scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)  # [B, N, T, T]

        # extract attention at spike and normal query positions (batch 0)
        attn_weights_spike.append(weights[0, :, spike_pos, :])   # [N, T]
        attn_weights_normal.append(weights[0, :, normal_pos, :]) # [N, T]

        # still need to advance the residual stream using the model's own forward pass
        attn_out = block.attn(ln_h)
        h = h + attn_out
        mlp_out = block.mlp(block.ln2(h))
        h = h + mlp_out

    return attn_weights_spike, attn_weights_normal


def plot_attention_weights(attn_weights_spike, attn_weights_normal,
                           spike_pos, normal_pos, layers_to_plot=None,
                           save_path="spike_attention.png"):
    """
    Plot attention weight heatmaps comparing spike vs normal query positions.

    Args:
        attn_weights_spike: list of [N, T] arrays per layer
        attn_weights_normal: list of [N, T] arrays per layer
        spike_pos: position index of the spiking token
        normal_pos: position index of the normal token
        layers_to_plot: list of layer indices to plot (default: layers 3-6 where divergence happens)
        save_path: where to save the figure
    """
    import matplotlib.pyplot as plt

    n_layers = len(attn_weights_spike)
    N = attn_weights_spike[0].shape[0]  # number of heads

    if layers_to_plot is None:
        # default to layers around the divergence point
        layers_to_plot = list(range(min(3, n_layers), min(7, n_layers)))


    for layer_idx in layers_to_plot:
            ws = np.array(attn_weights_spike[layer_idx])  # [N, T]
            print(f"\nLayer {layer_idx}:")
            
            # Iterate through each head to see if it has 'collapsed'
            for head_idx in range(N):
                head_weights = ws[head_idx, :spike_pos + 1]
                top_indices = np.argsort(head_weights)[-3:][::-1]
                top_values = head_weights[top_indices]
                total_mass = np.sum(head_weights)
                
                # This printout will confirm if the 'solid bar' sums to ~1.0
                head_weights_unmask = ws[head_idx, :]
                tot_mass_unmask = np.sum(head_weights_unmask)
                print(f"  Head {head_idx}: Top Weights {top_values} at Indices {top_indices} (Sum (causal): {total_mass:.4f}); Sum (unmask,total): {tot_mass_unmask:.4f}")

                # Verification Logic
                if top_values[0] > 0.95:
                    print(f"    ⚠️ CRITICAL: Head {head_idx} has saturated on index {top_indices[0]}")

    n_plot_layers = len(layers_to_plot)
    fig, axes = plt.subplots(n_plot_layers, 2, figsize=(16, 4 * n_plot_layers))
    if n_plot_layers == 1:
        axes = axes[None, :]

    V_MAX = 0.4
    for row, layer_idx in enumerate(layers_to_plot):
        ws = np.array(attn_weights_spike[layer_idx])   # [N, T]
        wn = np.array(attn_weights_normal[layer_idx])   # [N, T]

        # only show keys up to the query position (causal — rest is zero)
        ws_crop = ws[:, :spike_pos + 1]
        wn_crop = wn[:, :normal_pos + 1]

        ax = axes[row, 0]
        #im = ax.imshow(ws_crop, aspect='auto', cmap='hot', interpolation='nearest')
        im = ax.imshow(ws_crop, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=V_MAX)
        ax.set_title(f'Layer {layer_idx} — Spike pos {spike_pos}')
        ax.set_ylabel('Head')
        ax.set_xlabel('Key position')
        plt.colorbar(im, ax=ax, fraction=0.02)

        ax = axes[row, 1]
        im = ax.imshow(wn_crop, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=V_MAX)
        ax.set_title(f'Layer {layer_idx} — Normal pos {normal_pos}')
        ax.set_ylabel('Head')
        ax.set_xlabel('Key position')
        plt.colorbar(im, ax=ax, fraction=0.02)

    plt.suptitle('Attention Weights: Spike (left) vs Normal (right)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention weight plot to {save_path}")


def compute_spike_score(series, window=1000, threshold=7.0):
    """
    Compute spike score: percentage of values that are >= `threshold` standard
    deviations from a rolling average of the last `window` values.

    Reference: Wortsman et al. — "spike score as an objective measure."

    Args:
        series: 1D array-like of time series values (e.g. training loss, grad norm)
        window: rolling average window size (default 1000)
        threshold: number of std devs to count as a spike (default 7.0)

    Returns:
        spike_score: float, percentage of values that are spikes (0-100)
        spike_indices: list of indices where spikes occur
    """
    series = np.array(series, dtype=np.float64)
    n = len(series)
    if n < window + 1:
        return 0.0, []

    spike_count = 0
    spike_indices = []

    for i in range(window, n):
        w = series[i - window:i]
        mu = np.mean(w)
        sigma = np.std(w)
        if sigma > 0 and np.abs(series[i] - mu) >= threshold * sigma:
            spike_count += 1
            spike_indices.append(i)

    spike_score = 100.0 * spike_count / (n - window)
    return spike_score, spike_indices


def plot_attn_diagnostics(ws, loss_per_token, layer_idx=4, save_path=None):
    import matplotlib.pyplot as plt
    # ws shape: (layers, heads, q, k)
    # Get weights for the specific layer and average across heads
    # (q, k)
    attn_layer = ws[layer_idx].mean(axis=0) 
    
    # 1. Calculate Shannon Entropy: -sum(p * log(p))
    # We add epsilon to avoid log(0)
    epsilon = 1e-12
    entropy = -np.sum(attn_layer * np.log(attn_layer + epsilon), axis=-1)

    # loss_per_token has T-1 entries (no loss for position 0), align by dropping first query pos
    n_loss = len(loss_per_token)
    entropy = entropy[-n_loss:]
    sink_mass_all = attn_layer[:, 0][-n_loss:]

    # 2. Setup Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot A: Entropy vs Loss Scatter
    # Each dot is a token in the sequence
    ax1.scatter(entropy, loss_per_token, alpha=0.5, c='tab:blue', edgecolors='none')
    ax1.set_title(f'Layer {layer_idx}: Attention Entropy vs. Token Loss')
    ax1.set_xlabel('Shannon Entropy (Higher = Flatter Attention)')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot B: Attention "Sink" Importance
    # Mass assigned to the first token (index 0) vs Loss
    ax2.scatter(sink_mass_all, loss_per_token, alpha=0.5, c='tab:orange', edgecolors='none')
    ax2.set_title(f'Layer {layer_idx}: Sink Token (Pos 0) Mass vs. Loss')
    ax2.set_xlabel('Attention Weight on Token 0')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f">>> Diagnostic plot saved to: {save_path}")


def collect_full_attention_weights(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    h = model.token_embed_in(x)  # [B, T, D]
    B, T, D = h.shape
    position = jnp.arange(T)
    
    all_layers_weights = []

    for block in model.blocks:
        ln_h = block.ln1(h)
        # get raw Q, K, V
        q, k, v = block.attn.qkv_proj(ln_h)
        
        # Norm and RoPE
        q = block.attn.query_norm(q)
        k = block.attn.key_norm(k)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])
        
        # Transpose for matmul: [B, T, N, H] -> [B, N, T, H]
        q_t = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)
        k_t = jnp.transpose(k, (0, 2, 1, 3)).astype(jnp.float32)
        
        H_dim = q.shape[-1]
        scores = jnp.matmul(q_t, jnp.transpose(k_t, (0, 1, 3, 2))) / jnp.sqrt(H_dim)
        
        # Causal mask
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None], scores, -1e9)
        
        # Softmax to get full [B, N, T, T]
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Store batch 0: [N, T, T]
        all_layers_weights.append(weights[0])
        
        # Continue forward pass to get correct h for next layer
        h = h + block.attn(ln_h)
        h = h + block.mlp(block.ln2(h))
        
    # Stack to get [Layers, Heads, Q, K]
    return jnp.stack(all_layers_weights)



