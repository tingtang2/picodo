import math
import warnings
import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from omegaconf.dictconfig import DictConfig
from rope import apply_rope
from omegaconf import ListConfig


_UPDATE_METRIC_EPS = 1e-30
_SPECTRAL_POWER_ITERATIONS = 5


def _frobenius_norm(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(x)))


def _safe_ratio(numerator, denominator):
    return numerator / jnp.maximum(denominator, jnp.asarray(_UPDATE_METRIC_EPS, dtype=jnp.float32))


def _rmsnorm_operator_input(norm: nnx.RMSNorm, x):
    """Returns the normalized activation to which RMSNorm's scale is applied."""
    x_f32 = jnp.asarray(x, dtype=jnp.float32)
    mean_square = jnp.mean(jnp.square(x_f32), axis=norm.reduction_axes, keepdims=True)
    return x_f32 * jax.lax.rsqrt(mean_square + jnp.asarray(norm.epsilon, dtype=jnp.float32))


def _apply_rmsnorm_with_capture(norm, x, captures, path):
    y = norm(x)
    if captures is not None and getattr(norm, "scale", None) is not None:
        captures[path] = {
            "input": _rmsnorm_operator_input(norm, x),
            "raw_input_fro": _frobenius_norm(x),
            "output_fro": _frobenius_norm(y),
        }
    return y


def _spectral_norm_estimate(matrix, iterations: int = _SPECTRAL_POWER_ITERATIONS):
    """Deterministic two-probe power-iteration estimate of a matrix spectral norm."""
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    probe_index = jnp.arange(matrix.shape[1], dtype=jnp.float32) + 1.0
    probes = jnp.stack(
        (
            jnp.sin(probe_index * 0.734375) + jnp.cos(probe_index * 1.234375),
            jnp.sin(probe_index * 1.6171875) - jnp.cos(probe_index * 0.4453125),
        ),
        axis=1,
    )

    def normalize_columns(vectors):
        norms = jnp.sqrt(jnp.sum(jnp.square(vectors), axis=0, keepdims=True))
        return vectors / jnp.maximum(norms, _UPDATE_METRIC_EPS)

    vectors = normalize_columns(probes)

    def power_step(_, current):
        return normalize_columns(matrix.T @ (matrix @ current))

    vectors = jax.lax.fori_loop(0, iterations, power_step, vectors)
    estimates = jnp.sqrt(jnp.sum(jnp.square(matrix @ vectors), axis=0))
    return jnp.max(estimates)


def _linear_output_fro_from_grams(operator_input, matrix):
    """Computes ||XW||_F without materializing XW."""
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    flattened_input = jnp.asarray(operator_input, dtype=jnp.float32).reshape(-1, matrix.shape[0])
    input_gram = flattened_input.T @ flattened_input
    weight_gram = matrix @ matrix.T
    squared_norm = jnp.sum(input_gram * weight_gram)
    return jnp.sqrt(jnp.maximum(squared_norm, 0.0))


def _add_matrix_update_metrics(
    metrics,
    path,
    old_weight,
    new_weight,
    operator_input,
    old_output_norm,
    delta_output=None,
    old_matrix=None,
    new_matrix=None,
    delta_output_norm=None,
):
    old_weight = jnp.asarray(old_weight, dtype=jnp.float32)
    new_weight = jnp.asarray(new_weight, dtype=jnp.float32)
    delta_weight = new_weight - old_weight
    delta_weight_norm = _frobenius_norm(delta_weight)
    prefix = f"parameter_update/{path}"
    metrics[f"{prefix}/delta_param_fro"] = delta_weight_norm
    metrics[f"{prefix}/delta_param_rel_fro"] = _safe_ratio(delta_weight_norm, _frobenius_norm(old_weight))
    metrics[f"{prefix}/input_fro"] = _frobenius_norm(operator_input)
    if delta_output_norm is None:
        delta_output_norm = _frobenius_norm(delta_output)
    metrics[f"{prefix}/delta_output_fro"] = delta_output_norm
    metrics[f"{prefix}/delta_output_rel_fro"] = _safe_ratio(delta_output_norm, old_output_norm)

    old_matrix = old_weight if old_matrix is None else jnp.asarray(old_matrix, dtype=jnp.float32)
    new_matrix = new_weight if new_matrix is None else jnp.asarray(new_matrix, dtype=jnp.float32)
    delta_matrix = new_matrix - old_matrix
    metrics[f"{prefix}/delta_param_rel_spectral_estimate"] = _safe_ratio(
        _spectral_norm_estimate(delta_matrix),
        _spectral_norm_estimate(old_matrix),
    )


def _add_scale_update_metrics(metrics, path, old_scale, new_scale, capture):
    old_scale = jnp.asarray(old_scale, dtype=jnp.float32)
    new_scale = jnp.asarray(new_scale, dtype=jnp.float32)
    delta_scale = new_scale - old_scale
    operator_input = jnp.asarray(capture["input"], dtype=jnp.float32)
    delta_output = operator_input * delta_scale
    prefix = f"parameter_update/{path}"
    delta_scale_norm = _frobenius_norm(delta_scale)
    metrics[f"{prefix}/delta_param_fro"] = delta_scale_norm
    metrics[f"{prefix}/delta_param_rel_fro"] = _safe_ratio(delta_scale_norm, _frobenius_norm(old_scale))
    metrics[f"{prefix}/input_fro"] = _frobenius_norm(operator_input)
    metrics[f"{prefix}/raw_input_fro"] = capture["raw_input_fro"]
    delta_output_norm = _frobenius_norm(delta_output)
    metrics[f"{prefix}/delta_output_fro"] = delta_output_norm
    metrics[f"{prefix}/delta_output_rel_fro"] = _safe_ratio(delta_output_norm, capture["output_fro"])
    # A learned elementwise scale is a diagonal linear operator.
    metrics[f"{prefix}/delta_param_rel_spectral"] = _safe_ratio(
        jnp.max(jnp.abs(delta_scale)),
        jnp.max(jnp.abs(old_scale)),
    )


def _symmetric_uniform(bound: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def _megatron_init_stds(c: DictConfig) -> tuple[float, float] | None:
    """Return the input and residual projection stds for Megatron init."""
    if not bool(getattr(c, "megatron_init", False)):
        return None

    init_base_std = float(getattr(c, "init_base_std", 0.01))
    if init_base_std <= 0.0:
        raise ValueError(f"Expected `model.init_base_std` to be positive, got {init_base_std}.")

    residual_std = init_base_std / math.sqrt(2.0 * c.L)
    return init_base_std, residual_std


class TransformerDecoder(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        lm_head_dtype = getattr(c, "lm_head_dtype", c.activ_dtype)
        rmsnorm_use_scale = bool(getattr(c, "rmsnorm_use_scale", False))
        self.final_hidden_mean_centering = bool(getattr(c, "final_hidden_mean_centering", False))
        self.final_hidden_mean_centering_coeff = float(getattr(c, "alpha", 1.0))
        self.lm_head_oblique_learn_target_rms = bool(getattr(c, "lm_head_oblique_learn_target_rms", False))
        self.lm_head_oblique_initial_target_rms_from_random_init = bool(
            getattr(c, "lm_head_oblique_initial_target_rms_from_random_init", False)
        )
        self.lm_head_oblique_optimizer_type = str(getattr(c, "lm_head_oblique_optimizer_type", "row_oblique")).lower()
        megatron_stds = _megatron_init_stds(c)
        input_embedding_init = {} if megatron_stds is None else {
            "embedding_init": jax.nn.initializers.normal(megatron_stds[0])
        }
        # Lingua's Megatron path leaves its untied LM head at PyTorch Linear's
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) initialization.
        output_embedding_init = {} if megatron_stds is None else {
            "embedding_init": _symmetric_uniform(c.D ** -0.5)
        }
        self.token_embed_in = nnx.Embed(
            num_embeddings=c.V,
            features=c.D,
            dtype=c.activ_dtype,
            **input_embedding_init,
            rngs=rngs,
        )
        self.token_embed_out = nnx.Embed(
            num_embeddings=c.V,
            features=c.D,
            dtype=lm_head_dtype,
            **output_embedding_init,
            rngs=rngs,
        )
        if self.lm_head_oblique_learn_target_rms:
            if self.lm_head_oblique_initial_target_rms_from_random_init:
                output_embedding = jnp.asarray(self.token_embed_out.embedding.value, dtype=jnp.float32)
                if self.lm_head_oblique_optimizer_type == "column_oblique":
                    col_rms = jnp.sqrt(jnp.mean(jnp.square(output_embedding), axis=0))
                    initial_target_rms = jnp.mean(col_rms)
                else:
                    row_rms = jnp.sqrt(jnp.mean(jnp.square(output_embedding), axis=1))
                    initial_target_rms = jnp.mean(row_rms)
            else:
                initial_target_rms = float(getattr(c, "lm_head_oblique_initial_target_rms", 1.0))
                if initial_target_rms <= 0.0:
                    raise ValueError(
                        f"Expected `model.lm_head_oblique_initial_target_rms` to be positive, got {initial_target_rms}."
                    )
            self.lm_head_oblique_target_rms_log = nnx.Param(
                jnp.asarray(jnp.log(initial_target_rms), dtype=jnp.float32)
            )
        self.blocks = nnx.List(TransformerBlock(c, rngs, layer_idx=i) for i in range(c.L))
        self.out_ln = nnx.RMSNorm(c.D, use_scale=rmsnorm_use_scale, dtype=lm_head_dtype, rngs=rngs)
        
    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        return_update_inputs: bool = False,
        skip_output_logits: bool = False,
    ): # [B, S]
        qkv_outputs = {}
        update_inputs = {} if return_update_inputs else None
        # token embedding
        h = self.token_embed_in(x) # [B, T, D]
        if update_inputs is not None:
            update_inputs["token_embed_in.embedding"] = {
                "input": x,
                "output_fro": _frobenius_norm(h),
            }

        # transformer blocks
        for i, block in enumerate(self.blocks):
            if return_qkv:
                h, qkv = block(
                    h,
                    attention_mask=attention_mask,
                    return_qkv=True,
                    update_inputs=update_inputs,
                    metric_prefix=f"blocks.{i}",
                )
                qkv_outputs[i] = qkv
            else:
                h = block(
                    h,
                    attention_mask=attention_mask,
                    update_inputs=update_inputs,
                    metric_prefix=f"blocks.{i}",
                )

        # project back to vocabulary
        h = _apply_rmsnorm_with_capture(self.out_ln, h, update_inputs, "out_ln.scale")
        if self.final_hidden_mean_centering:
            h = h - self.final_hidden_mean_centering_coeff * jnp.mean(h, axis=-1, keepdims=True)
        if self.lm_head_oblique_learn_target_rms:
            unscaled_h = h
            target_rms = jnp.exp(jnp.asarray(self.lm_head_oblique_target_rms_log.value, dtype=h.dtype))
            h = h * target_rms
            if update_inputs is not None:
                update_inputs["lm_head_oblique_target_rms_log"] = {
                    "input": unscaled_h,
                    "output_fro": _frobenius_norm(h),
                }
        if update_inputs is not None:
            update_inputs["token_embed_out.embedding"] = {"input": h}

        if skip_output_logits:
            if not return_update_inputs:
                raise ValueError("skip_output_logits requires return_update_inputs=True.")
            if return_qkv:
                return h, qkv_outputs, update_inputs
            return h, update_inputs

        logits = self.token_embed_out.attend(h) # [B, T, V]
        if update_inputs is not None:
            update_inputs["token_embed_out.embedding"]["output_fro"] = _frobenius_norm(logits)

        if return_qkv and return_update_inputs:
            return logits, qkv_outputs, update_inputs
        if return_qkv:
            return logits, qkv_outputs
        if return_update_inputs:
            return logits, update_inputs

        return logits


def center_output_embeddings(model: TransformerDecoder):
    embeddings = model.token_embed_out.embedding.value
    mean = jnp.mean(embeddings.astype(jnp.float32), axis=0, keepdims=True).astype(embeddings.dtype)
    model.token_embed_out.embedding.value = embeddings - mean


def get_average_output_embedding_row_rms(model: TransformerDecoder):
    embeddings = jnp.asarray(model.token_embed_out.embedding.value, dtype=jnp.float32)
    row_rms = jnp.sqrt(jnp.mean(jnp.square(embeddings), axis=1))
    return float(jnp.mean(row_rms))


def get_average_output_embedding_col_rms(model: TransformerDecoder):
    embeddings = jnp.asarray(model.token_embed_out.embedding.value, dtype=jnp.float32)
    col_rms = jnp.sqrt(jnp.mean(jnp.square(embeddings), axis=0))
    return float(jnp.mean(col_rms))


def row_normalize_output_embeddings(model: TransformerDecoder, target_rms: float = 1.0, eps: float = 1e-8):
    embeddings = model.token_embed_out.embedding.value
    embeddings_f32 = embeddings.astype(jnp.float32)
    row_norms = jnp.linalg.norm(embeddings_f32, axis=1, keepdims=True)
    safe_row_norms = jnp.maximum(row_norms, jnp.asarray(eps, dtype=jnp.float32))
    target_norm = jnp.asarray(target_rms, dtype=jnp.float32) * jnp.sqrt(
        jnp.asarray(embeddings.shape[1], dtype=jnp.float32)
    )
    model.token_embed_out.embedding.value = (embeddings_f32 * (target_norm / safe_row_norms)).astype(
        embeddings.dtype
    )


def column_normalize_output_embeddings(model: TransformerDecoder, target_rms: float = 1.0, eps: float = 1e-8):
    embeddings = model.token_embed_out.embedding.value
    embeddings_f32 = embeddings.astype(jnp.float32)
    col_norms = jnp.linalg.norm(embeddings_f32, axis=0, keepdims=True)
    safe_col_norms = jnp.maximum(col_norms, jnp.asarray(eps, dtype=jnp.float32))
    target_norm = jnp.asarray(target_rms, dtype=jnp.float32) * jnp.sqrt(
        jnp.asarray(embeddings.shape[0], dtype=jnp.float32)
    )
    model.token_embed_out.embedding.value = (embeddings_f32 * (target_norm / safe_col_norms)).astype(
        embeddings.dtype
    )


class TransformerBlock(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs, layer_idx: int):
        rmsnorm_use_scale = bool(getattr(c, "rmsnorm_use_scale", False))
        self.ln1 = nnx.RMSNorm(c.D, use_scale=rmsnorm_use_scale, dtype=c.activ_dtype, rngs=rngs)
        self.ln2 = nnx.RMSNorm(c.D, use_scale=rmsnorm_use_scale, dtype=c.activ_dtype, rngs=rngs)
        
        qk_config = c.use_qk_norm
        
        if isinstance(qk_config, (list, tuple, ListConfig)):
            use_qk_norm = layer_idx in qk_config
        else:
            use_qk_norm = bool(qk_config)
        
        self.attn = MultiHeadAttention(c, rngs, use_qk_norm=use_qk_norm)
        self.mlp = MLP(c, rngs)
        
    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        update_inputs=None,
        metric_prefix: str = "",
    ): # [B, T, D]
        ln1_out = _apply_rmsnorm_with_capture(self.ln1, x, update_inputs, f"{metric_prefix}.ln1.scale")
        if return_qkv:
            attn_out, qkv = self.attn(
                ln1_out,
                attention_mask=attention_mask,
                return_qkv=True,
                update_inputs=update_inputs,
                metric_prefix=f"{metric_prefix}.attn",
            )
            x = x + attn_out
        else:
            x = x + self.attn(
                ln1_out,
                attention_mask=attention_mask,
                update_inputs=update_inputs,
                metric_prefix=f"{metric_prefix}.attn",
            )
        
        ln2_out = _apply_rmsnorm_with_capture(self.ln2, x, update_inputs, f"{metric_prefix}.ln2.scale")
        x = x + self.mlp(ln2_out, update_inputs=update_inputs, metric_prefix=f"{metric_prefix}.mlp")

        if return_qkv:
            return x, qkv

        return x


class MultiHeadAttention(nnx.Module):
    """Causal attention layer."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs, use_qk_norm: bool = None):
        rmsnorm_use_scale = bool(getattr(c, "rmsnorm_use_scale", False))
        megatron_stds = _megatron_init_stds(c)
        qkv_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[0])
        }
        out_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[1])
        }
        self.qkv_proj = nnx.Einsum(
            'BTd,SNdH->SBTNH',
            (3, c.N, c.D, c.H),
            dtype=c.activ_dtype,
            **qkv_init,
            rngs=rngs,
        )
        self.out_proj = nnx.Einsum(
            'BTnh,nhD->BTD',
            (c.N, c.H, c.D),
            dtype=c.activ_dtype,
            **out_init,
            rngs=rngs,
        )
        self.elementwise_attn_output_gate = bool(getattr(c, "elementwise_attn_output_gate", False))
        if self.elementwise_attn_output_gate:
            self.gate_proj = nnx.Einsum(
                'BTd,NdH->BTNH',
                (c.N, c.D, c.H),
                dtype=c.activ_dtype,
                **qkv_init,
                rngs=rngs,
            )
        
        if use_qk_norm is None:
            use_qk_norm = c.use_qk_norm
        
        self.query_norm = nnx.RMSNorm(c.H, use_scale=rmsnorm_use_scale, dtype=c.activ_dtype, rngs=rngs) if use_qk_norm else nnx.identity
        self.key_norm = nnx.RMSNorm(c.H, use_scale=rmsnorm_use_scale, dtype=c.activ_dtype, rngs=rngs) if use_qk_norm else nnx.identity
        if c.use_flash_attn and jax.devices()[0].platform == 'tpu' and (c.H % 128 != 0):
            warnings.warn('cannot use flash attention because `model.H` is not a multiple of 128.')
        c.use_flash_attn &= jax.devices()[0].platform == 'tpu'
        c.use_flash_attn &= (c.H % 128 == 0)
        self.attention = partial(tpu_causal_flash_attention) if c.use_flash_attn else partial(jax.nn.dot_product_attention, is_causal=False)
        self.use_flash_attn = c.use_flash_attn

    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        update_inputs=None,
        metric_prefix: str = "",
    ): # [B, T, D]
        B, T, D = x.shape

        # input projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv # [B, T, N, H]
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.qkv_proj.kernel"] = {
                "input": x,
                "output_fro": _frobenius_norm(qkv),
            }
        gate_score = self.gate_proj(x) if self.elementwise_attn_output_gate else None
        if update_inputs is not None and gate_score is not None:
            update_inputs[f"{metric_prefix}.gate_proj.kernel"] = {
                "input": x,
                "output_fro": _frobenius_norm(gate_score),
            }
        if return_qkv:
            raw_qkv = (q, k, v)

        # qk-norm
        if isinstance(self.query_norm, nnx.RMSNorm):
            q = _apply_rmsnorm_with_capture(
                self.query_norm, q, update_inputs, f"{metric_prefix}.query_norm.scale"
            )
        else:
            q = self.query_norm(q)
        if isinstance(self.key_norm, nnx.RMSNorm):
            k = _apply_rmsnorm_with_capture(
                self.key_norm, k, update_inputs, f"{metric_prefix}.key_norm.scale"
            )
        else:
            k = self.key_norm(k)

        # position embedding
        position = jnp.arange(T)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

        # attention
        if self.use_flash_attn:
            out = self.attention(q, k, v) # [B, T, N, H]
        else:
             # 1. Create causal mask (allows attending to past)
            # Shape [1, 1, T, T]
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_)).reshape(1, 1, T, T)
            
            # 2. Create padding mask (if provided)
            if attention_mask is not None:
                # attention_mask is [B, T]. Reshape to [B, 1, 1, T] (for broadcasting)
                # This mask is 1 (True) for real tokens and 0 (False) for padding
                padding_mask = attention_mask.astype(jnp.bool_).reshape(B, 1, 1, T)
                
                # 3. Combine masks: Both must be True to attend.
                # causal_mask broadcasts to [B, 1, T, T]
                # padding_mask broadcasts to [B, 1, T, T]
                # combined_mask shape [B, 1, T, T]
                combined_mask = jnp.logical_and(causal_mask, padding_mask)
            else:
                combined_mask = causal_mask
            
            # The mask will be broadcast by dot_product_attention to [B, N, T, T]
            out = self.attention(q, k, v, mask=combined_mask) # [B, T, N, H]

        if self.elementwise_attn_output_gate:
            out = out * jax.nn.sigmoid(gate_score)

        # output projection followed by contraction back to original dims
        out_proj_input = out
        out = self.out_proj(out_proj_input) # [B, T, D]
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.out_proj.kernel"] = {
                "input": out_proj_input,
                "output_fro": _frobenius_norm(out),
            }
        if return_qkv:
            return out, raw_qkv
        return out


def tpu_causal_flash_attention(q, k, v):
    """
    TPU Flash Attention.
    https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py
    https://github.com/AI-Hypercomputer/maxtext/blob/9ea52118535e970096c164460dbbfa478d157066/MaxText/layers/attentions.py#L562
    """
    B, T, N, H = q.shape
    assert H >= 128, 'TPU flash attention reqruies head dim. to be a multiple of 128'

    # scale query
    q /= jnp.sqrt(H)

    # kernel block sizes
    # https://github.com/AI-Hypercomputer/maxtext/blob/afcdf47f8b7c1e1864fa81012a873590c5408122/MaxText/configs/base.yml#L644
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=512,
        block_kv=512,
        block_kv_compute=min(H, 256),
        block_q_dkv=512,
        block_kv_dkv=512,
        block_kv_dkv_compute=min(H, 256),
        block_q_dq=512,
        block_kv_dq=512,
    )

    mesh = jax.sharding.get_abstract_mesh()
    sharding = P('data', None, 'model', None)
    @partial(shard_map, mesh=mesh, in_specs=(sharding, sharding, sharding), out_specs=sharding, check_rep=False)
    def attention(q, k, v):
        _, _, n, _ = q.shape
        causal_mask = splash_attention_mask.CausalMask(shape=(T, T))
        multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(causal_mask,) * n)
        splash_kernel = splash_attention_kernel.make_splash_mha(mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
        out = jax.vmap(splash_kernel)(
            q.swapaxes(1, 2),
            k.swapaxes(1, 2),
            v.swapaxes(1, 2)
        ).swapaxes(1, 2) # [B, T, N, H]
        return out

    return attention(q, k, v)


class MLP(nnx.Module):
    """Multilayer perceptron."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        megatron_stds = _megatron_init_stds(c)
        up_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[0])
        }
        down_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[1])
        }
        self.up_proj = nnx.Linear(
            in_features=c.D,
            out_features=c.F,
            use_bias=False,
            dtype=c.activ_dtype,
            **up_init,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=c.F,
            out_features=c.D,
            use_bias=False,
            dtype=c.activ_dtype,
            **down_init,
            rngs=rngs,
        )
        
    def __call__(self, x, update_inputs=None, metric_prefix: str = ""): # [B, T, D]
        up_output = self.up_proj(x)
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.up_proj.kernel"] = {
                "input": x,
                "output_fro": _frobenius_norm(up_output),
            }
        h = jax.nn.gelu(up_output) # [B, T, F]
        output = self.down_proj(h) # [B, T, D]
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.down_proj.kernel"] = {
                "input": h,
                "output_fro": _frobenius_norm(output),
            }
        return output


def compute_parameter_update_metrics(old_model: TransformerDecoder, new_model: TransformerDecoder, captures):
    """Computes per-operator update metrics while holding captured inputs fixed."""
    metrics = {}

    old_embedding = old_model.token_embed_in.embedding.value
    new_embedding = new_model.token_embed_in.embedding.value
    capture = captures["token_embed_in.embedding"]
    delta_embedding = jnp.asarray(new_embedding, dtype=jnp.float32) - jnp.asarray(old_embedding, dtype=jnp.float32)
    token_ids = capture["input"]
    # Each lookup input is conceptually one-hot, so its Frobenius norm is sqrt(number of tokens).
    _add_matrix_update_metrics(
        metrics,
        "token_embed_in.embedding",
        old_embedding,
        new_embedding,
        jnp.ones(token_ids.shape, dtype=jnp.float32),
        capture["output_fro"],
        jnp.take(delta_embedding, token_ids, axis=0),
    )

    for i, (old_block, new_block) in enumerate(zip(old_model.blocks, new_model.blocks)):
        block_prefix = f"blocks.{i}"

        for norm_name in ("ln1", "ln2"):
            old_norm = getattr(old_block, norm_name)
            new_norm = getattr(new_block, norm_name)
            if getattr(old_norm, "scale", None) is not None:
                path = f"{block_prefix}.{norm_name}.scale"
                _add_scale_update_metrics(
                    metrics, path, old_norm.scale.value, new_norm.scale.value, captures[path]
                )

        old_attn = old_block.attn
        new_attn = new_block.attn
        attn_prefix = f"{block_prefix}.attn"

        path = f"{attn_prefix}.qkv_proj.kernel"
        capture = captures[path]
        old_weight = old_attn.qkv_proj.kernel.value
        new_weight = new_attn.qkv_proj.kernel.value
        delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(old_weight, dtype=jnp.float32)
        old_matrix = jnp.transpose(jnp.asarray(old_weight, dtype=jnp.float32), (2, 0, 1, 3)).reshape(
            old_weight.shape[2], -1
        )
        new_matrix = jnp.transpose(jnp.asarray(new_weight, dtype=jnp.float32), (2, 0, 1, 3)).reshape(
            new_weight.shape[2], -1
        )
        _add_matrix_update_metrics(
            metrics,
            path,
            old_weight,
            new_weight,
            capture["input"],
            capture["output_fro"],
            jnp.einsum("BTd,SNdH->SBTNH", capture["input"], delta_weight),
            old_matrix,
            new_matrix,
        )

        if old_attn.elementwise_attn_output_gate:
            path = f"{attn_prefix}.gate_proj.kernel"
            capture = captures[path]
            old_weight = old_attn.gate_proj.kernel.value
            new_weight = new_attn.gate_proj.kernel.value
            delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(old_weight, dtype=jnp.float32)
            old_matrix = jnp.transpose(jnp.asarray(old_weight, dtype=jnp.float32), (1, 0, 2)).reshape(
                old_weight.shape[1], -1
            )
            new_matrix = jnp.transpose(jnp.asarray(new_weight, dtype=jnp.float32), (1, 0, 2)).reshape(
                new_weight.shape[1], -1
            )
            _add_matrix_update_metrics(
                metrics,
                path,
                old_weight,
                new_weight,
                capture["input"],
                capture["output_fro"],
                jnp.einsum("BTd,NdH->BTNH", capture["input"], delta_weight),
                old_matrix,
                new_matrix,
            )

        for norm_name in ("query_norm", "key_norm"):
            old_norm = getattr(old_attn, norm_name)
            new_norm = getattr(new_attn, norm_name)
            if isinstance(old_norm, nnx.RMSNorm) and getattr(old_norm, "scale", None) is not None:
                path = f"{attn_prefix}.{norm_name}.scale"
                _add_scale_update_metrics(
                    metrics, path, old_norm.scale.value, new_norm.scale.value, captures[path]
                )

        path = f"{attn_prefix}.out_proj.kernel"
        capture = captures[path]
        old_weight = old_attn.out_proj.kernel.value
        new_weight = new_attn.out_proj.kernel.value
        delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(old_weight, dtype=jnp.float32)
        _add_matrix_update_metrics(
            metrics,
            path,
            old_weight,
            new_weight,
            capture["input"],
            capture["output_fro"],
            jnp.einsum("BTnh,nhD->BTD", capture["input"], delta_weight),
            jnp.asarray(old_weight, dtype=jnp.float32).reshape(-1, old_weight.shape[-1]),
            jnp.asarray(new_weight, dtype=jnp.float32).reshape(-1, new_weight.shape[-1]),
        )

        for projection_name in ("up_proj", "down_proj"):
            old_projection = getattr(old_block.mlp, projection_name)
            new_projection = getattr(new_block.mlp, projection_name)
            path = f"{block_prefix}.mlp.{projection_name}.kernel"
            capture = captures[path]
            old_weight = old_projection.kernel.value
            new_weight = new_projection.kernel.value
            delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(old_weight, dtype=jnp.float32)
            _add_matrix_update_metrics(
                metrics,
                path,
                old_weight,
                new_weight,
                capture["input"],
                capture["output_fro"],
                jnp.asarray(capture["input"], dtype=jnp.float32) @ delta_weight,
            )

    if getattr(old_model.out_ln, "scale", None) is not None:
        path = "out_ln.scale"
        _add_scale_update_metrics(
            metrics,
            path,
            old_model.out_ln.scale.value,
            new_model.out_ln.scale.value,
            captures[path],
        )

    if old_model.lm_head_oblique_learn_target_rms:
        path = "lm_head_oblique_target_rms_log"
        capture = captures[path]
        old_log_scale = jnp.asarray(old_model.lm_head_oblique_target_rms_log.value, dtype=jnp.float32)
        new_log_scale = jnp.asarray(new_model.lm_head_oblique_target_rms_log.value, dtype=jnp.float32)
        old_scale = jnp.exp(old_log_scale)
        new_scale = jnp.exp(new_log_scale)
        delta_param = new_log_scale - old_log_scale
        delta_output = jnp.asarray(capture["input"], dtype=jnp.float32) * (new_scale - old_scale)
        prefix = f"parameter_update/{path}"
        metrics[f"{prefix}/delta_param_fro"] = _frobenius_norm(delta_param)
        metrics[f"{prefix}/delta_param_rel_fro"] = _safe_ratio(
            _frobenius_norm(delta_param), _frobenius_norm(old_log_scale)
        )
        metrics[f"{prefix}/input_fro"] = _frobenius_norm(capture["input"])
        metrics[f"{prefix}/delta_output_fro"] = _frobenius_norm(delta_output)
        metrics[f"{prefix}/delta_output_rel_fro"] = _safe_ratio(
            _frobenius_norm(delta_output), capture["output_fro"]
        )
        metrics[f"{prefix}/delta_param_rel_spectral"] = _safe_ratio(
            jnp.abs(new_scale - old_scale), jnp.abs(old_scale)
        )

    path = "token_embed_out.embedding"
    capture = captures[path]
    old_embedding = old_model.token_embed_out.embedding.value
    new_embedding = new_model.token_embed_out.embedding.value
    old_matrix = jnp.asarray(old_embedding, dtype=jnp.float32).T
    new_matrix = jnp.asarray(new_embedding, dtype=jnp.float32).T
    delta_matrix = new_matrix - old_matrix
    old_output_norm = _linear_output_fro_from_grams(capture["input"], old_matrix)
    delta_output_norm = _linear_output_fro_from_grams(capture["input"], delta_matrix)
    _add_matrix_update_metrics(
        metrics,
        path,
        old_embedding,
        new_embedding,
        capture["input"],
        old_output_norm,
        old_matrix=old_matrix,
        new_matrix=new_matrix,
        delta_output_norm=delta_output_norm,
    )

    return jax.tree.map(jax.lax.stop_gradient, metrics)


def create_sharded_model(c: DictConfig, key):
    seed = int(jax.random.randint(key, [1], 0, 1_000_000)[0])

    @nnx.jit
    def initialize_sharded_model():
        rngs = nnx.Rngs(seed)
        model = TransformerDecoder(c, rngs=rngs) # unsharded at this moment
        state = nnx.state(model) # the model's state, a pure pytree

        def add_sharding(path, v):
            key = jax.tree_util.keystr(path, simple=True, separator='/')
            pspec = None
            if key == 'token_embed_in/embedding': pspec = P('data', 'model')
            if 'up_proj' in key: pspec = P('data', 'model')
            if 'down_proj' in key: pspec = P('model', 'data')
            if 'qkv_proj' in key: pspec = P(None, 'model', 'data', None)
            if 'gate_proj' in key: pspec = P('model', 'data', None)
            if 'out_proj' in key: pspec = P('model', None, 'data')
            if key == 'token_embed_out/embedding': pspec = P('model', 'data')
            if pspec is None:
                return v
            return jax.lax.with_sharding_constraint(v, pspec)
        state = jax.tree.map_with_path(add_sharding, state)
        nnx.update(model, state) # the model is sharded now
        
        return model

    model = initialize_sharded_model()

    return model
