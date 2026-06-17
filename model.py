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


class TransformerDecoder(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        lm_head_dtype = getattr(c, "lm_head_dtype", c.activ_dtype)
        self.final_hidden_mean_centering = bool(getattr(c, "final_hidden_mean_centering", False))
        self.final_hidden_mean_centering_coeff = float(getattr(c, "alpha", 1.0))
        self.lm_head_oblique_learn_target_rms = bool(getattr(c, "lm_head_oblique_learn_target_rms", False))
        self.lm_head_oblique_initial_target_rms_from_random_init = bool(
            getattr(c, "lm_head_oblique_initial_target_rms_from_random_init", False)
        )
        self.lm_head_oblique_optimizer_type = str(getattr(c, "lm_head_oblique_optimizer_type", "row_oblique")).lower()
        self.token_embed_in = nnx.Embed(num_embeddings=c.V, features=c.D, dtype=c.activ_dtype, rngs=rngs)
        self.token_embed_out = nnx.Embed(num_embeddings=c.V, features=c.D, dtype=lm_head_dtype, rngs=rngs)
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
        self.out_ln = nnx.RMSNorm(c.D, use_scale=False, dtype=lm_head_dtype, rngs=rngs)
        
    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        return_analysis_tensors: bool = False,
        collect_activation_tensors: bool = True,
        collect_attention_tensors: bool = True,
        analysis_layer_indices: tuple[int, ...] | None = None,
    ): # [B, S]

        qkv_outputs = {}
        analysis_outputs = {}
        # token embedding
        h = self.token_embed_in(x, out_sharding=P('data', None, None)) # [B, T, D]

        # transformer blocks
        for i, block in enumerate(self.blocks):
            should_collect_layer_analysis = (
                return_analysis_tensors
                and (analysis_layer_indices is None or i in analysis_layer_indices)
            )
            if return_qkv and should_collect_layer_analysis:
                h, qkv, analysis = block(
                    h,
                    attention_mask=attention_mask,
                    return_qkv=True,
                    return_analysis_tensors=True,
                    collect_activation_tensors=collect_activation_tensors,
                    collect_attention_tensors=collect_attention_tensors,
                )
                qkv_outputs[i] = qkv
                analysis_outputs[i] = analysis
            elif return_qkv:
                h, qkv = block(h, attention_mask=attention_mask, return_qkv=True)
                qkv_outputs[i] = qkv
            elif should_collect_layer_analysis:
                h, analysis = block(
                    h,
                    attention_mask=attention_mask,
                    return_analysis_tensors=True,
                    collect_activation_tensors=collect_activation_tensors,
                    collect_attention_tensors=collect_attention_tensors,
                )
                analysis_outputs[i] = analysis
            else:
                h = block(h, attention_mask=attention_mask)

        # project back to vocabulary
        h = self.out_ln(h)
        if self.final_hidden_mean_centering:
            h = h - self.final_hidden_mean_centering_coeff * jnp.mean(h, axis=-1, keepdims=True)
        if self.lm_head_oblique_learn_target_rms:
            target_rms = jnp.exp(jnp.asarray(self.lm_head_oblique_target_rms_log.value, dtype=h.dtype))
            h = h * target_rms
        logits = self.token_embed_out.attend(h, out_sharding=P('data', None, 'model')) # [B, T, V]

        if return_qkv and return_analysis_tensors:
            return logits, qkv_outputs, analysis_outputs
        if return_qkv:
            return logits, qkv_outputs
        if return_analysis_tensors:
            return logits, analysis_outputs

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
        self.ln1 = nnx.RMSNorm(c.D, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        self.ln2 = nnx.RMSNorm(c.D, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        
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
        return_analysis_tensors: bool = False,
        collect_activation_tensors: bool = True,
        collect_attention_tensors: bool = True,
    ): # [B, T, D]
        attn_analysis = {}
        attn_input = self.ln1(x)
        if return_qkv and return_analysis_tensors:
            attn_out, qkv, attn_analysis = self.attn(
                attn_input,
                attention_mask=attention_mask,
                return_qkv=True,
                return_analysis_tensors=True,
                collect_activation_tensors=collect_activation_tensors,
                collect_attention_tensors=collect_attention_tensors,
            )
            x = x + attn_out
        elif return_qkv:
            attn_out, qkv = self.attn(attn_input, attention_mask=attention_mask, return_qkv=True)
            x = x + attn_out
        elif return_analysis_tensors:
            attn_out, attn_analysis = self.attn(
                attn_input,
                attention_mask=attention_mask,
                return_analysis_tensors=True,
                collect_activation_tensors=collect_activation_tensors,
                collect_attention_tensors=collect_attention_tensors,
            )
            x = x + attn_out
        else:
            x = x + self.attn(attn_input, attention_mask=attention_mask)

        mlp_input = self.ln2(x)
        if return_analysis_tensors and collect_activation_tensors:
            mlp_out, mlp_hidden = self.mlp(mlp_input, return_hidden=True)
        else:
            mlp_out = self.mlp(mlp_input)
            mlp_hidden = None
        x = x + mlp_out # MLP block

        if return_analysis_tensors:
            analysis = dict(attn_analysis)
            if collect_activation_tensors:
                analysis.update({
                    'attn_input': attn_input,
                    'mlp_input': mlp_input,
                    'mlp_hidden': mlp_hidden,
                })
            if return_qkv:
                return x, qkv, analysis
            return x, analysis

        if return_qkv:
            return x, qkv

        return x


class MultiHeadAttention(nnx.Module):
    """Causal attention layer."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs, use_qk_norm: bool = None):
        self.qkv_proj = nnx.Einsum('BTd,SNdH->SBTNH', (3, c.N, c.D, c.H), dtype=c.activ_dtype, rngs=rngs)
        self.out_proj = nnx.Einsum('BTnh,nhD->BTD', (c.N, c.H, c.D), dtype=c.activ_dtype, rngs=rngs)
        self.elementwise_attn_output_gate = bool(getattr(c, "elementwise_attn_output_gate", False))
        if self.elementwise_attn_output_gate:
            self.gate_proj = nnx.Einsum('BTd,NdH->BTNH', (c.N, c.D, c.H), dtype=c.activ_dtype, rngs=rngs)
        
        if use_qk_norm is None:
            use_qk_norm = c.use_qk_norm
        
        self.query_norm = nnx.RMSNorm(c.H, use_scale=False, dtype=c.activ_dtype, rngs=rngs) if use_qk_norm else nnx.identity
        self.key_norm = nnx.RMSNorm(c.H, use_scale=False, dtype=c.activ_dtype, rngs=rngs) if use_qk_norm else nnx.identity
        if c.use_flash_attn and jax.devices()[0].platform == 'tpu' and (c.H % 128 != 0):
            warnings.warn('cannot use flash attention because `model.H` is not a multiple of 128.')
        c.use_flash_attn &= jax.devices()[0].platform == 'tpu'
        c.use_flash_attn &= (c.H % 128 == 0)
        self.attention = partial(tpu_causal_flash_attention) if c.use_flash_attn else partial(jax.nn.dot_product_attention, is_causal=False)
        self.use_flash_attn = c.use_flash_attn

    def _build_attention_mask(self, B: int, T: int, attention_mask: jax.Array | None):
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_)).reshape(1, 1, T, T)
        if attention_mask is None:
            return causal_mask
        padding_mask = attention_mask.astype(jnp.bool_).reshape(B, 1, 1, T)
        return jnp.logical_and(causal_mask, padding_mask)

    def _analysis_attention_probs(self, q, k, attention_mask: jax.Array | None):
        B, T, _, H = q.shape
        q_f32 = jnp.asarray(q, dtype=jnp.float32) / jnp.sqrt(jnp.asarray(H, dtype=jnp.float32))
        k_f32 = jnp.asarray(k, dtype=jnp.float32)
        # Keep the layout [B, N, T, T] so it matches the causal/padding mask.
        scores = jnp.einsum('btnh,bsnh->bnts', q_f32, k_f32)
        mask = self._build_attention_mask(B, T, attention_mask)
        neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
        scores = jnp.where(mask, scores, neg_inf)
        probs = jax.nn.softmax(scores, axis=-1)
        valid_rows = jnp.any(mask, axis=-1, keepdims=True)
        return jnp.where(valid_rows, probs, 0.0)

    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        return_analysis_tensors: bool = False,
        collect_activation_tensors: bool = True,
        collect_attention_tensors: bool = True,
    ): # [B, T, D]
        B, T, D = x.shape

        # input projection
        q, k, v = self.qkv_proj(x, out_sharding=P(None, 'data', None, 'model', None)) # [B, T, N, H]
        gate_score = (
            self.gate_proj(x, out_sharding=P('data', None, 'model', None))
            if self.elementwise_attn_output_gate
            else None
        )
        if return_qkv:
            raw_qkv = (q, k, v)

        # qk-norm
        q = self.query_norm(q)
        k = self.key_norm(k)

        # position embedding
        position = jnp.arange(T)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

        # attention
        if self.use_flash_attn:
            out = self.attention(q, k, v) # [B, T, N, H]
        else:
            combined_mask = self._build_attention_mask(B, T, attention_mask)
            out = self.attention(q, k, v, mask=combined_mask) # [B, T, N, H]

        if self.elementwise_attn_output_gate:
            out = out * jax.nn.sigmoid(gate_score)

        # output projection followed by contraction back to original dims
        out = self.out_proj(out, out_sharding=P('data', None, None)) # [B, T, D]
        if return_analysis_tensors:
            analysis = {}
            if collect_attention_tensors:
                analysis['attention_probs'] = self._analysis_attention_probs(q, k, attention_mask)
            if return_qkv:
                return out, raw_qkv, analysis
            return out, analysis
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
        self.up_proj = nnx.Linear(in_features=c.D, out_features=c.F, use_bias=False, dtype=c.activ_dtype, rngs=rngs)
        self.down_proj = nnx.Linear(in_features=c.F, out_features=c.D, use_bias=False, dtype=c.activ_dtype, rngs=rngs)
        
    def __call__(self, x, return_hidden: bool = False): # [B, T, D]
        h = jax.nn.gelu(self.up_proj(x, out_sharding=P('data', None, 'model'))) # [B, T, F]
        out = self.down_proj(h, out_sharding=P('data', None, None)) # [B, T, D]
        if return_hidden:
            return out, h
        return out


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
            if key == 'token_embed_in/embedding/value': pspec = P('model', None)
            if 'up_proj' in key: pspec = P(None, 'model')
            if 'down_proj' in key: pspec = P('model', None)
            if 'qkv_proj' in key: pspec = P(None, 'model', None, None)
            if 'gate_proj' in key: pspec = P('model', None, None)
            if 'out_proj' in key: pspec = P('model', None, None)
            if key == 'token_embed_out/embedding/value': pspec = P('model', None)
            if pspec is None:
                return v
            # In explicit-sharding mode, with_sharding_constraint is only an
            # assertion in newer JAX releases; use an actual reshard here.
            return jax.reshard(v, pspec)
        state = jax.tree.map_with_path(add_sharding, state)
        nnx.update(model, state) # the model is sharded now
        
        return model

    model = initialize_sharded_model()

    return model
