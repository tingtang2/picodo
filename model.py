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


class TransformerDecoder(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        self.token_embed_in = nnx.Embed(num_embeddings=c.V, features=c.D, dtype=c.activ_dtype, rngs=rngs)
        self.token_embed_out = nnx.Embed(num_embeddings=c.V, features=c.D, dtype=c.activ_dtype, rngs=rngs)
        self.blocks = nnx.List(TransformerBlock(c, rngs) for _ in range(c.L))
        self.out_ln = nnx.RMSNorm(c.D, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        
    def __call__(self, x): # [B, S]

        # token embedding
        h = self.token_embed_in(x) # [B, T, D]

        # transformer blocks
        for block in self.blocks:
            h = block(h)

        # project back to vocabulary
        h = self.out_ln(h)
        logits = self.token_embed_out.attend(h) # [B, T, V]
        return logits


class TransformerBlock(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        self.ln1 = nnx.RMSNorm(c.D, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        self.ln2 = nnx.RMSNorm(c.D, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        self.attn = MultiHeadAttention(c, rngs)
        self.mlp = MLP(c, rngs)
        
    def __call__(self, x): # [B, T, D]
        x = x + self.attn(self.ln1(x)) # attention block
        return x + self.mlp(self.ln2(x)) # MLP block


class MultiHeadAttention(nnx.Module):
    """Causal attention layer."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        self.qkv_proj = nnx.Einsum('BTd,SNdH->SBTNH', (3, c.N, c.D, c.H), dtype=c.activ_dtype, rngs=rngs)
        self.out_proj = nnx.Einsum('BTnh,nhD->BTD', (c.N, c.H, c.D), dtype=c.activ_dtype, rngs=rngs)
        self.query_norm = nnx.RMSNorm(c.H, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        self.key_norm = nnx.RMSNorm(c.H, use_scale=False, dtype=c.activ_dtype, rngs=rngs)
        if c.use_flash_attn and jax.devices()[0].platform == 'tpu' and (c.H % 128 != 0):
            warnings.warn('cannot use flash attention because `model.H` is not a multiple of 128.')
        c.use_flash_attn &= jax.devices()[0].platform == 'tpu'
        c.use_flash_attn &= (c.H % 128 == 0)
        self.attention = partial(tpu_causal_flash_attention) if c.use_flash_attn else partial(jax.nn.dot_product_attention, is_causal=True)

    def __call__(self, x): # [B, T, D]
        B, T, D = x.shape

        # input projection
        q, k, v = self.qkv_proj(x) # [B, T, N, H]

        # qk-norm
        q = self.query_norm(q)
        k = self.key_norm(k)

        # position embedding
        position = jnp.arange(T)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

        # attention
        out = self.attention(q, k, v) # [B, T, N, H]

        # output projection followed by contraction back to original dims
        out = self.out_proj(out) # [B, T, D]
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
        
    def __call__(self, x): # [B, T, D]
        h = jax.nn.gelu(self.up_proj(x)) # [B, T, F]
        return self.down_proj(h) # [B, T, D]


def create_sharded_model(c: DictConfig, key):
    seed = int(jax.random.randint(key, [1], 0, 1_000_000)[0])

    @nnx.jit
    def initialize_sharded_model():
        rngs = nnx.Rngs(seed)
        model = TransformerDecoder(c, rngs=rngs) # unsharded at this moment
        state = nnx.state(model) # the model's state, a pure pytree

        def add_sharding(path, v):
            key = jax.tree_util.keystr(path, simple=True, separator='/')
            if 'token_embed_in' in key: pspec = P('data', 'model')
            if 'up_proj' in key: pspec = P('data', 'model')
            if 'down_proj' in key: pspec = P('model', 'data')
            if 'qkv_proj' in key: pspec = P(None, 'model', 'data', None)
            if 'out_proj' in key: pspec = P('model', None, 'data')
            if 'token_embed_out' in key: pspec = P('model', 'data')
            return jax.lax.with_sharding_constraint(v, pspec)
        state = jax.tree.map_with_path(add_sharding, state)
        nnx.update(model, state) # the model is sharded now
        
        return model

    model = initialize_sharded_model()

    return model
