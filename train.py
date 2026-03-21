import math
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial
from flax import nnx
import optax
from tqdm.auto import tqdm
from omegaconf.dictconfig import DictConfig
import data, utils
import model as model_lib
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy 
from etils import epath
from orbax.checkpoint._src.path import gcs_utils, step as step_lib
import os
import sys

import tiktoken

class _StandardNameFormatHNS(step_lib._StandardNameFormat):
    """Fixes GCS HNS listing when step_prefix is None."""

    def _glob_step_paths(self, base_path: epath.PathLike) -> list[epath.Path]:
        base_path = epath.Path(base_path)
        if gcs_utils.is_hierarchical_namespace_enabled(base_path):
            bucket_name, path_prefix = gcs_utils.parse_gcs_path(base_path)
            bucket = gcs_utils.get_bucket(bucket_name)
            result = bucket.list_blobs(
                prefix=path_prefix,
                delimiter='/',
                include_folders_as_prefixes=True,
            )
            for _ in result.pages:
                pass
            step_prefix = self.step_prefix or ''
            return [
                epath.Path(f'gs://{bucket_name}/{folder}')
                for folder in result.prefixes
                if folder.startswith(os.path.join(path_prefix, step_prefix))
            ]
        return list(
            epath.Path(base_path).glob(
                f'{step_lib.step_prefix_with_underscore(self.step_prefix)}*'
            )
        )

def compute_logit_distribution_stats(logits_bf16):
    # logits_fp32: [B, T-1, V]
    flat = logits_bf16.reshape(-1, logits_bf16.shape[-1])  # [N, V]
    print("flat is ", flat)

    stats = {
        # scale / imbalance
        "logit_mean": jnp.mean(flat),
        "logit_std": jnp.std(flat),
        "logit_rms": jnp.sqrt(jnp.mean(flat ** 2)),
        "logit_max": jnp.max(flat),
        "logit_min": jnp.min(flat),
        # shape (imbalance proxy)
        "logit_abs_mean": jnp.mean(jnp.abs(flat)),
        "logit_abs_max": jnp.max(jnp.abs(flat)),
    }

    probs = jax.nn.softmax(flat, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1)
    stats["softmax_entropy_mean"] = jnp.mean(entropy)

    '''
    # Precision diagnostics
    if logits_bf16 is not None:
        diff = logits_fp32 - logits_bf16.astype(jnp.float32)
        stats.update({
            "bf16_abs_err_mean": jnp.mean(jnp.abs(diff)),
            "bf16_abs_err_max": jnp.max(jnp.abs(diff)),
            "bf16_rel_err_rms": jnp.sqrt(
                jnp.mean((diff / (jnp.abs(logits_fp32) + 1e-6)) ** 2)
            ),
        })
    '''
    
    '''
    jax.debug.print(
        "logit_mean={m}, logit_std={s}, abs_max={a}",
        m=stats["logit_mean"],
        s=stats["logit_std"],
        a=stats["logit_abs_max"],
    )
    '''

    
    return stats

'''
@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits, qkv_dict = model(x, return_qkv=True) # [B, T, V]

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
    qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    
    return losses.mean(), (losses, qkv_stats)
'''

#@partial(jax.jit, static_argnames=('model_graphdef'))
@partial(jax.jit, static_argnames=('model_graphdef', 'tmp'))
def loss_fn(model_state, model_graphdef, x, tmp = False, pos = None):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits_bf16, qkv_dict = model(x, return_qkv=True)
    logits_fp32 = logits_bf16.astype(jnp.float32)

    #change precision here (either bf16 or fp32)
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits_fp32, y
    )
    losses = losses.at[:, -1].set(0)

    # Drop last token
    logits_fp32 = logits_fp32[:, :-1, :]
    #logits_bf16 = logits_bf16[:, :-1, :]

    logit_stats = compute_logit_distribution_stats(
        logits_fp32,
        #logits_bf16
    )

    if tmp:
        # We look at 922 because it predicts the target at 923 (" against")
        # We look at 923 because it predicts the target at 924 (" end" or "End")
        for spike_pos in [pos-3, pos-2, pos-1, pos]: 
            l_pos = logits_fp32[0, spike_pos]
                    
            # Calculate everything first
            mean = jnp.mean(l_pos)
            std = jnp.std(l_pos)
            norm = jnp.linalg.norm(l_pos)
            minimum = jnp.min(l_pos)
            maximum = jnp.max(l_pos)
            q1, med, q3 = jnp.percentile(l_pos, jnp.array([25, 50, 75]))
            vals, idxs = jax.lax.top_k(l_pos, k=5)
            target_id = y[0, spike_pos]

            # Build one giant format string so the output stays atomic
            report = (
                "\n--- POSITION {p} ANALYSIS (TARGET AT {t}) ---\n"
                "Target ID: {tid}\n"
                "Top 5 IDs: {i}\n"
                "Top 5 Logits: {v}\n"
                "Min: {mi:.2f} | Q1: {q1:.2f} | Med: {med:.2f} | Q3: {q3:.2f} | Max: {ma:.2f}\n"
                "Mean: {m:.4f} | Std: {s:.4f} | Norm: {n:.4f}\n"
                "------------------------------------------"
            )
            
            jax.debug.print(report, 
                            p=spike_pos, t=spike_pos+1, tid=target_id, 
                            i=idxs, v=vals, mi=minimum, q1=q1, med=med, 
                            q3=q3, ma=maximum, m=mean, s=std, n=norm)

    qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
    qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    return losses.mean(), (losses, qkv_stats, logit_stats)

@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn_z_loss(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits, qkv_dict = model(x, return_qkv=True) # [B, T, V]
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)  

    z_loss = (z**2).mean()
    lam = 1e-4

    qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
    qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    
    return losses.mean() + lam * z_loss, (losses, qkv_stats)

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the raw losses
    #(loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_state.model, model_graphdef, batch)

    (loss, aux), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                        )(opt_state.model, model_graphdef, batch)
    
    raw_losses, qkv_stats, logit_stats = aux


    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    

    #return opt_state, loss, (raw_losses, qkv_stats, logit_stats), grads
    return opt_state, loss, (raw_losses, qkv_stats), grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step_z_loss(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(opt_state.model, model_graphdef, batch)
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step_centered(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the raw losses
    #(loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_state.model, model_graphdef, batch)
    (loss, aux), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                        )(opt_state.model, model_graphdef, batch)
    raw_losses, qkv_stats, logit_stats = aux

    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    
    #return opt_state, loss, raw_loss, grads
    return opt_state, loss, (raw_losses, qkv_stats), grads

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step_z_loss_centered(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(opt_state.model, model_graphdef, batch)
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logits_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    return logits.reshape(-1, logits.shape[-1]).astype(jnp.float32).mean(axis=0) # [B * T, V] -> [V]

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_gaps_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x).astype(jnp.float32)[:, :-1, :] # [B, T, V]
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    gaps = logits - max_logits

    mean_gaps = jnp.mean(gaps, axis=(0, 1)) # [V]
    y = jnp.roll(x, -1, axis=1)[:, :-1]
    target_logits = jnp.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
    target_gaps = target_logits - max_logits.squeeze(-1) # [B, T]

    return mean_gaps, target_gaps

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_mean_and_norm_output_logit(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    logits = logits[:, :-1].astype(jnp.float32)
    return logits.mean(), utils.get_l2_norm(logits)

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_grad_sum_stats(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32) # [B, T, V]

    def loss_from_logits(l):
        log_probs = jax.nn.log_softmax(l, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        losses = losses[:, :-1]
        return losses.mean()

    grad_logits = jax.grad(loss_from_logits)(logits) # dL/dz
    grad_sum = grad_logits.sum(axis=-1)[:, :-1] # [B, T-1]

    return {
        'logit_grad_sum_mean_abs': jnp.mean(jnp.abs(grad_sum)),
        'logit_grad_sum_max_abs': jnp.max(jnp.abs(grad_sum)),
        'logit_grad_sum_mean': jnp.mean(grad_sum),
    }


def eval_step(c, model_state, model_graphdef, dataset, inspect_spike = False, tokenizer = None):
    loss_sum = jnp.zeros([], dtype=jnp.float32)
    raw_losses = []
    total_logits = []
    logit_mean_sum = jnp.zeros([], dtype=jnp.float32)

    logit_stats_sum = None
    for i in range(len(dataset)):
        batch = dataset[i]
        if c.opt.use_z_loss:
            batch_loss, (raw_loss, _) = loss_fn_z_loss(model_state, model_graphdef, batch)
        else:
            #TODO changed this 
            batch_loss, (raw_loss, _, logit_stats) = loss_fn(model_state, model_graphdef, batch)
        loss_sum += batch_loss
        raw_losses.append(raw_loss)
        logit_mean_sum += get_mean_and_norm_output_logit(model_state, model_graphdef, batch)[0]

        if c.diagnostics.save_raw_losses:
            total_logits.append(get_logits_by_lm_head(model_state, model_graphdef, batch).astype(jnp.float32))

    mean_loss = loss_sum / len(dataset)
    mean_output_logit = logit_mean_sum / len(dataset)

    mean_logit_stats = None
    if logit_stats_sum is not None:
        #logit_stats_sum = logit_stats_sum / len(dataset)
        mean_logit_stats = jax.tree.map(
           lambda x: x / len(dataset),
           logit_stats_sum,
        )
    
    if inspect_spike: 
        all_raw_losses = jnp.concatenate(raw_losses, axis=0) # [TotalSamples, SeqLen]
        num_sequences, seq_len = all_raw_losses.shape

        flat_losses = all_raw_losses.flatten()
        top_k = 5
        idx_dec = jnp.argsort(flat_losses)[::-1][:top_k]

        top_k_seqs = []
        top_k_pos = []
        print(f"\n--- [DIAGNOSTIC] TOP {top_k} HIGHEST LOSS POSITIONS ---")
        for i, glob_idx in enumerate(idx_dec):
            loss_val = flat_losses[glob_idx]
            seq_idx, pos = glob_idx // seq_len, glob_idx % seq_len
            top_k_pos.append(pos)

            batch_idx = int(seq_idx // c.opt.batch_size)
            sample_in_batch = int(seq_idx % c.opt.batch_size)

            spiky_batch = dataset[batch_idx]
            spiky_seq = spiky_batch[sample_in_batch]
            top_k_seqs.append(spiky_seq)

            target_token_id = spiky_seq[pos + 1]
            target_token = tokenizer.decode([int(target_token_id)])

            # The context leading UP TO the prediction (includes the "trigger" token)
            trigger_token = tokenizer.decode([int(spiky_seq[pos])]) 
            context_window = tokenizer.decode(spiky_seq[max(0, pos-5):pos])

            print(f"Rank {i+1} | Loss: {loss_val:.2f} | Sequence Index: {seq_idx}")
            print(f"  Context: \"...{context_window}\"")
            print(f"  Trigger Token (at pos {pos}): '{trigger_token}'")
            print(f"  Target Token (at pos {pos+1}): '{target_token}'")


        #CALL COUNTERFACTUAL ANALYSIS HERE
        for i in range(top_k):
            counterfactual_analysis(
                c =c, model_state=model_state,
                model_graphdef=model_graphdef,
                tokenizer=tokenizer, # Make sure tokenizer is passed into eval_step or is global
                #sequence=spiky_sequence,
                #spike_pos=int(spike_pos)
                sequence= top_k_seqs[i],
                spike_pos= int(top_k_pos[i])
            )
    return mean_loss, raw_losses, total_logits, mean_output_logit, mean_logit_stats

def find_similar_tokens(model_state, tokenizer, trigger_str=" end", top_k=5):
    trigger_id = tokenizer.encode(trigger_str)[0]
    embs = model_state['token_embed_in']['embedding'].value
    # 3. Normalize for Cosine Similarity
    norm = jnp.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norm + 1e-9)
    trigger_vec = embs_norm[trigger_id]
    cos_sim = jnp.dot(embs_norm, trigger_vec)
    
    top_ids = jnp.argsort(cos_sim)[-(top_k+1):-1][::-1]
    '''
    print(f"Tokens most similar to {trigger_str!r}:")
    for tid in top_ids:
        print(f"ID: {tid:<6} | Token: {tokenizer.decode([int(tid)])!r} | Sim: {cos_sim[tid]:.4f}")
    '''
    return top_ids

def find_critical_context_length(model_state, model_graphdef, tokenizer, full_seq, spike_pos):
    pad_id = 50256 
    step_size = 50
    res = [] 

    for n in range(0, spike_pos, step_size):
        #mask everything until spike_pos - n - tail of context
        test_seq = jnp.array(full_seq)
        test_seq = test_seq.at[:spike_pos - n].set(pad_id)
        input_data = jnp.repeat( test_seq[None, :], 4, axis = 0)
        _, (losses, qkv_stats, logit_stats) = loss_fn(model_state, model_graphdef, input_data, tmp = False)
        current_loss, current_norm = losses[0, spike_pos], logit_stats['norm'][0, spike_pos]
        res.append((n, current_loss, current_norm))

    return res 

def counterfactual_analysis(c, model_state, model_graphdef, tokenizer, sequence, spike_pos, swap_token_str=" "):
    context_size = c.model.T
    pad_token_id = tokenizer.eot_token
    original_seq = jnp.array(sequence)
    
    top_sim_ids = find_similar_tokens(model_state, tokenizer, trigger_str=tokenizer.decode([ int(original_seq[spike_pos]) ] ), top_k=5)
    swap_token_id = int(top_sim_ids[0])
    swap_token_str = tokenizer.decode([swap_token_id])
    #swap_token_id = tokenizer.encode(swap_token_str)[0]

    #construct counterfactual sequence 
    cf_seq = original_seq.copy()
    start_swap = max(0, spike_pos - 0) #just swap spike_pos itself 
    #cf_seq = cf_seq.at[start_swap : spike_pos + 1].set(swap_token_id)

    print(f"\n--- Counterfactual Analysis at Position {spike_pos} ---")
    orig_context_str = tokenizer.decode(original_seq[start_swap:spike_pos + 1].tolist())
    print(f"Original Context (Indices {start_swap}-{spike_pos}): {orig_context_str!r}")
    print(f"Swapping tokens from index {start_swap} to {spike_pos} with {swap_token_str!r}")

    start_idx = max(0, spike_pos - 2)
    for i in range(start_idx, spike_pos):
        top_sim_ids = find_similar_tokens(model_state, tokenizer, trigger_str=tokenizer.decode([ int(original_seq[i]) ] ), top_k=5)
        swap_token_id = int(top_sim_ids[0])
        swap_token_str = tokenizer.decode([swap_token_id])

        cf_seq = cf_seq.at[i].set(swap_token_id)
        jax.debug.print("Position {pos}: Swapped '{orig}' -> '{new}'", pos=i, orig= tokenizer.decode([ int(original_seq[i]) ] ), new=swap_token_str)

    # We will look at the next 10 tokens to see if the model "recovers"
    window = 10
    end_pos = min(spike_pos + window, len(sequence) - 1)

    def get_full_results(seq_data):
        input_data = jnp.repeat(seq_data[None, :], 4, axis=0)
        _, (losses, qkv_stats, logit_stats) = loss_fn(model_state, model_graphdef, input_data, tmp = True, pos = spike_pos)
        return losses[0]

    # Run the model only twice total
    orig_losses = get_full_results(original_seq)
    cf_losses = get_full_results(cf_seq)

    print("-" * 65)
    print(f"{'Step':<6} | {'Target':<12} | {'Orig Loss':<10} | {'CF Loss':<10} | {'Status'}")

    for step in range(spike_pos-3, end_pos):
            target_id = int(original_seq[step+1])
            target_str = tokenizer.decode([target_id]).replace("\n", "\\n")
            
            loss_orig = float(orig_losses[step])
            loss_cf = float(cf_losses[step])
            # question: does swapping the token cuts loss significantly?
            
            print(f"{step:<6} | {target_str[:10]:<12} | {loss_orig:<10.4f} | {loss_cf:<10.4f}")
            
    print("\n" + "="*40)
    print("GLOBAL SEQUENCE COMPARISON")
    print("="*40)
    print(f"{'Metric':<15} | {'Original':<12} | {'CF Swap':<12}")
    print("-" * 45)
    print(f"{'Mean Loss':<15} | {jnp.mean(orig_losses):<12.4f} | {jnp.mean(cf_losses):<12.4f}")
    tail_orig = orig_losses[spike_pos+1:]
    tail_cf = cf_losses[spike_pos+1:]
    print(f"{'Tail Mean':<15} | {jnp.mean(tail_orig):<12.4f} | {jnp.mean(tail_cf):<12.4f}")

    res = find_critical_context_length(model_state, model_graphdef, tokenizer, original_seq, spike_pos)
    import pdb; pdb.set_trace()


    #--------
    '''
    eos_id = tokenizer.eos_id  # or whatever your EOS token ID is
    indices = jnp.where(current_seq == eos_id)[0]
    if len(indices) > 0:
        start_pos = indices[0] + 1
        # Slice from the start of the second doc to the trigger token (' end')
        # spike_pos is the index of ' end' we found earlier
        isolated_seq = current_seq[start_pos : spike_pos + 1]
        new_pos = len(isolated_seq) - 1
        
        jax.debug.print("Isolated Sequence Length: {l}", l=len(isolated_seq))
        jax.debug.print("First 3 Tokens of Isolated Doc: {t}", 
                        t=tok.decode(isolated_seq[:3].tolist()))

        loss_eos = get_full_results(isolated_seq)
        print("new loss is ", )
    '''
    #do this stress testing - how little of context feed in to get spike 


def train_and_evaluate(c: DictConfig):
    # init distributed env if using multiple vms
    jax.distributed.initialize()
    
    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print('sharding mesh:', ', '.join(f'{k}={v}' for k, v in mesh.shape.items()))

    # model
    print('initializing model...')
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count()) # round V up to enable sharding
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)

    # get num. model parameters
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')
    
    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.batch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        c.opt.b2,
        eps=c.opt.eps,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    
    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)


    # set up checkpointing
    start_step = 0
    ckpt_mngr = None
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    if c.checkpoint.turn_on:
        run_name = c.run_name if c.run_name else 'picodo_run'

        # Use GCP bucket if specified, otherwise use local workdir
        gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
        if gcp_bucket:
            # Format: gs://bucket-name/path
            if not gcp_bucket.startswith('gs://'):
                gcp_bucket = f'gs://{gcp_bucket}'
            ckpt_dir = os.path.join(gcp_bucket, run_name)
            if jax.process_index() == 0:
                print(f'Checkpoints will be saved to GCS bucket: {ckpt_dir}')
        else:
            ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)

        step_prefix = getattr(c.checkpoint, 'step_prefix', None)

        # Base checkpoint manager options
        mngr_options_kwargs = {
            'create': True,
            'preservation_policy': preservation_policy.LatestN(c.checkpoint.max_to_keep)
        }

        if gcp_bucket:
            mngr_options_kwargs['step_name_format'] = _StandardNameFormatHNS(
                step_prefix=step_prefix
            )

        # Add multihost settings if running on multiple hosts
        is_multihost = jax.process_count() > 1
        if is_multihost:
            mngr_options_kwargs['enable_async_checkpointing'] = True
            mngr_options_kwargs['multiprocessing_options'] = ocp.multiprocessing.MultiprocessingOptions(
                primary_host=0,
                active_processes=set(range(jax.process_count()))
            )
            if jax.process_index() == 0:
                print(f'Multihost checkpointing enabled with {jax.process_count()} processes')

        mngr_options = ocp.CheckpointManagerOptions(**mngr_options_kwargs)

        ckpt_mngr = ocp.CheckpointManager(
            ckpt_dir,
            options=mngr_options
        )
        
        print(f'Checking for existing checkpoints in: {ckpt_dir}')
        latest_step = c.checkpoint.start_step if c.checkpoint.start_step != None else ckpt_mngr.latest_step()

        if latest_step is not None:
            print(f'Restoring checkpoint from step {latest_step} in {ckpt_dir}...')
            
            restored_data = ckpt_mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_opt_state),
                    training_metadata=ocp.args.JsonRestore(),
                ),
            )
            opt_state = restored_data['state']
            training_metadata = restored_data.get('training_metadata', {})
            meta_step = training_metadata.get('step')
            meta_next_step = training_metadata.get('next_step')
            if meta_next_step is not None:
                start_step = meta_next_step
            elif meta_step is not None:
                if meta_step == latest_step + 1:
                    start_step = meta_step
                elif meta_step == latest_step:
                    start_step = meta_step + 1
                else:
                    print(
                        'Warning: checkpoint metadata step does not match checkpoint id '
                        f'(metadata step={meta_step}, checkpoint id={latest_step}). '
                        'Falling back to resume from checkpoint id + 1.'
                    )
                    start_step = latest_step + 1
            else:
                print(
                    'Warning: checkpoint metadata missing step/next_step; '
                    'falling back to resume from checkpoint id + 1.'
                )
                start_step = latest_step + 1
            print(f'Successfully restored checkpoint. Resuming from step {start_step}.')
        else:
            print('No checkpoint found. Starting from scratch.')

    model = nnx.merge(model_graphdef, opt_state.model)

    if c.diagnostics.save_raw_losses:
        if ckpt_dir:
            diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            utils.save_to_numpy(save_dir=diagnostics_dir, name='val_dataset', data=ds_valid)
            utils.save_to_numpy(save_dir=diagnostics_dir, name='train_dataset', data=ds_train[:c.diagnostics.end_step])

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
    log_metrics_per_step = bool(getattr(c, "log_metrics_per_step", False))

    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step
    
    mucentering = bool(getattr(c.opt, "mucentering", False))

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)

    #TODO add spike step 
    spike_step = [10, 1110, 1114, 1115, 1116, 1117, 1118] 

    for step in pbar:

        batch = ds_train[step]
        if step in spike_step:
            print("SPIKE STEP HIT", step)
            batch = ds_train[step]
            import pickle
            host_batch = jax.device_get(batch)  # move to CPU
            #with open(f"spike_batch_{step}.pkl", "wb") as f:
            #    pickle.dump(host_batch, f)
            loss_value, (losses, qkv_stats, _) = loss_fn(
                opt_state.model,
                model_graphdef,
                batch
            )

            print("Mean loss:", float(loss_value))

            import numpy as np

            # Move to host (very important — avoids JitTracer issues)
            losses_host = np.array(jax.device_get(losses))

            '''
            with open(f"tmp_spike_batch_{step}.pkl", "wb") as f:
                pickle.dump({
                    "batch": host_batch,
                    "losses": losses_host
                }, f)
            '''

            flat = losses_host.reshape(-1)

            top_k = 20
            top_idx = np.argpartition(flat, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

            print("\nTop 10 token losses:")
            T = losses_host.shape[1]

            for rank, idx in enumerate(top_idx):
                b = idx // T
                t = idx % T

                print(f"\nRank {rank+1}")
                print("  Loss:", float(flat[idx]))
                print("  Batch index:", int(b),"  Token position:", int(t),"  Target token ID:", int(batch[b, t+1]))  # +1 due to shift
             
 

        if c.log_logit_grad_stats:
            logit_grad_stats = get_logit_grad_sum_stats(opt_state.model, model_graphdef, ds_train[step])
        # training step
        if c.opt.use_z_loss:
            if mucentering:
                opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_z_loss_centered(
                    opt_state, opt_graphdef, model_graphdef, ds_train[step]
                )
            else:
                opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_z_loss(
                    opt_state, opt_graphdef, model_graphdef, ds_train[step]
                )
        else:
            '''
            #opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step(opt_state, opt_graphdef, model_graphdef, ds_train[step])
            opt_state, batch_loss, (train_raw_loss, qkv_stats, logit_stats), grads = train_step(opt_state, opt_graphdef, model_graphdef, ds_train[step])
            '''

            if mucentering:
                opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_centered(
                    opt_state, opt_graphdef, model_graphdef, ds_train[step]
                )
            else:
                opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step(
                    opt_state, opt_graphdef, model_graphdef, ds_train[step]
                )
        # if jax.process_index() == 0:
        #     min_train_loss = float(jax.device_get(jnp.min(train_raw_loss)))
        #     assert min_train_loss >= 0.0, f"negative train loss: {min_train_loss}"
        
        if c.diagnostics.save_raw_losses:
            train_logit_gaps, train_target_gaps = get_logit_gaps_by_lm_head(opt_state.model, model_graphdef, ds_train[step])

        # logging
        if log_metrics_per_step:
            metrics = {}
            metrics['train_loss'] = batch_loss
            metrics['train_med_loss'] = jnp.median(train_raw_loss)
            metrics['train_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(train_raw_loss)
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            output_logit_mean, output_logit_norm = get_mean_and_norm_output_logit(opt_state.model, model_graphdef, ds_train[step])
            metrics['train_output_logit_mean'] = output_logit_mean
            metrics['train_output_logit_norm'] = output_logit_norm
            metrics['lr'] = lr_schedule(step)
            metrics.update(utils.get_layer_grad_norms_split(grads))
            metrics.update(utils.get_layer_weight_norms_split(opt_state.model))
            metrics.update(utils.get_layer_moment_norms(opt_state))
            if c.log_logit_grad_stats:
                metrics.update(logit_grad_stats)
            # Add QKV stats to metrics
            metrics.update(qkv_stats)

            if jax.process_index() == 0:
                wandb.log(metrics, step)
                pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
        else:
            train_loss_sum += batch_loss
            train_med_loss_sum += jnp.median(train_raw_loss)
            train_lower_90th_mean_loss_sum += utils.compute_lower_90th_percentile_mean(train_raw_loss)
            train_loss_num += 1
            if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
                metrics = {}
                metrics['train_loss'] = train_loss_sum / train_loss_num
                metrics['train_med_loss'] = train_med_loss_sum / train_loss_num
                metrics['train_lower_90th_mean_loss'] = train_lower_90th_mean_loss_sum / train_loss_num
                metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
                output_logit_mean, output_logit_norm = get_mean_and_norm_output_logit(opt_state.model, model_graphdef, ds_train[step])
                metrics['train_output_logit_mean'] = output_logit_mean
                metrics['train_output_logit_norm'] = output_logit_norm
                metrics['lr'] = lr_schedule(step)
                metrics.update(utils.get_layer_grad_norms_split(grads))
                metrics.update(utils.get_layer_weight_norms_split(opt_state.model))
                metrics.update(utils.get_layer_moment_norms(opt_state))
                if c.log_logit_grad_stats:
                    metrics.update(logit_grad_stats)
                
                # Add QKV stats to metrics
                metrics.update(qkv_stats)

                if jax.process_index() == 0:
                    wandb.log(metrics, step)
                    pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
                train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
        
        # eval and checkpointing
        tok = tiktoken.get_encoding("gpt2")

        if step == 1620: 
            print("testing eval")
            val_loss, val_raw_losses, val_logits, val_mean_logit, val_stats = eval_step(
                c, opt_state.model, model_graphdef, ds_valid, inspect_spike = True, tokenizer = tok
            )
            
            print(f"Eval complete. Mean Loss: {val_loss:.4f}")
            if step == 1630: 
                break


        if step % c.eval_every_steps == 0:
            eval_loss, eval_raw_loss, eval_logits, mean_eval_output_logit, eval_logit_stats = eval_step(c, opt_state.model, model_graphdef, ds_valid)
            flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
            metrics = {}
            metrics['eval_loss'] = eval_loss
            metrics['eval_output_logit_mean'] = mean_eval_output_logit
            metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
            metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            '''
            if jax.process_index() == 0:
                for k, v in logit_stats.items(): 
                    metrics[f"logits/{k}"] = float(jax.device_get(v))

                wandb.log(metrics, step)
            '''
            
            # diagnostics
            if c.diagnostics.save_raw_losses:
                if ckpt_dir:
                    diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
                    os.makedirs(diagnostics_dir, exist_ok=True)
                    
                    # save diagnostic data
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'train_raw_losses_step_{step}.npy', data=train_raw_loss)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{step}.npy', data=eval_raw_loss)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'train_mean_logit_gaps_step_{step}.npy', data=train_logit_gaps)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'train_target_logit_gaps_step_{step}.npy', data=train_target_gaps)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_logits_step_{step}.npy', data=eval_logits)

        # Determine if we should checkpoint at this step
        should_checkpoint = False
        if c.checkpoint.turn_on:
            # Check if specific checkpoint steps are configured
            checkpoint_steps = getattr(c.checkpoint, 'checkpoint_steps', None)
            if checkpoint_steps is not None:
                # If checkpoint_steps is specified, only checkpoint at those exact steps
                should_checkpoint = step in checkpoint_steps
            else:
                # Otherwise, use the regular interval-based checkpointing
                should_checkpoint = step % c.checkpoint.checkpoint_every_steps == 0

        if should_checkpoint:
            ckpt_mngr.save(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(opt_state),
                    training_metadata=ocp.args.JsonSave({
                        'step': step,
                        'next_step': step + 1,
                    }),
                ),
                force=True,
            )
            # Wait for async checkpoint to complete in multihost setting
            if jax.process_count() > 1:
                ckpt_mngr.wait_until_finished()
    
    if num_opt_steps != len(ds_train):
        print('exiting early')
        wandb.finish()
        ckpt_mngr.close()
        sys.exit(1)

    # eval at end of training
    eval_loss, eval_raw_loss, eval_logits, mean_eval_output_logit, eval_logit_stats = eval_step(c, opt_state.model, model_graphdef, ds_valid)
    metrics = {}
    flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
    metrics['eval_loss'] = eval_loss
    metrics['eval_output_logit_mean'] = mean_eval_output_logit
    metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
    metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)

    if eval_logit_stats is not None:
        for k, v in eval_logit_stats.items():
            metrics[f"eval/{k}"] = v

    if jax.process_index() == 0:
        wandb.log(metrics)
        wandb.finish()
        if c.diagnostics.save_raw_losses:
            if ckpt_dir:
                diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
                os.makedirs(diagnostics_dir, exist_ok=True)
                
                # save diagnostic data
                utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{num_opt_steps}.npy', data=eval_raw_loss)
            
    # final checkpoint
    if c.checkpoint.turn_on and not c.diagnostics.save_raw_losses:
        final_step = max(num_opt_steps - 1, 0)
        ckpt_mngr.save(
            final_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(opt_state),
                training_metadata=ocp.args.JsonSave({
                    'step': final_step,
                    'next_step': final_step + 1,
                }),
            ),
            force=True,
        )

        ckpt_mngr.wait_until_finished()
        if jax.process_index() == 0:
            print(f'Saved final checkpoint at step {final_step} to {ckpt_mngr.directory}')
        ckpt_mngr.close()
