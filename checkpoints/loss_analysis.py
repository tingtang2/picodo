from transformers import GPT2Tokenizer
import numpy as np

step = 2000
tok = GPT2Tokenizer.from_pretrained("gpt2")

eval_losses = np.load(f"eval_raw_losses_step_{step}.npy")
val_tokens = np.load("val_dataset.npy", allow_pickle=True)

print(eval_losses.shape)  # (2441, 8, 1024)
print(val_tokens.shape)   # (2441, 8, 1024)


def analyze_batch_seq_slice(b,s, top_k=20):
    losses = eval_losses[b,s]
    tokens = val_tokens[b,s]

    top_idx = losses.argsort()[-top_k:][::-1] #top 20 highest loss tokens in this (b,s) slice
    print(f"\nTop 20 hardest tokens in batch {b}, sequence {s}:\n")

    for i in top_idx:
        token_id = int(tokens[i])
        decoded = tok.decode([token_id])    # must pass list to decode
        print(f"Token ID {token_id:5d} → '{decoded}' → loss {losses[i]:.3f}")

def analyze_entire_step(top_k = 20):
    print(f"Aggregating across entire dataset at step {step}")
    eval_flat = eval_losses.flatten()
    toks_flat = val_tokens.flatten()

    top_idx = eval_flat.argsort()[-top_k:][::-1] #top 20 highest loss toks in ENTIRE checkpoint 
    top_losses = eval_flat[top_idx]
    top_tokens = toks_flat[top_idx]
    print("\nTop 20 hardest tokens:\n")
    for tid, loss in zip(top_tokens, top_losses):
        print(f"Token ID {int(tid):5d} → loss {loss:.3f} → '{tok.decode([int(tid)])}'")


for i in range(3):
    for j in range(3):
        analyze_batch_seq_slice(i,j)
analyze_entire_step()

