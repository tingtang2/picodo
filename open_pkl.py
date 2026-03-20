
import pickle
import tiktoken

# Initialize GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

'''
# Load the spike batch file
with open('spike_batch_1115.pkl', 'rb') as f:
    data = pickle.load(f)

print("Data shape:", data.shape)

# Top spike positions from your log (step 1115)
spike_positions = [
    (10, 598),    # Rank 1
    (154, 910),   # Rank 2
    (235, 611),   # Rank 3
    (120, 860),   # Rank 4
    (10, 787),    # Rank 5
    (40, 714),    # Rank 6
    (42, 934),    # Rank 7
    (156, 594),   # Rank 8
    (192, 963),   # Rank 9
    (175, 544),   # Rank 10
]



# How many tokens before/after to show
context_window = 5

# Decode the tokens and show context
for rank, (b_idx, t_pos) in enumerate(spike_positions, start=1):
    token_id = int(data[b_idx, t_pos])
    word = tokenizer.decode([token_id])

    # Get context slice, safely handling edges
    start = max(0, t_pos - context_window)
    end = min(data.shape[1], t_pos + context_window + 1)
    context_ids = data[b_idx, start:end].astype(int).tolist()
    context_text = tokenizer.decode(context_ids)

    print(f"Rank {rank}: Batch {b_idx}, Position {t_pos}, Token ID {token_id}, Word: '{word}'")
    print(f"  Context ({start}-{end-1}): '{context_text}'\n")

'''
import pickle
import matplotlib.pyplot as plt

# Load the spike batch
with open("tmp_spike_batch_10.pkl", "rb") as f:
        data = pickle.load(f)

        batch = data["batch"]
        losses = data["losses"]

        # Choose the batch index you want to plot
        batch_idx = 10

        # Plot token losses for this batch
        plt.figure(figsize=(12, 4))
        plt.plot(losses[batch_idx])
        plt.xlabel("Token position")
        plt.ylabel("Loss")
        plt.title(f"Token losses for batch {batch_idx}")
        plt.grid(True)

        # Save as PNG so you can SCP it to your local machine
        plt.savefig(f"spike_batch_1115_batch{batch_idx}.png")
        print(f"Figure saved as spike_batch_1115_batch{batch_idx}.png")

