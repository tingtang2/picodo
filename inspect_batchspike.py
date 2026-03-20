# inspect_spike.py
import numpy as np

spike = np.load("spikes/spike_step_19391.npy", allow_pickle=True).item()

print("Step:", spike["step"])
print("Loss:", spike["loss"])
print("Baseline μ:", spike["baseline_mu"])
print("Baseline σ:", spike["baseline_sigma"])

batch = spike["batch"]
print(batch.keys())

