
import numpy as np

logits_1112 = np.load("logits_step_12_teacher_force_checkpoint_1112.npy")
logits_1116 = np.load("logits_step_12_teacher_force_checkpoint_1116.npy")
logits_1118 = np.load("logits_step_12_teacher_force_checkpoint_1118.npy")

top_1112 = logits_1112.argsort()[::-1][:10]
top_1116 = logits_1116.argsort()[::-1][:10]
top_1118 = logits_1118.argsort()[::-1][:10]
print("Top tokens 1112:", top_1112)
print("Top tokens 1116:", top_1116)
print("Top tokens 1118:", top_1118)

import matplotlib.pyplot as plt

plt.scatter(logits_1112.flatten(), logits_1116.flatten(), alpha=0.3)
plt.xlabel("1112 logits")
plt.ylabel("1116 logits")
plt.title("Step 12 logits")
plt.show()

