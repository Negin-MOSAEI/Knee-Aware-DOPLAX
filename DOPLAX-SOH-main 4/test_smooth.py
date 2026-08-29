import numpy as np
from post_proc import postprocess_capacity

pred_soh = np.array([0.90360737, 0.9020324, 0.9005229, 0.89907885, 0.8976929])
print("Original:", pred_soh)
smoothed = postprocess_capacity(pred_soh, jump_thresh=0.01, hampel_window=3, savgol_window=3, polyorder=1, monotonic=True)
print("Smoothed:", smoothed)
