import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

# Usage: python plot_spectrogram.py <path_to_h5_file>
h5_path = "./realtime_data/FRBs/FRB20181209A_22154974_beamformed.h5"

# Load
with h5py.File(h5_path, 'r') as f:
    # Find power dataset
    for key in ['tiedbeam_power', 'intensity', 'data', 'beamformed_data', 'beam_power']:
        if key in f:
            power_key = key
            break

    ds = f[power_key]
    if ds.ndim == 3:  # (freq, beam, time)
        power = ds[:, 0, :].T  # → (time, freq)
    else:
        power = ds[()].T if ds.shape[1] > ds.shape[0] else ds[()]

    # Grab DM if available
    dm = None
    if 'tiedbeam_power' in f and 'DM_coherent' in f['tiedbeam_power'].attrs:
        dm = f['tiedbeam_power'].attrs['DM_coherent']
    print(f"Shape: {power.shape}, DM: {dm}")

power = power.astype(np.float32)

# Downsample time for visibility
factor = 32
n_trim = (power.shape[0] // factor) * factor
power_ds = np.nanmean(power[:n_trim].reshape(n_trim // factor, factor, -1), axis=1)


# Per-channel normalization
mean = np.nanmean(power_ds, axis=0, keepdims=True)
std = np.nanstd(power_ds, axis=0, keepdims=True)
std[std == 0] = 1
normed = (power_ds - mean) / std

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(normed.T, aspect='auto', origin='lower',
          cmap='inferno', vmin=-1, vmax=5,
          extent=[0, normed.shape[0], 400, 800])
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Frequency (MHz)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/spectrogram.svg', bbox_inches='tight')
plt.show()
print("Saved figures/spectrogram.svg")
