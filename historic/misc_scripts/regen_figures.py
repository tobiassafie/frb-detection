"""
Standalone regeneration of confusion_matrix.svg and detection_latency.svg
using the already-trained cnn_v1 model (no retraining).

Mirrors the logic in cnn_training.ipynb's confusion-matrix cell and
measure_detection_latency() cell, with two changes:
  1. Font sizes bumped across both figures (title/labels/ticks/legend/annotations).
  2. detection_latency.svg panel 3 ("Probability curves aligned to burst onset")
     reworked: x-axis is now % of FRB lifetime (0% = onset, normalized by each
     burst's own detected duration) instead of raw time bins, starts at 0%,
     and each curve is labeled instead of bare overlaid lines.

Data note: augmented_dataset/lstm_dataset_v6.npz was deleted from the working
tree (see git status); "augmented_dataset/lstm_dataset_v6 - Copy.npz" has
identical shapes/class-balance and is used in its place.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

# --- Font bump, shared by both figures ---
TITLE_FS = 18
LABEL_FS = 15
TICK_FS = 13
LEGEND_FS = 12
ANNOT_FS = 20

DATA_PATH = "./augmented_dataset/lstm_dataset_v6 - Copy.npz"
MODEL_PATH = "./models/cnn_v1/best_model.keras"
FIGURES_DIR = "./figures"

data = np.load(DATA_PATH)
X_test, y_test = data["X_test"], data["y_test"]
X_test = X_test[..., np.newaxis]

model = keras.models.load_model(MODEL_PATH)

y_pred_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

# ============================================================
# 1. Confusion matrix — same data/computation, bigger fonts only
# ============================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Noise', 'FRB'], fontsize=TICK_FS)
ax.set_yticklabels(['Noise', 'FRB'], fontsize=TICK_FS)
ax.set_xlabel('Predicted', fontsize=LABEL_FS)
ax.set_ylabel('True', fontsize=LABEL_FS)
ax.set_title('Confusion Matrix', fontsize=TITLE_FS)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=ANNOT_FS,
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/confusion_matrix.svg')
plt.close(fig)
print(f"Saved {FIGURES_DIR}/confusion_matrix.svg")
print(cm)

# ============================================================
# 2. Detection-latency 3-panel figure
# ============================================================
EFFECTIVE_TIME_RES = 2.56e-6 * 256  # seconds per bin (~0.655 ms)


def find_burst_onset(patch, threshold_sigma=2.5, freq_frac=0.1):
    n_time, n_freq = patch.shape
    n_top = max(1, int(n_freq * freq_frac))
    top_band = patch[:, -n_top:]
    profile = np.mean(top_band, axis=1)
    med = np.median(profile)
    mad = np.median(np.abs(profile - med))
    sigma = mad * 1.4826
    if sigma < 1e-10:
        sigma = np.std(profile)
        if sigma < 1e-10:
            return None
    above = np.where(profile > med + threshold_sigma * sigma)[0]
    if len(above) == 0:
        return None
    return above[0]


def find_burst_duration(patch, onset, threshold_sigma=2.5, freq_frac=0.1):
    """Total burst duration in bins, derived from the same top-band
    significance profile used by find_burst_onset: span from onset to the
    last bin still above threshold. No per-burst duration is stored in the
    dataset, so this is recomputed directly from the patch (raw sim) data."""
    n_time, n_freq = patch.shape
    n_top = max(1, int(n_freq * freq_frac))
    top_band = patch[:, -n_top:]
    profile = np.mean(top_band, axis=1)
    med = np.median(profile)
    mad = np.median(np.abs(profile - med))
    sigma = mad * 1.4826
    if sigma < 1e-10:
        sigma = np.std(profile)
        if sigma < 1e-10:
            return 1
    above = np.where(profile > med + threshold_sigma * sigma)[0]
    above_after_onset = above[above >= onset]
    if len(above_after_onset) == 0:
        return 1
    return max(int(above_after_onset.max()) - int(onset) + 1, 1)


def measure_detection_latency(model, patch, n_steps=256, fill_value=None,
                               trigger_thresh=0.5, converge_thresh=0.95):
    n_time = patch.shape[0]
    step_size = max(1, n_time // n_steps)

    probs = []
    bins = []

    for t in range(step_size, n_time + 1, step_size):
        masked = np.copy(patch)
        if fill_value is None:
            masked[t:, :] = masked[:t, :].mean()
        elif fill_value == 'noise':
            revealed = masked[:t, :]
            masked[t:, :] = np.random.normal(
                revealed.mean(), max(revealed.std(), 1e-6), masked[t:, :].shape
            )
        else:
            masked[t:, :] = fill_value

        pred = model.predict(masked[np.newaxis, ..., np.newaxis], verbose=0)[0, 0]
        probs.append(pred)
        bins.append(t)

    probs = np.array(probs)
    bins = np.array(bins)

    burst_onset = find_burst_onset(patch)

    trigger_bin = None
    converge_bin = None

    trigger_idxs = np.where(probs >= trigger_thresh)[0]
    if len(trigger_idxs) > 0:
        trigger_bin = bins[trigger_idxs[0]]

    converge_idxs = np.where(probs >= converge_thresh)[0]
    if len(converge_idxs) > 0:
        converge_bin = bins[converge_idxs[0]]

    return burst_onset, trigger_bin, converge_bin, bins, probs


rng = np.random.default_rng(42)

frb_test_idxs = np.where(y_test == 1)[0]

results = []
for idx in frb_test_idxs:
    patch = X_test[idx, :, :, 0]
    onset, trig, conv, bins, probs = measure_detection_latency(
        model, patch, n_steps=256, fill_value=None
    )
    if onset is not None and trig is not None:
        results.append({
            'idx': idx,
            'burst_onset': onset,
            'trigger_bin': trig,
            'converge_bin': conv,
            'bins_to_trigger': trig - onset,
            'bins_to_converge': conv - onset if conv is not None else None,
            'bins': bins,
            'probs': probs,
            'duration': find_burst_duration(patch, onset),
        })

print(f"Measured {len(results)} / {len(frb_test_idxs)} FRB patches "
      f"(skipped {len(frb_test_idxs) - len(results)} with no detectable burst or no trigger)")

bins_to_trigger = np.array([r['bins_to_trigger'] for r in results])
bins_to_converge = np.array([r['bins_to_converge'] for r in results
                              if r['bins_to_converge'] is not None])

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Panel 1: Histogram of bins to trigger
ax = axes[0]
ax.hist(bins_to_trigger, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(np.median(bins_to_trigger), color='red', ls='--', lw=2,
           label=f'median = {np.median(bins_to_trigger):.0f} bins')
ax.set_xlabel('Time bins after burst onset', fontsize=LABEL_FS)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.set_title('Bins to P(FRB) > 0.5', fontsize=TITLE_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.grid(True, alpha=0.3)

# Panel 2: Histogram of bins to convergence
ax = axes[1]
ax.hist(bins_to_converge, bins=30, edgecolor='black', alpha=0.7, color='darkorange')
ax.axvline(np.median(bins_to_converge), color='red', ls='--', lw=2,
           label=f'median = {np.median(bins_to_converge):.0f} bins')
ax.set_xlabel('Time bins after burst onset', fontsize=LABEL_FS)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.set_title('Bins to P(FRB) > 0.95', fontsize=TITLE_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.grid(True, alpha=0.3)

# Panel 3: Probability curves aligned to burst onset, normalized to % of FRB lifetime
ax = axes[2]
show_idxs = rng.choice(len(results), min(10, len(results)), replace=False)
for i in show_idxs:
    r = results[i]
    shifted = r['bins'] - r['burst_onset']
    mask = shifted >= 0
    pct_lifetime = shifted[mask] / r['duration'] * 100
    ax.plot(pct_lifetime, r['probs'][mask], alpha=0.75, linewidth=1.4,
            label=f"FRB #{r['idx']}")

ax.axhline(0.5, color='r', ls='--', alpha=0.5, label='P = 0.5')
ax.axhline(0.95, color='orange', ls='--', alpha=0.5, label='P = 0.95')
ax.set_xlim(left=0)
ax.set_xlabel('% of FRB lifetime', fontsize=LABEL_FS)
ax.set_ylabel('P(FRB)', fontsize=LABEL_FS)
ax.set_title('Probability curves aligned to burst onset', fontsize=TITLE_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS, ncol=2, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/detection_latency.svg')
plt.close(fig)
print(f"Saved {FIGURES_DIR}/detection_latency.svg")
