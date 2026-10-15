"""
Step F: generate real testbench vectors (tb_data/tb_input_features.dat +
tb_output_predictions.dat) for co-simulating the UNMODIFIED baseline
frb_pooled_hls project.

Samples come from the pooled model's actual dataset (classifier/fetch/test_data.hdf5,
see 00b_fetch_dataset_provenance.md), using the exact same preprocessing as
classifier/frb_classifier_pooled.ipynb: read `data_dm_time` (NOT data_freq_time),
resize (256,256,1) -> (64,64,1) with tf.image.resize, no further normalization.

Read-only w.r.t. all source data and w.r.t. frb_pooled_hls/. Writes only into
synth_ablation/F_cosim/tb_data_generated/ (copied into the *copy* of the project
by generate_F_cosim_project.py, never into frb_pooled_hls itself).
"""
import os
import h5py
import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TEST_H5 = os.path.join(ROOT, "classifier", "fetch", "test_data.hdf5")
MODEL_PATH = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation", "rebuilt_pooled_model.keras")
OUT_DIR = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation", "F_cosim", "tb_data_generated")

N_PER_CLASS = 4  # 4 RFI + 4 FRB = 8 samples total, chunk-aligned reads for speed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(TEST_H5, "r") as f:
        labels = f["data_labels"][:2000]  # small slice just to locate class-0/1 indices cheaply
        neg_idx = np.where(labels == False)[0][:N_PER_CLASS]  # noqa: E712
        # positives may not appear in the first 2000 (see 00b_fetch_dataset_provenance.md);
        # widen the search only for positives
        pos_idx = np.where(f["data_labels"][:10000] == True)[0][:N_PER_CLASS]  # noqa: E712

        chosen = sorted(set(neg_idx.tolist()) | set(pos_idx.tolist()))
        dm_imgs = np.stack([f["data_dm_time"][i, ..., 0] for i in chosen]).astype(np.float32)
        chosen_labels = np.array([f["data_labels"][i] for i in chosen])

    print(f"Chosen indices: {chosen}")
    print(f"Labels (0=RFI,1=FRB): {chosen_labels.astype(int).tolist()}")

    resized = tf.image.resize(dm_imgs[..., np.newaxis], (64, 64)).numpy()  # (N,64,64,1), matches notebook exactly

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preds = model.predict(resized, verbose=0)  # (N,2) softmax reference

    in_path = os.path.join(OUT_DIR, "tb_input_features.dat")
    out_path = os.path.join(OUT_DIR, "tb_output_predictions.dat")
    with open(in_path, "w") as fin, open(out_path, "w") as fout:
        for img, pred in zip(resized, preds):
            fin.write(" ".join(f"{v:.8f}" for v in img.flatten()) + "\n")
            fout.write(" ".join(f"{v:.8f}" for v in pred) + "\n")

    print(f"Wrote {len(resized)} samples -> {in_path}, {out_path}")
    print(f"Reference (float Keras) predictions:\n{preds}")


if __name__ == "__main__":
    main()
