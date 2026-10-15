# Notebooks (moved here from the repo root for readability)

Notebooks were moved into these folders, but they still use repo-root-relative paths
(e.g. `./augmented_dataset/lstm_dataset_v6.npz`). Run them with the repo root as the working
directory, or adjust the paths.

* `augmentation/`: how the datasets were built.
  * `frb_augmentation_v6.ipynb`: the v6 dataset.
  * `frb_augmentation_v6_with_dm.ipynb`: identical rerun that also saves DM; produced the on-disk file.
  * `frb_augmentation_v5.ipynb`: the previous version.
  * `dm_extraction(1).ipynb`: builds `dm_catalog.json`.
  * `data_processing.ipynb`.
* `training/`: the original v6 training notebooks.
  * `cnn_training*.ipynb`: cnn_v1, 97.4 %.
  * `lstm_training*.ipynb`: lstm_v1–v3; lstm_v3 = 96.4 %.
  * `frb_ablation_study.ipynb`: the September filter / pooling / time-resolution ablation behind `fpga_handoff`.
  * `train_cnn.py`.
* `historic/`: earlier explorations.
  * FETCH-data classifiers (`classifier/`).
  * The pooled-CNN synthesis notebooks (`context/`, the 61.5 µs Vitis result).
  * The streaming / realtime notebooks.
  * The RFI classifier.
