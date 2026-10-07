# SocialSDE

Core code and processed training data for **Unveiling hidden features of social evolution by inferring Langevin dynamics from data**.

Paper: [arXiv:2601.17772](https://arxiv.org/abs/2601.17772)

## Contents

- `main_LBN_OLE.py`: training entry point for Langevin drift and diffusion models.
- `LBN/model/`: neural models, SWAG training, shared Langevin diagnostics, and stochastic force inference (SFI).
- `LBN/misc/`: data loading, configuration, and shared utilities.
- `data/DIG/DIG_pca_results.csv`: processed DIG training data.
- `data/Ngrams/ngram_emo2rat_coarse2_1850_thre0.0005.csv`: processed Ngram training data.
- `data/Seshat/seshat_nga_thread_mean_SCvalues.csv`: processed Seshat training data.

## Dependencies

The core code uses Python, PyTorch, NumPy, pandas, SciPy, and tqdm. A pinned environment specification is not included in this initial snapshot.

## Status

This initial snapshot contains core code and selected training data. Jupyter notebooks, trained checkpoints, and figure-reproduction workflows are not included. End-to-end execution of this snapshot has not yet been validated.
