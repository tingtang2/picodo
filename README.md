# Picodo: fast Transformer decoder training in JAX/NNX

- Picodo has <200 SLOC
- can run on GPUs, TPUs, Google Colab, or even locally on a Mac
- achieves 64% MFU on TPU v2-8 when training GPT2-small (124M)
- supports Fully Sharded Data Parallel (FSDP) parallelism
- uses the [new Flax NNX Api](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/nnx_basics.html)
- uses [Hydra](https://github.com/facebookresearch/hydra) for experiment management
- uses [Weights & Biases](https://github.com/facebookresearch/hydra) for experiment tracking

# Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb)

Picodo requires a pretokenized dataset for training following the same format as [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext). This speeds up training and simplifies the codebase. To get started, I prepared a pretokenized sample of 2.5B tokens from fineweb-edu here: [train.bin](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/datasets/fineweb_edu_gpt2_train.bin), [valid.bin](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/datasets/fineweb_edu_gpt2_val.bin).

The simplest way to use this codebase is by using the provided [Colab notebook](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb), which automatically installs requirements, downloads the dataset, and starts training a model.

To run this codebase using bash, simply set the [config name](configs) and any overrides:
```bash
python main.py -cn colab opt.peak_lr=0.004
```

You can also run `main.py` directly, which uses the `local.yaml` config by default (designed for local development).

# Inspiration

This repository was originally a fork of [deepmind/NanoDO](https://github.com/google-deepmind/nanodo) but it no longer shares any lines of code. Some notable changes:
- NanoDO has ~1800 SLOC while Picodo only has ~200 SLOC
- Picodo doens't rely on [grain](https://github.com/google/grain) for data loading so it can run locally on a Mac
- Picodo uses the [new Flax NNX Api](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/nnx_basics.html)
- Picodo uses Hydra and Weights & Biases instead of Google's ConfigDict / Tensorboard
