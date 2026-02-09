# ECGFlow

> A foundation model for ECG-based assessment of cardiac and coronary
  function

This repository contains code used in the study:

```
Moody JB, Poitrasson-Riviere A, Renaud JM, et al., A foundation
transformer model with self-supervised learning for ECG-based
assessment of cardiac and coronary function.
2025;2023.10.25.23297552.
```

(Preprint available from
[medRxiv](https://www.medrxiv.org/content/10.1101/2023.10.25.23297552v3))

The model architecture is a vanilla [Vision
Transformer](https://arxiv.org/abs/2010.11929) adapted to accept ECG
waveforms as input (1dViT).
Our implementation uses code based on [Python Image Models
v0.9.14dev0](https://github.com/huggingface/pytorch-image-models) with
additions to support ECG data and self-supervised learning (SSL).
Our modified Pytorch Image Models can be found
[here](https://github.com/4dm-labs/pytorch-image-models-ecgflow).
Note that use of ECG data input with models other than Vision Transformer has not been fully tested and may not work.

ECGFlow uses an SSL implementation based on code from
(https://github.com/facebookresearch/mae).
During SSL pretraining we use an asymmetric encoder/decoder
architecture as described in the paper [Masked autoencoders are
scalable vision learners](https://arxiv.org/abs/2111.06377).
After pretraining the decoder is discarded and the encoder is
fine-tuned for downstream tasks.

As a demonstration, we provide scripts to pretrain the 1dViT with the
publicly available [MIMIC-IV-ECG
dataset](https://physionet.org/content/mimic-iv-ecg/1.0/).

This model is then fine-tuned for three ECG auto-interpretation tasks
using the publicly available [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/)
 and [PTB-XL benchmarks](https://www.nature.com/articles/s41597-020-0495-6).

# Installation

 1. Clone this repo
 ```
$ git clone https://github.com/4dm-labs/ecgflow
```

 3. Clone our modified python-image-models-ecgflow repo
    (ensure both repos are located in a common parent directory) 
 ```
$ git clone https://github.com/4dm-labs/pytorch-image-models-ecgflow
```

 4. Install the astral `uv` tool
 ```
$ curl -LsSF https://astral.sh/uv/install.sh | sh
```

 5. Create and populate the local python virtual environment
 ```
$ cd ecgflow
$ uv sync
 ```

# Usage

## Directory structure

 The provided scripts use the following directory structure:

 - Root workspace :: ~/ecgflow
   - Data directories ::
     - MIMIC-IV-ECG :: ~/ecgflow/data/mimic-iv-ecg
     - PTB-XL :: ~/ecgflow/data/ptb-xl
   - Output directories ::
     - SSL-pretrained model :: ~/ecgflow/experiments/mimic/mtsm-p50-d12-h8
     - Fine-tuned SSL model ::
       - ~/ecgflow/experiments/ptbxl_diag/mvtst-p50-d12-h8.mimic-1
       - ~/ecgflow/experiments/ptbxl_rhythm/mvtst-p50-d12-h8.mimic-1
       - ~/ecgflow/experiments/ptbxl_form/mvtst-p50-d12-h8.mimic-1
     - De novo supervised learning model ::
       - ~/ecgflow/experiments/ptbxl_diag/mvtst-p50-d12-h8-1
       - ~/ecgflow/experiments/ptbxl_rhythm/mvtst-p50-d12-h8-1
       - ~/ecgflow/experiments/ptbxl_form/mvtst-p50-d12-h8-1

The Data directories must be created and populated with data before
running the provided scripts.
The Output directories are created automatically.

## Running the scripts

Use the `uv` tool within the top-level `ecgflow` directory to run the scripts.

### SSL pretraining

```
$ cd ecgflow
$ uv run scripts/pretrain-mtsm-mimic.sh
```

### Supervised fine-tuning

```
$ cd ecgflow
$ uv run scripts/fine-tune-mvtst-ptbxl_diag1d.sh
$ uv run scripts/fine-tune-mvtst-ptbxl_rhythm1d.sh
$ uv run scripts/fine-tune-mvtst-ptbxl_form1d.sh
```

### De novo supervised training

```
$ cd ecgflow
$ uv run scripts/train-mvtst-ptbxl_diag1d.sh
$ uv run scripts/train-mvtst-ptbxl_rhythm1d.sh
$ uv run scripts/train-mvtst-ptbxl_form1d.sh
```

# Pretrained models

The following pytorch pretrained models used in our paper are
available:

 - [SSL-pretrained checkpoint](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ssl_mimic)
   
 - The SSL-pretrained model fine-tuned for three multi-label tasks
   (see [PTB-XL, a large publicly available electrocardiography
   dataset](https://www.nature.com/articles/s41597-020-0495-6) and
   [Deep learning for ECG analysis: benchmarks and insights from
   PTB-XL](https://arxiv.org/abs/2004.13701)):

   - [Prediction of Diagnostic labels (SSL)](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ssl_mimic_ft_ptbxl-diag)

   - [Prediction of Rhythm labels (SSL)](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ssl_mimic_ft_ptbxl-rhythm)

   - [Prediction of Form labels (SSL)](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ssl_mimic_ft_ptbxl-form)

 - The same 1dViT architecture trained de novo from random initialization for the same three multi-label tasks:
 
   - [Prediction of Diagnostic labels (de novo)](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ptbxl-diag)

   - [Prediction of Rhythm labels (de novo)](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ptbxl-rhythm)

   - [Prediction of Form labels (de novo)](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ptbxl-form)
