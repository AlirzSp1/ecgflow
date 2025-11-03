# ECGFlow

> A foundation model for ECG-based assessment of cardiac and coronary
  function

This repository contains code used in the study:

```
Moody JB, Poitrasson-Riviere A, Renaud JM, et al. Self-supervised
deep representation learning of a foundation transformer model
enabling efficient ECG-based assessment of cardiac and coronary
function with limited labels. 2024;2023.10.25.23297552.
```
(Preprint available from
[medRxiv](https://www.medrxiv.org/content/10.1101/2023.10.25.23297552v2))

The model architecture is a vanilla [Vision
Transformer](https://arxiv.org/abs/2010.11929) adapted to accept ECG
waveforms as input (1dViT).
Our implementation uses code based on [Python Image Models
v0.9.14dev0](https://github.com/huggingface/pytorch-image-models) with
additions to support ECG data and self-supervised learning (SSL).
Our modified Pytorch Image Models can be found
[here](https://github.com/4dm-labs/pytorch-image-models).
Note that use of ECG data input with models other than Vision Transformer has not been fully tested and may not work.

ECGFlow uses an SSL implementation based on code from
(https://github.com/facebookresearch/mae).
During SSL pretraining we use an asymmetric encoder/decoder
architecture as described in the paper [Masked autoencoders are
scalable vision learners](https://arxiv.org/abs/2111.06377).
After pretraining the decoder is discarded and the encoder is
fine-tuned for downstream tasks.

As a demonstration, we pretrain the 1dViT with the publicly available
[MIMIC-IV-ECG dataset](https://physionet.org/content/mimic-iv-ecg/1.0/). 

This model is then fine-tuned for three ECG auto-interpretation tasks
using the publicly available [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/)
 and [PTB-XL benchmarks](https://www.nature.com/articles/s41597-020-0495-6).

# Pretrained models

The following pytorch pretrained models used in our paper are
available:

 - [SSL-pretrained checkpoint]()
   
 - The SSL-pretrained model fine-tuned for three multi-label tasks
   (see [PTB-XL, a large publicly available electrocardiography
   dataset](https://www.nature.com/articles/s41597-020-0495-6) and
   [Deep learning for ECG analysis: benchmarks and insights from
   PTB-XL](https://arxiv.org/abs/2004.13701)):

   - [Prediction of Diagnostic labels]()

   - [Prediction of Rhythm labels]()

   - [Prediction Form labels]()