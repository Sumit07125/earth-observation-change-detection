# Geo-Nexus v3.2 Pretrained Foundation Weights

This directory contains the pretrained foundation backbones for Geo-Nexus (MH-DAPT-CD v3.2).

## Files & Checkpoints

### 1. Optical Branch: `resnet18_s2c_moco.pth`
- **Source:** SSL4EO-S12 (TorchGeo / Hugging Face `torchgeo/resnet18_sentinel2_all_moco`)
- **Architecture:** ResNet-18
- **Input Channels:** 13 (Sentinel-2 L1C multispectral bands)
- **Pretraining:** MoCo-v2 self-supervised pretraining across 251,079 global locations × 4 seasonal timestamps
- **License:** [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Reference:** Wang et al., *"SSL4EO-S12: A Large-Scale Multi-Modal Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation"*, IEEE GRSM 2023.

### 2. SAR Branch: `resnet18_s1_bigearthnet.pth`
- **Source:** BIFOLD BigEarthNet v2.0 (Hugging Face `BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0`)
- **Architecture:** ResNet-18 (classification head removed, clean feature extractor trunk)
- **Input Channels:** 2 (Sentinel-1 GRD dual-pol backscatter: VV, VH)
- **Pretraining:** Pretrained on 590,326 Sentinel-1 patches across Europe
- **License:** [MIT License](https://opensource.org/licenses/MIT)
- **Reference:** Clasen et al., *"Refined BigEarthNet (reBEN)"*, arXiv:2407.03653.

## Dataset License on Kaggle
Because this dataset combines CC-BY-4.0 and MIT licensed model checkpoints, it is staged under the Kaggle license identifier `other` with full upstream attribution documented here and in `pretrained_weights_manifest.json`.
