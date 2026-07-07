# RASC-Net

**Reliability-Aware Subset-Consistent Multi-Scale Disentanglement for Incomplete Modality Brain Tumor Segmentation**

Official implementation of the paper *Reliability-Aware Subset-Consistent
Multi-Scale Disentanglement for Incomplete Modality Brain Tumor Segmentation*
(submitted to *IEEE Journal of Biomedical and Health Informatics*).

RASC-Net is a reliability-aware, stability-driven framework for robust brain
tumor segmentation from **incomplete multimodal MRI**. It separates
modality-invariant anatomical structure from modality-specific appearance at
multiple scales, estimates the reliability of each available modality, gates out
unreliable contributions before fusion, and enforces consistency across
arbitrary modality subsets.

<div align="center"> 
  <img src="assets/Supplementary Video Output.mp4" width="90%"/> 
</div>

> This code base extends the DC-Seg (MICCAI 2025) framework. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a component-by-component map
> from code to the manuscript equations, and for the differences from DC-Seg.

---

## Highlights

- **Multi-scale anatomy–style disentanglement** at semantic scales i∈{3,4} with
  hierarchical contrastive and reconstruction objectives.
- **Reliability-aware anatomical gating**: a lightweight per-modality quality
  score q_m ∈ [0,1] down-weights noisy/corrupted inputs; missing modalities are
  gated to exactly zero.
- **Stability-constrained fusion**: availability-masked spatial-attention softmax
  so only available modalities contribute and per-voxel weights sum to one.
- **Subset consistency regularization**: invariance across *any* subset of
  modalities, not just full/single-modality pairs.

## Method overview

```
Input ─► Shared multi-scale encoder ─► Anatomy/Style disentanglement (i∈{3,4})
      ─► Reliability-aware gating ─► Stability-constrained fusion
      ─► Region-aware segmentation decoder ─► Tumor segmentation
                         ▲
             Subset consistency regularization
```

## Installation

```bash
conda create -n rascnet python=3.10
conda activate rascnet
pip install -r requirements.txt
```

## Data

Three benchmarks are supported: BraTS 2015, BraTS 2018, and BraTS 2020.
See [`docs/DATA.md`](docs/DATA.md) for the expected `.npy` layout, the fixed
modality ordering `[FLAIR, T1ce, T1, T2]`, and preprocessing with
`preprocess.py`.

## Training

Edit `DATAPATH` / `SAVEPATH` in the scripts, then:

```bash
# BraTS 2020 (all 15 missing-modality combinations)
bash scripts/train_brats2020.sh

# BraTS 2018 (split 3) or BraTS 2015
bash scripts/train_brats2018_2015.sh BRATS2018
bash scripts/train_brats2018_2015.sh BRATS2015
```

Key hyper-parameters (defaults follow the manuscript, Sec. III-H):
`crop_size 112`, `num_epochs 500`, `batch_size 2`, Adam `lr 2e-4`,
`alpha 0.5` (L_dis weight), `beta 0.4` (L_cons weight), `tau 0.1`.

## Evaluation

```bash
bash scripts/evaluate.sh /path/to/model_last.pth BRATS2020
```

This runs sliding-window inference and reports Dice for whole tumor (WT),
tumor core (TC), and enhancing tumor (ET) across all 15 modality combinations.

## Ablations

`scripts/ablation.sh` reproduces the component study (manuscript Table IV) by
toggling the multi-scale disentanglement loss (`--no_disentangle_loss`),
reliability-aware gating (`--no_gating`), stability fusion (`--no_fusion`), and
subset consistency (`--no_consistency`).

## Reported results

Average Dice (%) on the three benchmarks, as reported in the manuscript:

| Dataset    | WT    | TC    | ET    |
|------------|-------|-------|-------|
| BraTS 2020 | 89.32 | 82.58 | 66.97 |
| BraTS 2018 | 87.85 | 80.74 | 59.16 |
| BraTS 2015 | 89.43 | 74.17 | 60.58 |

(BraTS 2020 numbers are averaged over all 15 missing-modality combinations.)

## Repository layout

```
RASC-Net/
├── rascnet/                # model + new modules
│   ├── encoder.py          # shared encoder, decoders, region-aware fusion
│   ├── disentangle.py      # multi-scale anatomy/style disentanglement
│   ├── reliability.py      # reliability-aware anatomical gating
│   ├── fusion.py           # stability-constrained fusion
│   ├── losses.py           # contrastive / reconstruction / subset losses
│   └── model.py            # full RASC_Net model
├── data/                   # BraTS dataset loaders & transforms
├── utils/                  # criterions, lr scheduler, parser
├── scripts/                # train / evaluate / ablation shell scripts
├── docs/                   # DATA.md, ARCHITECTURE.md
├── train.py                # training / evaluation entry point
├── predict.py              # sliding-window test-time inference
└── preprocess.py           # raw BraTS -> npy preprocessing
```

## Acknowledgements

This implementation builds on
[DC-Seg](https://github.com/CuCl-2/DC-Seg) (Li et al., MICCAI 2025) and the
region-aware fusion design of
[RFNet](https://github.com/dyh127/RFNet) (Ding et al., CVPR 2021). The
preprocessed BraTS data layout follows the RFNet convention. We thank the
authors for releasing their code and data.

## Citation

If you find this work useful, please cite the paper:

```bibtex
@article{rascnet2026,
  title   = {Reliability-Aware Subset-Consistent Multi-Scale Disentanglement
             for Incomplete Modality Brain Tumor Segmentation},
  author  = {Ray, Swachha and Yin, Yunfei and Dey, Argho and Liu, Hongyu and
             Huang, Zhiqiu and Islam, Md Rakibul and Yuan, Zheng and
             Xiong, Sijing and Timbi, Zumnan and Islam, Md Minhazul},
  journal = {IEEE Journal of Biomedical and Health Informatics (under review)},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
