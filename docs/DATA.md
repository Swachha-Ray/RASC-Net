# Data Preparation

RASC-Net follows the same data layout as RFNet / DC-Seg. Three datasets are
supported: **BraTS 2015**, **BraTS 2018**, and **BraTS 2020**.

## Expected directory layout

After preprocessing, each dataset directory should look like:

```
BRATS2020_Training_none_npy/
├── vol/
│   ├── BraTS20_Training_001_vol.npy      # (H, W, D, 4) float32, modalities [FLAIR, T1ce, T1, T2]
│   └── ...
├── seg/
│   ├── BraTS20_Training_001_seg.npy      # (H, W, D) uint8 label map
│   └── ...
├── train.txt                              # one subject id per line
└── test.txt
```

For BraTS 2018 the loader defaults to `train3.txt` / `test3.txt` (split 3),
consistent with the standard three-fold protocol.

## Preprocessing

`preprocess.py` converts raw BraTS NIfTI volumes into the cropped, intensity-
normalized `.npy` files expected above. Edit the `src_path` / `tar_path`
variables at the top of the script, then run:

```bash
python preprocess.py
```

Preprocessing performs, per the manuscript (Sec. III-H):
skull-stripping is assumed already done by BraTS; the script crops the brain
region, and intensities are normalized to zero mean / unit variance within the
brain mask. During training, 3D patches of size `112 x 112 x 112` are randomly
cropped and random modality dropout simulates incomplete-modality inputs.

## Modality ordering

The channel order is fixed throughout the codebase as:

| index | 0     | 1    | 2   | 3   |
|-------|-------|------|-----|-----|
| modal | FLAIR | T1ce | T1  | T2  |

The 15 missing-modality masks in `train.py` follow this order.

## Public preprocessed data

The DC-Seg authors released a preprocessed BraTS 2020 copy (courtesy of the
RFNet authors). If you use it, please cite RFNet and DC-Seg accordingly. The
same `.npy` layout applies.
