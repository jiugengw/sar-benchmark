# Dataset audit: leakage in public SAR detection benchmarks

This section is separate from the course project (the CS701 model work in the rest of this repository). It checks
whether published SAR detection benchmarks repeat training content in their evaluation splits, as we found in the
course data, where 26% of the test images were flipped copies of labelled images.

## Datasets

| Dataset | Folder | Status |
|---|---|---|
| SARDet-100K (NeurIPS 2024) | [sardet100k/](sardet100k/) | First pass done: 50.1% of val and at least 27.9% of test repeat train content |
| RSAR (rotated boxes on SARDet-100K images) | `rsar/` (to do) | Not started |
| M4-SAR (CVPR 2025, optical and SAR pairs) | `m4sar/` (to do) | Not started |

## What counts as leakage

| Type | Example | How we find it |
|---|---|---|
| Exact or re-saved copy | The same chip in train and val | Full-resolution correlation |
| Flipped or turned copy | Augmented copies split after augmenting | Search under all 8 flips and 90-degree turns |
| Re-processed copy, or another look at the same place | The same ship and coastline with new speckle or brightness | Correlation after averaging out speckle, plus label transfer |
| Overlapping tiles | Crops of one large scene with a shared strip | Shift or crop search (to do) |
| Same area, no shared pixels | Neighbouring tiles from one scene | Metadata only, where it exists |

Matches are confirmed by **label transfer**: the matched training boxes, turned back onto the evaluation image,
should recover its boxes. Boxes from a random image of the same source must not, which rules out layout
coincidences such as "ship always in the middle".

## Layout

Each dataset folder has its own `README.md` (method, results, how to reproduce), its scripts, a `results/` folder
(committed), and a `data/` folder for the download and caches (git-ignored).

## Environment

Runs on a laptop CPU, not the cluster: the datasets have more files than the cluster's 100k-file quota. Create the
environment with `python3 -m venv ~/.venvs/sar-audit` and install `kaggle six faiss-cpu numpy pillow`.
