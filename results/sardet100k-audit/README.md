# SARDet-100K leakage audit (first pass, 13 Nov 2026)

Data: Kaggle `greatbird/sardet-100k` (train 94,493 and val 10,492 with labels; test 11,613 images without labels),
kept locally in `data/sardet100k/` (git-ignored). Source dataset per image is inferred from size, format and classes
(the annotation files have no source field); the counts match the per-source table in the official repo.

## Method

1. `tools/analysis/sardet_candidates.py`: every image becomes a 32 px area-averaged thumbnail (averaging removes most
   speckle). Each val or test image is compared with every train image under all 8 flips and 90-degree turns (FAISS,
   Pearson r), and the best match is kept.
2. `tools/analysis/sardet_verify.py`: each match with thumbnail r >= 0.7 is checked again:
   - `full_r`: correlation at full resolution;
   - `low_r`: correlation after 8x averaging (speckle removed);
   - val only: the matched train boxes are turned back onto the val image to see how many val boxes they recover
     (label transfer, IoU >= 0.5, same class).
   As a baseline, boxes from a random train image of the same source recover 0.00-0.01 of the val boxes in every
   source except SIVED (0.19).
3. Visual checks: `sheets/val_pairs_by_band.png` and `sheets/val_bandC_label_hits.png`.

## Leak groups

| Group | Rule | What it is (from the sheets) |
|---|---|---|
| A, exact copy | full_r >= 0.95 | The same file, often flipped or turned |
| B, same image, re-processed | low_r >= 0.95 | The same chip with small pixel differences |
| C+, same target, labels transfer | low_r >= 0.8 and >= 50% of the val boxes recovered | The same ship in the same place, with different intensity or noise (another look or polarisation) |

Matches with low_r below 0.8, or with labels that don't transfer, look like different scenes, mostly dark sea with
one bright object. They are not counted.

## Results

| Split | Images | A | B | C+ | Leaked |
|---|---|---|---|---|---|
| val | 10,492 | 1,531 | 1,921 | 1,801 | 5,253 (50.1%); 43.4% of the val boxes are on these images |
| test | 11,613 | 1,321 | 1,915 | n/a (no labels) | 3,236 (27.9%); lower bound |

Using the same image-only rule as test (A+B), val is 32.9%.

Leaked share by source (val):

| Source | Leaked |
|---|---|
| OGSOD | 99.5% |
| SADD | 100% |
| 256 px ship chips (ShipDataset/MSAR) | 58.5% |
| AIR_SARShip-1/MSAR (512 px) | 31% |
| SSDD | 18% |
| MSAR | 18% |
| SAR-AIRcraft | 10% |
| SIVED | 6% |
| HRSID | 4% |
| AIR_SARShip-2 | 0% |

Test shows the same pattern: OGSOD 99.5%, SADD 98%, 256 px ship chips 25% (A+B only).

44% of the leaked val images match a train image that is flipped or turned, so these look like augmentation copies
that were split after augmenting. 50 val images match a train image from another source, for example HRSID val
against 256 px ship chips (not yet checked by eye).

## Not covered yet, so these numbers are lower bounds

- Shifted or overlapping tiles (crops of one large scene): stage 1 matches whole images only. This matters for
  SAR-AIRcraft, HRSID and MSAR.
- Duplicates inside one split. Stage 1 gives 17.5% of val and 25.2% of test with a thumbnail r >= 0.99 twin inside
  the same split; these are not verified yet.
- How a trained model scores on the leaked group compared with the clean group.
