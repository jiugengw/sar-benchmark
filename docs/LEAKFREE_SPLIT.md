# A leak-free re-split of the course data (19 Nov 2026)

After the TA released the val and test labels (13 Nov), all 12,883 labelled images can be pooled and split again so
that no val or test image repeats a training image. This page gives the rule, how much the official split leaks, and
the new split: [`results/leakfree/split.csv`](../results/leakfree/split.csv) (md5 `38a4c36ce3ea5fe28b998bb85c3d3608`).

## How much the official split leaks

Every image was compared with every other image, on the radar pixels, by the copy searches of our public-benchmark
leakage audit (see [Method](#method)). A val / test image leaks when it has a link to a training image:

| | val | test |
|---|---:|---:|
| **Images with a copy in train** | **679 of 1,426 (47.6%)** | **985 of 2,065 (47.7%)** |
| whole copy, flipped or turned 180° | 334 | 477 |
| whole copy, unflipped | 45 | 61 |
| partial overlap (shifted or neighbouring tile, 10-90% shared) | 294 | 444 |
| zoomed copy (a 256 px chip of a 512 px tile) | 6 | 3 |

Per class (test): aircraft 88%, oil tank 78%, wind turbine 73%, airport 64%, harbor 48%, playground 22%, car 16%,
ship 12%, bridge 8%.
- The thumbnail check of 13 Nov ([RELEASED_LABELS.md](RELEASED_LABELS.md)) found about 32% whole copies. Partial
  overlaps add about 15 points; aircraft tiles in particular overlap their neighbours (whole airports cut into
  overlapping tiles).
- 47.7% is a lower bound: thin overlaps are hard to find on speckled radar (see the controls).

## The rule for the new split

**Split by group, not by image.** Images that share radar pixels (copies, flipped copies, overlapping tiles, zoomed
chips) form one group, and a whole group goes into one split. This is the standard practice for data with repeated or
neighbouring samples:
- group (spatial) cross-validation: Roberts et al., *Ecography* 2017, "Cross-validation strategies for data with
  temporal, spatial, hierarchical, or phylogenetic structure"; scikit-learn's `GroupKFold` / `StratifiedGroupKFold`;
- removing near-duplicates of training images from test: Barz & Denzler 2020 ("Do we train on test data? Purging
  CIFAR of near-duplicates"); Recht et al. 2019 (ImageNet v2);
- in aerial detection, DOTA splits by original large image before cutting tiles.

Steps (`tools/leakfree/summary.py`):
1. **Links**: every pair of images whose radar pixels match, with the rules below.
2. **Groups**: any two linked images go in one group, following chains (union-find). 7,468 groups; the largest has 145
   images; 6,049 images match nothing.
3. **Split**: groups are shuffled with seed 0 and taken largest first; each goes whole to the split (train / val /
   test) that most needs its (class, image size) mix. The targets are the official split's own counts, so the new
   split keeps the TA's class mix in each split.
4. **Checks**: no link crosses two splits (0); class and image-size counts per split equal the official split's.

| | train | val | test |
|---|---:|---:|---:|
| images (official = new) | 9,392 | 1,426 | 2,065 |
| per class and image size (512 / 256 / other) | same as official | same as official | same as official |
| links crossing splits | | 0 | |

5,373 images change split. `split.csv` lists every image: `file` (as in `data/course-data/`), `official_split`,
`new_split`, `group`, `class`, `size`, `width`, `height`.

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `tools/leakfree/pack.py` | All labelled images in one zip, one id each (`ids.csv`) |
| 2 | `tools/leakfree/coursesearch.py pix` (runs `pixsearch.py`) | Same-zoom copies under the 8 flips / turns: shortlist on blurred thumbnails (whole-image descriptors and 16 px patch votes, finds shifted and overlapping tiles), then aligned correlation at 128 px and at full resolution over the shared region |
| 3 | `tools/leakfree/coursesearch.py scale` (runs `scalesearch.py`, all pairs) | Zoomed copies, zoom 1.1-2 (one image a magnified part of another; a 256 px chip of a 512 px tile is zoom 2 after resizing to 512) |
| 4 | `tools/leakfree/summary.py` | Rules, leak rates, groups, split |

The cluster job is `sbatch/staging/job_ai.sbatch` (about 3.7 h on a 24 GB GPU slice). The search code is copied from
our separate audit of public SAR benchmarks (SARDet-100K, RSAR, M4-SAR), where it was written and checked.

**Link rules**, set from the controls:

| link | rule (Pearson r of the radar pixels over the shared region) |
|---|---|
| same zoom, >= 25% of the image shared | r >= 0.97 |
| same zoom, 10-25% shared | r >= 0.99 |
| zoomed, >= 30% of the magnified image on the other | r >= 0.98 |

**Controls** (`results/leakfree/summary.json`):
- *False links*: val / test images searched against a fake train pool in which every image is cut into 4 x 4 blocks,
  shuffled and turned (no copies possible). No image passes either rule. The best fake match reaches r 0.953 (same
  zoom, thin overlap), 0.929 (25-50% shared) and 0.976 (zoomed); real zoomed chips score about 0.998.
- *Recall*: copies of val / test images planted in the pool (re-saved as JPEG, shifted or zoomed). Same zoom, by
  share of the image shared: 100% at 75-100%, 90 / 100 at 50%, 72 / 100 at 30%, 18 / 100 at 15%. Zoomed: 142 / 200
  enlarged crops and 64 / 100 shrunk copies fully inside a tile. So some thin overlaps are missed, and the leak rate is
  a lower bound.

## Using it

Train the same recipe on the new `train` (+ `val` where the recipe uses the holdout), unchanged, and score on the new
`test` with `scripts/score_released.py`'s rules; report it next to the official test score. Settings chosen on the
official val stay as they were: nothing is tuned on the new test.
