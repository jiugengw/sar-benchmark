# Our final model on the released val and test labels (13 Nov 2026)

The TA released the val and test labels on Hugging Face on 13 Nov 2026, after the leaderboard closed
([doem1997/cs701-sar-course-data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data): `val_labels.csv`,
`val_instances.json`, `test_labels.csv`, `test_instances.json`; kept in the git-ignored `data/course-data/released-labels/`).
The submissions are unchanged: these labels are used only to analyse them.

## The local scorer matches Codabench

`scripts/score_released.py ZIP val|test` gives the same numbers as Codabench, so every ZIP can now be scored offline.

| ZIP | Codabench | Local |
|---|---|---|
| [`submissions/final/final.zip`](../submissions/final/) on val | Δm +65.85, F1 95.40, mAP 59.96 | Δm +65.86, F1 95.40, mAP 59.96, AP50 75.61 |
| [`submissions/phase2/phase2.zip`](../submissions/phase2/) on test | Δm +71.57, F1 96.11, mAP 61.21, AP50 77.75 | Δm +71.58, F1 96.11, mAP 61.21, AP50 77.75 |

The Δm values differ only in rounding. Δm uses each phase's reference: val F1 93.31 / mAP 26.13, test 93.99 / 25.41.

## Per class on test (Phase 2 ZIP)

| Class | Images | AP | AP50 | F1 |
|---|---:|---:|---:|---:|
| aircraft | 270 | 68.60 | 93.02 | 99.82 |
| airport | 190 | 81.45 | 94.04 | 96.35 |
| **bridge** | 227 | **6.86** | 14.85 | 86.03 |
| car | 153 | 70.36 | 97.08 | 100.00 |
| harbor | 128 | 82.80 | 89.26 | 97.64 |
| oil_tank | 204 | 79.80 | 90.16 | 97.52 |
| **playground** | 246 | **21.56** | 31.50 | 88.57 |
| ship | 253 | 54.33 | 91.54 | 99.21 |
| wind_turbine | 394 | 85.16 | 98.33 | 99.87 |

- Bridge and playground are the whole gap, as on the holdout ([hard classes](HISTORY.md)).
- Of the 78 misclassified test images, 45 are bridge and playground swapped: 23 bridges called playground and 22 playgrounds called bridge.

## Copies vs new images

`tools/analysis/released_groups.py ZIP val|test` compares every image with all training images of its size (64 px
thumbnails, Pearson r, under the dataset's flips). It then scores each group with the same rules as the holdout
groups:
- copy: r ≥ 0.9;
- near-copy: 0.7–0.9;
- borderline: 0.6–0.7;
- new: r < 0.6.

| Group | Test images | Test Δm | Test F1 | Test mAP | Val images | Val Δm | Val mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 2,065 | +71.58 | 96.11 | 61.21 | 1,426 | +65.86 | 59.96 |
| Copies, r ≥ 0.9 (flipped + unflipped) | 657 (32%) | +112.23 | 98.80 | 81.15 | 479 (34%) | +114.84 | 84.61 |
| ... flipped copies | 610 | +105.58 | 100.00 | 77.44 | 438 | +134.49 | 94.54 |
| Near-copies 0.7–0.9 | 261 | +46.70 | 94.69 | 48.96 | 191 | +47.01 | 51.40 |
| Borderline 0.6–0.7 | 165 | +53.40 | 93.45 | 52.70 | 90 | +40.15 | 49.43 |
| **New, r < 0.6** | **982 (48%)** | **+42.96** | 93.23 | **47.45** | 666 (47%) | +32.93 | 43.44 |

- **On new images the final model scores mAP 47.5 on test.** Copies score 81.2, so a third of the test set lifts the
  overall 61.2.
- **new6 on test is 46.8.** That is the mean AP of the six classes with enough new images. It is what we ranked
  Phase 2 candidates by, and it matches the holdout estimates of 31 Oct (46.5–47.0).
  - So the holdout predicted performance on new images well.
- Mean AP over all nine classes lands slightly higher (47.5) because airport and harbor are easy.
- Test new images do better than val new images (mAP 47.5 vs 43.4). Their class mix differs.

Results: `results/released_labels/` (`phase2_test.json`, `final_val.json`, and the group of every image in
`groups_{val,test}.json`, with its best training match).
