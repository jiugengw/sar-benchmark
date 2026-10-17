# Paired ViT / TerraMind detection review

Compared selected 48-epoch-schedule checkpoints on identical annotations:
939 images, 3,281 objects, including 1,704 smaller than 16×16 pixels after
resizing to 512×512. All runs use seed 0, MoE-LoRA, large anchors and
photometric augmentation. Selected epochs differ because each maximizes its
own local Delta m. No official evaluation labels are used.

| Model | Selected epoch | Image macro F1 | COCO mAP | AP50 | AP75 | Delta m |
|---|---:|---:|---:|---:|---:|---:|
| TerraMind, box loss 2 | 46 | 93.45% | 35.54% | 65.40% | 32.41% | +18.08 |
| ViT, box loss 1 | 48 | 93.44% | 32.74% | 63.38% | 29.25% | +12.72 |
| ViT, box loss 2 | 44 | 92.11% | 33.44% | 63.12% | 30.18% | +13.34 |

## What accounts for the observed error gap

Both missed objects and box accuracy contribute. The table below is diagnostic
greedy, one-to-one, class-aware matching at confidence >=0.3. It is not COCO AP.
IoU >=0.75 requires tighter box alignment than IoU >=0.5.

| Measurement | TerraMind | ViT loss 1 | ViT loss 2 |
|---|---:|---:|---:|
| Matched objects, IoU >=0.5 | 2,436 | 2,341 | 2,350 |
| Missed objects, IoU >=0.5 | 845 | 940 | 931 |
| Matched small objects, IoU >=0.5 | 1,109 | 1,079 | 1,075 |
| Matched objects, IoU >=0.75 | 1,469 | 1,304 | 1,324 |
| Loose same-class predictions (diagnostic category) | 1,310 | 1,298 | 1,404 |
| Background-category false predictions | 1,762 | 1,498 | 1,764 |
| Duplicates | 189 | 160 | 155 |
| Wrong-class predictions overlapping a label at IoU >=0.5 | 0 | 0 | 1 |

Against ViT loss 2, TerraMind matches 86 more objects at IoU 0.5 and 145 more
at IoU 0.75. Its small-object match advantage is 34, so the difference also
includes larger objects. At confidence >=0.05 it still matches more objects:
2,607 versus 2,513 at IoU 0.5; 1,512 versus 1,364 at IoU 0.75. Thus the result
persists at a lower score cutoff, although score calibration/ranking still
affects AP. The background-category counts at confidence 0.3 are almost equal
for TerraMind and ViT loss 2. Confident wrong-class box matches are rare in this
diagnostic; image-classifier label confusions are a separate issue.

The AP gap is class-dependent:

| Class | TerraMind AP | ViT loss-2 AP | TerraMind minus ViT |
|---|---:|---:|---:|
| Ship | 39.04% | 32.26% | +6.78 |
| Car | 56.02% | 49.79% | +6.23 |
| Oil tank | 44.12% | 40.19% | +3.94 |
| Aircraft | 41.04% | 37.99% | +3.05 |
| Wind turbine | 39.04% | 36.82% | +2.22 |
| Bridge | 3.19% | 2.10% | +1.09 |
| Airport | 49.36% | 51.87% | -2.51 |
| Harbor | 45.18% | 46.46% | -1.28 |
| Playground | 2.88% | 3.48% | -0.60 |

Bridge and playground remain weak for both. The larger overall gaps come from
ship, car, oil tank, aircraft and wind turbine. ViT does better on airport,
harbor and playground AP, so TerraMind is not better on every class.

## Visually inspected examples

The [contact sheet](comparison-contact-sheet.jpg) shows ground truth, TerraMind,
ViT loss 1 and ViT loss 2 in that order. The [interactive gallery](gallery/index.html)
has 36 examples with class/error filters. All ground-truth boxes are drawn;
at most 20 predictions are drawn per panel for readability. Counts include all
predictions above threshold. Green labels are matched, orange labels missed;
blue predictions match, purple have loose localization, yellow are duplicates,
and red are background/wrong-class predictions.

- **Car image 4851:** TerraMind matches 9/11 labelled cars; both ViTs match
  1/11. The cars form a close row beside a road. TerraMind separates more of
  them; ViT leaves many annotations unmatched. These objects do not satisfy
  the review's small-object definition, so this example also motivates looking
  beyond tiny-anchor coverage.
- **Wind-turbine image 3900:** all 41 labels are small. TerraMind matches 34,
  ViT loss 1 matches 25 and ViT loss 2 matches 22. All models produce loose
  boxes around some bright structures, and TerraMind retains more matches.
- **Ship image 1607:** one ship at the image edge. TerraMind and ViT loss 1
  match it; ViT loss 2 has a high-confidence same-class prediction with loose
  overlap. This is an example of localization error despite recognizing a ship.
- **Airport image 9112:** ViT matches 2/2 labels while TerraMind matches 1/2.
  The ViT panels also show extra overlapping boxes. This is a counterexample
  to uniformly superior TerraMind detections.

Examples were deliberately selected for opposing differences and are not
estimates of how frequent those situations are. Full-holdout counts above
provide the frequency evidence. Greedy error categories are heuristic and
are not an exact decomposition of the COCO mAP difference. This review
identifies observable errors; it does not isolate pretraining, architecture
or adaptation as their cause. All experiments are one seed.

## Experiment implications

The new anchors are reasonable for improving small/elongated-box coverage,
but the existing gap includes larger targets and strict localization errors.
They cannot be assumed to fix the full gap. Task uncertainty weighting can
test classification/detection balancing; it is not a demonstrated fix for
these detection errors. A direct fixed-weight TerraMind run with the same new
anchors is needed to isolate the effect of learned weighting.

Reproduce numerical analysis with `tools/analyze_backbone_predictions.py --runs`
followed by the three run paths in `paired_error_analysis.json`, then
`--labels TerraMind ViT_loss1 ViT_loss2 --out reports/vit-terramind-48`.
Reproduce the gallery with `tools/inspect_detections.py`, those same run paths and
labels, and `--out reports/vit-terramind-48/gallery --examples 36`.
