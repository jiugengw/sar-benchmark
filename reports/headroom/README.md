# Headroom: how much AP is left to win, per class (19 October 2026)

Clean holdout (720 images without a training twin), best model so far: memorise at lr 0.002
(`runs/ft40-memorise-lr2-332326`, clean mAP 50.40, official Δm +49.21), one2many head, NMS 0.7 as submitted.

**Short version:** a perfect model would not get 100 here, because many labels are inconsistent or mark things
the image barely shows. Where label noise can be measured, little box-tightness headroom is left (about +1 mAP).
The two biggest loose-box losses (ship, wind turbine) could not be measured. Misses are the largest bucket, but
whether they are winnable is still open: bridge and playground are the test case (see "Open question").

## The table

AP points lost per class. Columns sum to 100 − AP.

| Class | AP | Missed | False alarms | Loose boxes | Winnable from loose boxes | Evidence |
|---|---:|---:|---:|---:|---|---|
| ship | 46.5 | 13.4 | 2.8 | **37.3** | unknown | label padding varies by image; misses mostly on land or noise |
| wind turbine | 61.3 | 3.9 | 1.1 | **33.6** | unknown; +8 to +14 if edges reached 0.2–0.3 px | 8 px objects, our edges already ~0.44 px off |
| car | 66.8 | 4.3 | 1.5 | 27.3 | +3.4 | label noise ≈ 70% of our edge-error variance (3 twin pairs, weak) |
| airport | 63.9 | 6.8 | 4.9 | 24.4 | unknown | no twin pairs with differing labels |
| aircraft | 68.1 | 5.2 | 5.8 | 20.8 | unknown | twin method not valid (see caveats) |
| oil tank | 66.9 | 13.8 | 2.3 | 17.1 | unknown | twin method not valid; misses are 4–9 px tanks in clutter |
| harbor | 60.0 | 22.0 | 1.1 | 17.0 | +2.2 | label noise ≈ 2/3 of our edge-error variance (2,020 edges, solid); misses on plain coastline |
| playground | 13.9 | 62.3 | 11.0 | 12.8 | unknown | labels drawn on optical images |
| bridge | 6.2 | **71.9** | 12.0 | 10.0 | +3.7 (44 edges, weak) | many labels hard to see |
| **mAP points** | **50.4** | **22.6** | **4.7** | **22.3** | **≈ +1 measured** | |

- **Missed** = 100 − maximum recall at IoU 0.5: labels with no matching box at any confidence.
- **False alarms** = maximum recall − AP50: wrong boxes ranked among the right ones. Some may be real objects nobody
  labelled.
- **Loose boxes** = AP50 − AP: found, but not tight enough for the 0.55–0.95 thresholds.

Raw numbers: [headroom_study.json](headroom_study.json).

## How the loose-box ceiling was estimated

1. **Label noise from twins.** Some scenes appear twice among the training images (`sarbench/train_twins.json`).
   Most twin pairs carry identical, copied boxes, which say nothing about noise. In the pairs whose boxes were drawn
   separately, the shared shift of all boxes (image registration) is removed, and the box-by-box edge differences
   give the label noise: σ_label = robust σ of the edge differences ÷ √2.
2. **Our edge error** σ_ours against the holdout labels (boxes with IoU ≥ 0.5). It contains the label noise:
   σ_ours² = σ_model² + σ_label².
3. **Simulated tightness** (AP ÷ AP50) for a given edge noise, on the class's holdout box sizes. The simulation
   overestimates our actual tightness by 0.07–0.13 in every class (the "check" column of the study), so each ceiling
   is scaled by actual ÷ simulated before use.
4. **Ceiling** = AP50 × tightness at σ_label; winnable = ceiling − AP.

Code: [`models/yolo/headroom_study.py`](../../models/yolo/headroom_study.py); twin check:
[`tools/analysis/twin_label_noise.py`](../../tools/analysis/twin_label_noise.py).

### Caveats

- **Oil tank and aircraft:** their separately drawn twin labels disagree more (σ 6.7–6.9%) than our model
  disagrees with the holdout labels (1.9–2.4%). Real label noise cannot exceed our total error, so those twin
  pairs are not typical labels, and the ceiling for these classes is unknown.
- **Few twin pairs** for car (3), bridge (2) and none with differing labels for ship, wind turbine and airport.
- **Misses are judged from 8 random samples per class**, by eye. A model can use speckle statistics and context a
  person cannot see in a crop, so "not visible" is an impression, not proof.

## Ships

[Worst ship boxes](../class-review/ship_worst_boxes.png) (label green, ours red):

- 144 of 149 clean-holdout ships are in 256 px images whose image class is "ship": one source.
- Median size 28 px. At that size a 1 px offset in each direction already gives IoU 0.87.
- Our boxes are unbiased (median width −1.8%, height +1.4%) but scatter by about 1 px.
- Tall (north–south) ships are worst: height error 15% vs width error 7% for wide ships. Radar smears strong
  targets into vertical streaks, and labels sometimes include the streak and sometimes not.
- Labels pad the ship by different amounts (e.g. images 7373, 4088, 146, 3645, 6379); some are shifted (3526, 3645).
- Real model misses exist: image 7859 (108 px ship boxed by half), ships close together (75% found vs 90% alone).

Code: [`tools/analysis/box_error_study.py`](../../tools/analysis/box_error_study.py),
[`tools/analysis/worst_box_crops.py`](../../tools/analysis/worst_box_crops.py).

## Misses

[Random missed harbors, oil tanks and ships](../class-review/missed_harbor_oiltank_ship.png) (captions are under
each crop; green label, red our same-class boxes, blue other classes):

- **Harbor:** labels on stretches of coastline with no visible harbour structure.
- **Oil tank:** 4–9 px tanks among similar bright pixels in industrial areas. The images are not upscaled
  (neighbouring pixels are equal only 5–18% of the time); the tanks are simply tiny. Oil tank labels come in two size
  groups, about 7–11 px and 25–31 px; 44% are under 10 px.
- **Ship:** several labels on land or noise; one clearly visible ship missed (7896).

Code: [`tools/analysis/misses_vs_false_alarms.py`](../../tools/analysis/misses_vs_false_alarms.py),
[`tools/analysis/missed_crops.py`](../../tools/analysis/missed_crops.py).

## Open question: are bridge and playground data-limited?

YOLO26 on M4-SAR (similar Sentinel-1 radar, about 13× more images) reaches 48.8 AP50 on bridge and 75.8 on
playground; we reach 16.2 and 26.7. So these objects are detectable in this kind of imagery. The gap could be data
quantity, an easier test set, or labels that don't match our images. The **learning curve**
([`sbatch/learning_curve.sbatch`](../../sbatch/learning_curve.sbatch), the 100-epoch stage on half the training
images) decides it: a clear drop with half the data means still data-limited.

## Related reading

- Murrugarra-Llerena et al., [Can We Trust Bounding Box Annotations for Object Detection?](https://mlanthology.org/cvprw/2022/murrugarrallerena2022cvprw-we)
  (CVPR workshops 2022): label noise distorts AP most for small objects and strict thresholds.
- [TOLF: Noise-Robust Tiny Object Localization with Flows](https://arxiv.org/abs/2601.00617) (2026): models box
  noise and down-weights uncertain labels, +1.2 AP on AI-TOD with DINO.
