# YOLO26m-P2 second continuation

Submission: [deadline-yolo26m-329644_continue_2.zip](deadline-yolo26m-329644_continue_2.zip).

The detector was fine-tuned for another 40 epochs from `continue_1/weights/best.pt`,
then fitted with a spatial image-classification head. Both tasks use the same
detector backbone. The fixed 939-image holdout was excluded from training.

| Local holdout metric | Previous continuation | Second continuation |
|---|---:|---:|
| Macro-F1 | 94.5821% | 94.4325% |
| Detection mAP | 44.7547% | 44.9806% |
| Delta m | 36.3202 | 36.6724 |

The local gain is 0.3522 Delta m. Official scores for this second continuation
are pending; the previous continuation's official Delta m is 34.5697. These
local and official measurements use different images.

The ZIP covers all 1,426 course validation images, using the one-to-many
detection branch, native detector classes, 512px inputs, and no ensemble or
test-time augmentation. `candidate.json` records checkpoint paths;
`holdout.metrics.json` and the inference JSON record evaluation and ZIP provenance.
Model checkpoints remain in the local `runs/deadline-yolo26m-329644/continue_2/`
directory. Holdout predictions for both detection branches are saved under
[`results/yolo26m_p2_finetune_329644/continue_2/`](../../../results/yolo26m_p2_finetune_329644/continue_2/).
