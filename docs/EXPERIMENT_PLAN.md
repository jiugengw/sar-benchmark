# Plan to beat the leaderboard snapshot

Target supplied by the user: beat **Delta m +8.22**. The leading entry has macro-F1 **92.13%**, mAP **30.75%**,
accuracy **92.50%**, and AP50 **61.30%**. These are a snapshot, so check the latest leaderboard before final selection.

The score reference used by the current trainer is macro-F1 93.31%, mAP 26.13%. With macro-F1 92%, mAP must exceed
approximately 30.79% to beat +8.22. With macro-F1 93%, mAP must exceed approximately 30.51%. Aim for mAP above 31%
while preserving macro-F1 around 92–93%, with some margin rather than a rounded tie.

## Baseline comparison

Compare TerraMind + MoE-LoRA, ViT + MoE-LoRA, TerraMind + LoRA, and ViT + LoRA using final-epoch metrics. Use seed 0
and the same labeled-training holdout for the initial comparison. Use the local holdout to rank candidates; its
score is not a prediction of the official leaderboard score.

Creating a submission ZIP consumes no Codabench submission slots; uploading it does. With a daily limit of five
uploads, uploading three baseline ZIPs leaves two slots, while generating the ZIPs leaves all five available.

## Experiments after the baseline comparison

1. Upload the strongest local baseline first to establish an official score. Upload a runner-up only if comparing
   its different backbone/adaptation would provide useful evidence. Save the other slots for improvements.
2. Inspect detection errors on the labeled holdout, including missed small objects, wrong-category detections, and
   duplicate boxes. Preserve the holdout and the official validation images; do not train on held-out labels.
3. Try inexpensive inference variants on the holdout first: reweight each detection confidence using the image
   classifier's probability for that detection category, with powers 0 (unchanged), 0.5, and 1. Test detector NMS
   IoU thresholds 0.3, 0.5, and 0.7. Each variant still uses the same single model and one image input, with no TTA
   or ensemble. These are hypotheses, not guaranteed improvements. Keep a variant only if measured local Delta m
   improves without a damaging classification tradeoff. The current submission CLI needs additional options before
   these variants can be evaluated or submitted.
4. Inspect the winning baseline's learning curve. If mAP is still improving near epoch 24, try a new 36-epoch run
   using that configuration and the same seed, data split, batch size, and learning rate. This is a new run with a
   longer schedule; the current trainer has no resume option. If it has plateaued, prioritize a separate learning
   rate trial (5e-5 versus the baseline 1e-4) instead of assuming more epochs will help.
5. Submit only locally supported candidates, changing one factor at a time. Use the public score to check transfer
   from the holdout rather than to search a large grid. Once a method looks strong, check another training seed
   locally; each submission must remain a single model.

## Daily upload allocation

| Slot | Intended use |
|---|---|
| 1 | Strongest completed baseline |
| 2 | Best locally validated inference variant, or an informative runner-up |
| 3 | Best locally validated training variant |
| 4 | Further improved candidate, only with supporting local evidence |
| 5 | Reserve for the strongest final candidate or a necessary corrected submission |

If three baseline ZIPs are uploaded, spend the two remaining slots on the best supported improvement and
one reserve submission. Use fewer than five if there is no meaningful new candidate. Do not spend uploads on
unmeasured thresholds or trivial filename changes.

Stay within the README's limits: 130M total parameters, 40M trainable including heads, one model, input at most
512 x 512, no ensembles, and no test-time augmentation. Full fine-tuning is a local upper-bound experiment and is
not a compliant final method. Baseline recipes alone do not establish that +8.22 will be beaten.

Reference for detector NMS settings: [torchvision Faster R-CNN implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py).
