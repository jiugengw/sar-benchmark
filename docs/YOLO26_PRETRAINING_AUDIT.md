# YOLO26-P2 initialization audit — 18 October 2026

Job 330601 was intended to compare DOTA optical aerial pretraining against the
original COCO baseline 328339. Its first 61 completed epochs had exactly the same
training box/class losses and validation mAP as that baseline. It was cancelled,
with logs and checkpoints preserved, because it was repeating the baseline.

The script transferred pretrained weights into a YAML-created wrapper's modules,
but did not register them as the wrapper's training initialization. It also passed
`pretrained=False`. Ultralytics constructed a fresh seeded model for training,
discarding the manually transferred weights. The prior YOLO26-P2 baselines produced
by this script should therefore be described as random-initialized rather than
COCO-initialized. Their measured scores remain measurements of those trained models.

The corrected script uses the wrapper's public `load()` API to register the
prepared P2 model, then `pretrained=True` to preserve it at training setup. It
checks that the source first layer reaches P2, records the source checkpoint hash,
and fingerprints all prepared parameters. A callback compares those parameter
values with the actual trainer before its first optimizer step and fails if they
changed. The audit is saved in `initialization_audit.json`.

`tests/test_yolo_initialization.py` exercises the installed wrapper's actual
`Model.train` handoff on CPU, including a real model reconstruction and a negative
check that detects altered weights.

The replacement pair, `sbatch/yolo26m_pretraining_compare.sbatch`, runs:

Submitted array **330705**: DOTA **330705_0** is running and its live
`initialization_audit.json` confirms the trainer parameters exactly match the
prepared DOTA-initialized model. COCO **330705_1** is queued behind the two-job
limit while the matched large job continues.

| Setting | DOTA task 0 | COCO task 1 |
|---|---|---|
| Initialization | `pretrained/yolo26m-obb.pt` | `pretrained/yolo26m.pt` |
| Detector | YOLO26m-P2, 100 epochs, seed 0, batch 16, 512 input | Same |
| Augmentation and optimizer | Original baseline defaults | Same |
| Classifier | Spatial pooling, 1024 hidden units, 240 epochs | Same |
| Evaluation | Both branches, full and clean holdout | Same |
| Continuation, speckle, box calibration | None | None |

The fixed 8,453 / 939 split and same installed library environment are used for
both. The matched large job 330632 keeps its allocation; no DETR retry is submitted.
These are single-seed comparisons using different pretrained checkpoints, not
proof that one pretraining dataset is universally superior.
