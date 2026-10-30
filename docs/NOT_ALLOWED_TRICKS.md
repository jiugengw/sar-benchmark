# Tricks we thought of that aren't allowed

Ideas that raised the Codabench score but aren't allowed under the course rules. We tried them as probes only:
none was posted to the leaderboard, and none is part of our method or our final model. Our best allowed result
stays **Δm +65.92** (mAP 60.04, F1 95.24; [submissions/README.md](../submissions/README.md)).

## Label transfer from the most similar labelled image (30 October)

**The idea.** Many test images repeat a labelled training scene. Of the 1,426 Phase 1 test images, 341 are exact
copies of a labelled image (similarity ≥ 0.98, often flipped), and 342 are near copies of the same place
(0.45 to 0.98). Airports, wind turbines, playgrounds and bridges don't move between images, so on these images
the labelled boxes beat the network's. So, after the network has run:

1. Compare each test image with every labelled image by pixel correlation. The labelled images are tried in four
   orientations (as is and three flips, using the dataset's 1–2 px flip offset); the test image is never changed.
2. If the best match is above a cut-off and is an airport, wind turbine, playground or bridge image, replace the
   network's boxes of that class with the labelled image's boxes, at score 1.0.

**Why it isn't allowed.**
- It copies training labels onto test images instead of predicting them.
- It needs the labelled images at test time: 2.0 B values at full size, far over the 130 M parameter limit.
  Stored at 64 × 64 ("version B"), they're 38.6 M.
- Each test image is read a second time, outside the network, to find its match.

**Results.** Each probe starts from the +65.92 ZIP (refit 2 + two-head classifier + low-score tail). Scores are
from Codabench, not posted.

| ZIP | Variant | Matched test images | Images given copied boxes | mAP | F1 | Δm |
|---|---|---:|---:|---:|---:|---:|
| [transferfull-all](../submissions/alldata-s1fsr2-tail-transferfull-all-not-allowed/) | Full-size match, similarity 0.45–1.0 (near and exact copies) | 683 | 446 | 64.06 | 95.24 | +73.61 |
| [transferfull-first](../submissions/alldata-s1fsr2-tail-transferfull-first-not-allowed/) | As above, but matching first: matched images take the match's boxes and class and skip the network | 683 | 446 | 63.82 | 94.94 | +72.99 |
| [transfer-all-replace](../submissions/alldata-s1fsr2-tail-transfer-all-replace-not-allowed/) | 64 px thumbnail match, 0.6–1.0 | 760 | 441 | 63.58 | 95.24 | +72.70 |
| [transferfull-near](../submissions/alldata-s1fsr2-tail-transferfull-near-not-allowed/) | Full-size match, 0.45–0.98 (near copies only) | 342 | 179 | 63.46 | 95.24 | +72.47 |
| [transferb](../submissions/alldata-s1fsr2-tail-transferb-not-allowed/) | Version B: labelled images stored at 64 × 64, enlarged back to the test image's size when comparing; cut-off 0.485 | 683 | 416 | 62.95 | 95.24 | +71.49 |
| [transferb-first](../submissions/alldata-s1fsr2-tail-transferb-first-not-allowed/) | Version B, matching first | 683 | 416 | 62.57 | 94.65 | +70.45 |
| [transfer-near-replace](../submissions/alldata-s1fsr2-tail-transfer-near-replace-not-allowed/) | 64 px thumbnail match, 0.6–0.9 (near copies only) | 281 | 105 | 62.39 | 95.24 | +70.42 |

A first version that added the copied boxes and kept the network's ones too was never uploaded
([`nonsubmissions/301026/transfer/alldata-s1fsr2-tail-transfer-not-allowed.zip`](../nonsubmissions/301026/transfer/alldata-s1fsr2-tail-transfer-not-allowed.zip)).

**What we learned.**
- Most of the gain came from near copies: +6.55 Δm from near copies alone, then +1.14 more from the exact ones. The
  network had already memorised the exact copies.
- Matching first scored lower. When the match's class and the classifier disagreed (8–10 images), the match was
  usually wrong, for example an aircraft or oil-tank image matched to a bridge.
- On new scenes the trick does nothing, so it wouldn't carry over to images that repeat no labelled scene.

**Code and diagrams** (kept as a record):
- [`sbatch/staging/transfer.py`](../sbatch/staging/transfer.py): 64 px thumbnail match.
- [`sbatch/staging/transfer_full.py`](../sbatch/staging/transfer_full.py): full-size match.
- [`sbatch/staging/job_b.py`](../sbatch/staging/job_b.py): version B.
- [`sbatch/staging/transfer_cascade.py`](../sbatch/staging/transfer_cascade.py): matching first.
- [`nonsubmissions/301026/ta/`](../nonsubmissions/301026/ta/): diagrams of each version.
