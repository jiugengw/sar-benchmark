# Final model weights

The model we submit (Phase 1 and Phase 2): **the flip soup without refits, Δm +65.80** (mAP 59.94, macro-F1 95.40,
[`submissions/alldata-s1fs-soup6-rounded-hedged-s1two-tail/`](../submissions/alldata-s1fs-soup6-rounded-hedged-s1two-tail/)).
The weights are too large for git, so they are attached to the GitHub release
[`model-flipsoup-twohead`](https://github.com/jiugengw/sar-benchmark/releases/tag/model-flipsoup-twohead).

| File | Size | What it is | Trained as |
|---|---:|---|---|
| `detector.pt` | 43 MB | YOLO26m-P2 detector, stride-1 first layer: six flip-memorise runs averaged into one model | the flip soup: `runs/soup-s1fs/alldata-s1fs-six.pt` |
| `branch_head.pt` | 41 MB | image classifier, branch head | `runs/soup-s1fs/heads-339995/branch_head.pt` |
| `conv_head.pt` | 8 MB | image classifier, conv head | `runs/soup-s1fs/heads-339995/conv_head.pt` |

How each one was trained: [docs/TEAM_GUIDE.md](../docs/TEAM_GUIDE.md) section 2 and
[docs/BEST_MODEL_RECIPE.md](../docs/BEST_MODEL_RECIPE.md).

**Why not the refits.** Refit 2 (19 more flip epochs on this soup) scored +65.92, only +0.12 more, within run-to-run
noise, and the refits only target the test's flipped copies of training images. From 30 Oct we submit the simpler
model. Refit 2's weights stay on the release
[`model-refit2-twohead`](https://github.com/jiugengw/sar-benchmark/releases/tag/model-refit2-twohead).

## Download and check

From the repository root:

```bash
gh release download model-flipsoup-twohead --pattern '*.pt' --dir final_model
(cd final_model && shasum -a 256 -c SHA256SUMS)
```

Without the `gh` command, download the three `.pt` files from the release page into this folder. The check must
print `OK` for all three.

## Make the ZIP

```bash
python models/yolo/make_final_zip.py --out final.zip
```

See [docs/TEAM_GUIDE.md](../docs/TEAM_GUIDE.md#0-replicate-our-submission-no-training) for the setup it needs, how to
run it on new images, and how to check the result against our uploaded ZIP.
