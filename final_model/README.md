# Final model weights

The model behind our Phase 1 result, **Δm +65.92** (mAP 60.04, macro-F1 95.24,
[`submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/`](../submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/)).
The weights are too large for git, so they are attached to the GitHub release
[`model-refit2-twohead`](https://github.com/jiugengw/sar-benchmark/releases/tag/model-refit2-twohead).

| File | Size | What it is | Trained as |
|---|---:|---|---|
| `detector.pt` | 43 MB | YOLO26m-P2 detector, stride-1 first layer | refit 2: `runs/ft40-s1adfsr2-340754/train/weights/last.pt` |
| `branch_head.pt` | 41 MB | image classifier, branch head | `runs/soup-s1fs/fliprefit-heads-341248/branch_head.pt` |
| `conv_head.pt` | 8 MB | image classifier, conv head | `runs/soup-s1fs/fliprefit-heads-341248/conv_head.pt` |

How each one was trained: [docs/TEAM_GUIDE.md](../docs/TEAM_GUIDE.md) section 2 and
[docs/BEST_MODEL_RECIPE.md](../docs/BEST_MODEL_RECIPE.md).

## Download and check

From the repository root:

```bash
gh release download model-refit2-twohead --pattern '*.pt' --dir final_model
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
