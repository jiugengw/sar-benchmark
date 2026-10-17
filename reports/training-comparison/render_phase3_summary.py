#!/usr/bin/env python3
"""Render shareable official-score plots and same-image E5/joint detections."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_comparison import diagnose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLORS = ("#a35f24", "#176b62")


def render_scores(audit: dict, baseline: dict, joint: dict) -> None:
    before, after = baseline["official_metrics"], joint["official_metrics"]
    reference = audit["target"]
    target = reference["delta_m"]
    def required_map(f1: float) -> float:
        return after["mAP"] * 100 + (target - after["delta_m"] -
               50 * (f1 / 100 - after["macro_f1"]) / reference["f1_reference"]) * reference["map_reference"] * 2

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), gridspec_kw={"height_ratios": [1, 1.3]})
    fig.suptitle("YOLO26m-P2: baseline vs joint fine-tuning", x=0.06, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.06, 0.925, "Official scores from 1,426 images · local diagnostics use a separate 939-image holdout", color="#5b6a72")
    ax = axes[0, 0]
    keys = ("macro_f1", "mAP", "AP50")
    x = np.arange(3)
    for offset, scores, label, color in ((-0.18, before, "100-epoch baseline", COLORS[0]),
                                          (0.18, after, "+15 joint epochs", COLORS[1])):
        bars = ax.bar(x + offset, [scores[key] * 100 for key in keys], 0.34, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set(xticks=x, xticklabels=["Macro-F1", "mAP (50–95)", "AP50"], ylim=(0, 105), ylabel="Official score (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Classification improved; detection traded off slightly", loc="left", fontsize=12)

    ax = axes[0, 1]
    ax.barh([1, 0], [before["delta_m"], after["delta_m"]], color=COLORS, height=0.45)
    ax.set(yticks=[1, 0], yticklabels=["Baseline", "Joint fine-tune"], xlim=(0, 34), xlabel="Official Δm")
    for y, scores in ((1, before), (0, after)):
        ax.text(scores["delta_m"] + 0.3, y, f"{scores['delta_m']:+.2f}", va="center", fontweight="bold")
    ax.axvline(target, color="#2376d1", linestyle="--")
    ax.text(target + 0.3, 0.8, f"Target +{target:g}", color="#2376d1", rotation=90)
    ax.set_title(f"{target - after['delta_m']:.2f} Δm points from the target", loc="left", fontsize=12)

    ax = axes[1, 0]
    frontier_x = np.linspace(88, 100, 120)
    frontier_y = np.array([required_map(value) for value in frontier_x])
    ax.plot(frontier_x, frontier_y, color=COLORS[1], lw=2.5, label=f"Δm +{target:g} frontier")
    ax.fill_between(frontier_x, frontier_y, 46, alpha=0.07, color=COLORS[1])
    for scores, label, color, offset in ((before, "Baseline", COLORS[0], (10, 12)),
                                         (after, "Joint fine-tune", COLORS[1], (10, -20))):
        ax.scatter(scores["macro_f1"] * 100, scores["mAP"] * 100, color=color, s=65, zorder=4)
        ax.annotate(label, (scores["macro_f1"] * 100, scores["mAP"] * 100), xytext=offset, textcoords="offset points", color=color)
    ax.scatter(95, required_map(95), color="#2376d1", s=70, zorder=4)
    ax.annotate(f"95% F1 + {required_map(95):.2f}% mAP", (95, required_map(95)), xytext=(-85, 20),
                textcoords="offset points", color="#2376d1", arrowprops={"arrowstyle": "-", "color": "#2376d1"})
    ax.set(xlim=(88, 100), ylim=(37, 45), xlabel="Official macro-F1 (%)", ylabel="Official mAP (%)")
    ax.grid(alpha=0.18)
    ax.set_title("40% mAP with 95% F1 would give only about Δm +27.44", loc="left", fontsize=11)

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["Training validator", f"{joint['training_metrics']['macro_f1']*100:.2f}%", f"{joint['training_metrics']['mAP']*100:.2f}%"],
        ["Exact local ZIP pipeline", f"{joint['metrics']['macro_f1']*100:.2f}%", f"{joint['metrics']['mAP']*100:.2f}%"],
        ["Official scoring", f"{after['macro_f1']*100:.2f}%", f"{after['mAP']*100:.2f}%"],
    ]
    table = ax.table(cellText=rows, colLabels=["Evaluation path", "Macro-F1", "mAP"], colWidths=[0.55, 0.225, 0.225],
                     bbox=[0, 0.53, 1, 0.4], cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), item in table.get_celld().items():
        item.set_edgecolor("#d7e0e3")
        if row == 0: item.set_facecolor("#e9f3f1")
    local_gap = 100 * (joint["training_metrics"]["mAP"] - joint["metrics"]["mAP"])
    split_gap = 100 * (joint["metrics"]["mAP"] - after["mAP"])
    f1_gap = 100 * (joint["metrics"]["macro_f1"] - after["macro_f1"])
    ax.text(0, 0.41, f"mAP gap: {local_gap:.2f} pp evaluation-path difference\n+ {split_gap:.2f} pp local-to-official split difference.", va="top", linespacing=1.7)
    ax.text(0, 0.22, f"F1 gap: {f1_gap:.2f} pp across splits. The exact local\nZIP pipeline reproduces the training confusion matrix.", va="top", linespacing=1.7)
    ax.text(0, 0.04, "This comparison alone does not establish overfitting.", color="#5b6a72", fontsize=10)
    ax.set_title("What accounts for the lower official result?", loc="left", fontsize=12)
    fig.subplots_adjust(top=0.87, left=0.07, right=0.97, bottom=0.07, hspace=0.4, wspace=0.3)
    fig.savefig(HERE / "phase3-official-comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


def render_detections(baseline: dict, joint: dict) -> None:
    truth = json.loads((ROOT / joint["ground_truth_path"]).read_text())
    categories = {row["id"]: row["name"] for row in truth["categories"]}
    annotations, by_class = defaultdict(list), defaultdict(list)
    for annotation in truth["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    for image in truth["images"]:
        if annotations[image["id"]]:
            by_class[categories[annotations[image["id"]][0]["category_id"]]].append(image)
    predictions = []
    for run in (baseline, joint):
        by_image = defaultdict(list)
        for prediction in json.loads((ROOT / run["predictions_path"]).read_text()):
            if prediction["score"] >= 0.3: by_image[prediction["image_id"]].append(prediction)
        predictions.append(by_image)
    names = ("bridge", "playground", "wind_turbine", "oil_tank", "ship", "aircraft")
    cell_width, image_height, row_height = 400, 360, 405
    canvas = Image.new("RGB", (1240, 125 + len(names) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    heading = ImageFont.truetype("DejaVuSans.ttf", 23)
    draw.text((20, 12), "Same-image detection review: E5 vs joint fine-tune", fill="#17232b", font=heading)
    draw.text((20, 47), "939-image local holdout · confidence ≥ 0.30 · IoU ≥ 0.50 · up to 20 prediction boxes shown", fill="#5b6a72", font=font)
    draw.text((20, 70), "Images selected by median object count within each class; these are not official evaluation images.", fill="#5b6a72", font=font)
    for column, title in enumerate(("Ground truth", "E5 · 100 epochs", "E5 + 15 joint epochs")):
        draw.text((20 + column * cell_width, 100), title, fill="#17232b", font=font)
    for row, name in enumerate(names):
        candidates = sorted(by_class[name], key=lambda image: (len(annotations[image["id"]]), image["id"]))
        image = candidates[len(candidates) // 2]
        image_truth = annotations[image["id"]]
        original = Image.open(ROOT / "data/course-data/train" / image["file_name"]).convert("RGB")
        scale = min((cell_width - 25) / image["width"], image_height / image["height"])
        for column in range(3):
            panel = original.resize((round(image["width"] * scale), round(image["height"] * scale)), Image.Resampling.BILINEAR)
            boxes = ImageDraw.Draw(panel)
            review = diagnose(image, image_truth, predictions[column - 1][image["id"]]) if column else None
            missed = set(review["missed_indices"]) if review else set()
            for index, annotation in enumerate(image_truth):
                x, y, w, h = annotation["bbox"]
                boxes.rectangle([x*scale, y*scale, (x+w)*scale, (y+h)*scale], outline="#ffae31" if index in missed else "#32e477", width=2)
            if review:
                for prediction in review["predictions"]:
                    x, y, w, h = prediction["bbox"]
                    color = "#43a4ff" if prediction["kind"] == "matched" else "#fa4b55"
                    boxes.rectangle([x*scale, y*scale, (x+w)*scale, (y+h)*scale], outline=color, width=2)
            x0, y0 = 20 + column * cell_width, 125 + row * row_height
            detail = f"{name} · image {image['id']} · {len(image_truth)} GT"
            if review: detail = f"matched {review['counts'].get('matched',0)} · missed {review['counts'].get('missed',0)}"
            draw.text((x0, y0), detail, fill="#17232b", font=font)
            canvas.paste(panel, (x0, y0 + 30))
    canvas.save(HERE / "phase3-detection-comparison.jpg", quality=92)


def main() -> None:
    audit = json.loads((HERE / "score-audit.json").read_text())
    baseline = next(run for run in audit["runs"] if run["job_id"] == "E5-yolo26m-100")
    joint = next(run for run in audit["runs"] if run["job_id"] == audit["target"]["job_id"])
    render_scores(audit, baseline, joint)
    render_detections(baseline, joint)
    print("Wrote phase3-official-comparison.png and phase3-detection-comparison.jpg")


if __name__ == "__main__":
    main()
