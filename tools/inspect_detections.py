"""Compare saved detection predictions on a shared labeled holdout, without GPU inference."""
import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def diagnose(image, truth, predictions, threshold, match_iou):
    """Greedy class-aware matches for inspection; these counts are not COCO AP."""
    used, tagged = set(), []
    counts = Counter({'objects': len(truth)})
    small = {i for i, ann in enumerate(truth)
             if ann['bbox'][2] * ann['bbox'][3] * 512**2 / (image['width'] * image['height']) < 16**2}
    counts['small_objects'] = len(small)
    for pred in sorted(predictions, key=lambda p: p['score'], reverse=True):
        if pred['score'] < threshold:
            continue
        overlap = [iou(pred['bbox'], ann['bbox']) for ann in truth]
        same = [i for i, ann in enumerate(truth) if ann['category_id'] == pred['category_id']]
        available = [i for i in same if i not in used and overlap[i] >= match_iou]
        if available:
            best = max(available, key=lambda i: overlap[i])
            used.add(best)
            kind = 'matched'
        elif any(overlap[i] >= match_iou for i in same):
            kind = 'duplicate'
        elif any(value >= match_iou for value in overlap):
            kind = 'wrong_class'
        elif any(overlap[i] >= 0.1 for i in same):
            kind = 'localization'
        else:
            kind = 'background'
        counts[kind] += 1
        tagged.append({**pred, 'kind': kind, 'matched_gt_index': best if kind == 'matched' else None})
    missed = set(range(len(truth))) - used
    counts['missed'] = len(missed)
    counts['missed_small'] = len(missed & small)
    counts['matched_small'] = len(small & used)
    return {'counts': dict(counts), 'missed_indices': sorted(missed), 'predictions': tagged}


COLORS = {'matched': '#3b9dff', 'duplicate': '#ffd541', 'wrong_class': '#ff4455',
          'localization': '#ed62ff', 'background': '#ff4455'}


def draw_example(source, truth, review, categories, destination):
    # A missing review renders the annotations alone, all in green.
    if review is None:
        review = {'missed_indices': [], 'predictions': []}
    with Image.open(source) as original:
        original_size = original.size
        image = original.convert('RGB')
    image.thumbnail((600, 600))
    sx, sy = image.width / original_size[0], image.height / original_size[1]
    draw = ImageDraw.Draw(image)

    def box(bbox):
        x, y, width, height = bbox
        return (x * sx, y * sy, (x + width) * sx, (y + height) * sy)

    for index, ann in enumerate(truth):
        draw.rectangle(box(ann['bbox']), outline='#ff9b35' if index in review['missed_indices'] else '#57e874', width=2)
    # Limit displayed prediction labels to keep crowded examples readable; counts use every eligible prediction.
    for pred in review['predictions'][:20]:
        coordinates = box(pred['bbox'])
        color = COLORS[pred['kind']]
        draw.rectangle(coordinates, outline=color, width=2)
        text = f"{categories[pred['category_id']]} {pred['score']:.2f}"
        left, top = max(0, coordinates[0]), max(0, coordinates[1] - 12)
        bounds = draw.textbbox((left, top), text)
        draw.rectangle(bounds, fill='#101820')
        draw.text((left, top), text, fill=color)
    image.save(destination, quality=90)


def write_report(args):
    if not 0 <= args.score_threshold <= 1 or not 0 < args.iou <= 1 or args.examples < 1:
        raise ValueError('Require score threshold in [0,1], IoU in (0,1], and at least one example.')
    ground_truth = json.loads((args.runs[0] / 'instances_val_holdout.json').read_text())
    images = ground_truth['images']
    categories = {category['id']: category['name'] for category in ground_truth['categories']}
    annotations = defaultdict(list)
    for ann in ground_truth['annotations']:
        if not ann.get('iscrowd', 0):
            annotations[ann['image_id']].append(ann)
    models, totals, reviews, per_class, scores = [], {}, {}, {}, {}
    labels = getattr(args, 'labels', None)
    if labels is not None and len(labels) != len(args.runs):
        raise ValueError('--labels must contain one name per run.')
    for run_index, run in enumerate(args.runs):
        other_gt = json.loads((run / 'instances_val_holdout.json').read_text())
        if other_gt['images'] != images or other_gt['annotations'] != ground_truth['annotations']:
            raise ValueError(f'{run} does not use the same holdout and annotations.')
        name = labels[run_index] if labels else run.name
        if name in models:
            raise ValueError('Run directory names must be unique.')
        models.append(name)
        scores[name] = json.loads((run / 'metrics.json').read_text())['val']
        predictions = defaultdict(list)
        for pred in json.loads((run / 'predictions_val.json').read_text()):
            predictions[pred['image_id']].append(pred)
        totals[name], reviews[name] = Counter(), {}
        per_class[name] = defaultdict(Counter)
        for image in images:
            truth = annotations[image['id']]
            review = diagnose(image, truth, predictions[image['id']], args.score_threshold, args.iou)
            reviews[name][image['id']] = review
            totals[name].update(review['counts'])
            image_class = categories[truth[0]['category_id']] if truth else 'unlabeled'
            per_class[name][image_class].update(review['counts'])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'images').mkdir(exist_ok=True)
    summary = {'images': len(images), 'score_threshold': args.score_threshold, 'match_iou': args.iou,
               'small_definition': 'box area below 16x16 pixels after resizing to 512x512',
               'models': {name: {'total': dict(totals[name]), 'coco_per_class_AP': scores[name]['per_class_AP'],
                                 'by_class': {key: dict(value) for key, value in per_class[name].items()}}
                          for name in models}}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))

    # Include examples from every class, then fill with the largest remaining errors.
    def severity(image):
        return max((3 * reviews[name][image['id']]['counts'].get('missed_small', 0)
                    + reviews[name][image['id']]['counts'].get('missed', 0)
                    + 2 * reviews[name][image['id']]['counts'].get('wrong_class', 0)
                    + reviews[name][image['id']]['counts'].get('duplicate', 0)
                    + reviews[name][image['id']]['counts'].get('localization', 0))
                   / max(1, len(annotations[image['id']])) for name in models)
    ranked = sorted(images, key=severity, reverse=True)
    selected, class_counts = [], Counter()
    for image in ranked:
        truth = annotations[image['id']]
        category = categories[truth[0]['category_id']] if truth else 'unlabeled'
        if class_counts[category] < 3 and len(selected) < args.examples:
            selected.append(image)
            class_counts[category] += 1
    selected_ids = {image['id'] for image in selected}
    selected.extend(image for image in ranked if image['id'] not in selected_ids)
    selected = selected[:args.examples]
    cards = []
    for image in selected:
        truth = annotations[image['id']]
        category = categories[truth[0]['category_id']] if truth else 'unlabeled'
        flags = set()
        truth_filename = f'{image["id"]}_truth.jpg'
        draw_example(args.data / 'train' / image['file_name'], truth, None, categories,
                     args.out / 'images' / truth_filename)
        panels = [f'<div><h3>Ground truth only — correct boxes</h3>'
                  f'<p>Class: {html.escape(category)}; {len(truth)} annotated objects. All boxes are green.</p>'
                  f'<a href="images/{truth_filename}"><img src="images/{truth_filename}" '
                  f'alt="Ground truth for image {image["id"]}" loading="lazy"></a></div>']
        for model_index, name in enumerate(models):
            review = reviews[name][image['id']]
            counts = review['counts']
            flags.update(key for key in ('missed_small', 'missed', 'wrong_class', 'duplicate', 'localization')
                         if counts.get(key, 0))
            filename = f'{image["id"]}_{model_index}.jpg'
            draw_example(args.data / 'train' / image['file_name'], truth, review, categories, args.out / 'images' / filename)
            detail = ', '.join(f'{key}: {counts.get(key, 0)}' for key in ('matched', 'missed', 'wrong_class', 'duplicate', 'localization'))
            panels.append(f'<div><h3>{html.escape(name)}</h3><p>{detail}</p>'
                          f'<a href="images/{filename}"><img src="images/{filename}" loading="lazy"></a></div>')
        cards.append(f'<article data-class="{html.escape(category)}" data-errors="{" ".join(sorted(flags))}">'
                     f'<h2>Image {image["id"]}: {html.escape(category)} — {len(truth)} objects</h2>'
                     f'<p>{html.escape(image["file_name"])}</p><div class="panels">{"".join(panels)}</div></article>')
    fields = ('objects', 'matched', 'missed', 'missed_small', 'wrong_class', 'duplicate', 'localization', 'background')
    table = '<table><tr><th>Model</th>' + ''.join(f'<th>{key.replace("_", " ")}</th>' for key in fields) + '</tr>'
    for name in models:
        table += f'<tr><th>{html.escape(name)}</th>' + ''.join(f'<td>{totals[name][key]}</td>' for key in fields) + '</tr>'
    table += '</table>'
    ap_table = '<table><tr><th>Class</th>' + ''.join(f'<th>{html.escape(name)} AP (%)</th>' for name in models) + '</tr>'
    for category in categories.values():
        ap_table += f'<tr><th>{html.escape(category)}</th>' + ''.join(
            f'<td>{100 * scores[name]["per_class_AP"][category]:.2f}</td>' for name in models) + '</tr>'
    ap_table += '</table>'
    options = ''.join(f'<option>{html.escape(name)}</option>' for name in categories.values())
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAR detection review</title><style>
body{{font:16px system-ui;max-width:1320px;margin:30px auto;padding:0 18px;background:#101720;color:#e8edf5}}
p{{line-height:1.5}} table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border:1px solid #465266}}
article{{background:#1b2634;padding:20px;margin:22px 0;border-radius:10px}}.panels{{display:grid;grid-template-columns:repeat({len(models) + 1},minmax(0,1fr));gap:20px}}
img{{max-width:100%;height:auto}}h3{{font-size:16px}}select{{padding:9px;margin:8px}}a{{color:#8bc7ff}}
@media(max-width:750px){{.panels{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}}}
</style><h1>SAR detection review</h1>
<p>All {len(images)} labeled holdout images, confidence ≥ {args.score_threshold:g}, matching IoU ≥ {args.iou:g}.
Diagnostic counts use greedy class-aware matching; they are not COCO mAP. A missed object here may have a prediction below the selected confidence threshold.</p>
{table}<h2>Saved COCO AP by class</h2><p>These AP scores come from each run's full validation evaluation; the gallery confidence filter does not change them.</p>{ap_table}
<p>The first panel shows only the correct annotations, with every box in green. The model panels add predictions and mark annotation matches using the following legend.</p>
<p><b style="color:#57e874">Green: matched ground truth</b> · <b style="color:#ff9b35">Orange: unmatched ground truth</b> ·
<b style="color:#3b9dff">Blue: matching prediction</b> · <b style="color:#ffd541">Yellow: duplicate</b> ·
<b style="color:#ed62ff">Purple: inaccurate location</b> · <b style="color:#ff4455">Red: wrong class / background</b></p>
<p>Small = area below one 16×16 patch after resizing to 512×512. Every ground-truth box is drawn; at most 20 predictions are drawn per panel. Summary counts include all eligible predictions. Examples are selected for errors and class coverage.</p>
<label>Class <select id="class"><option value="">All</option>{options}</select></label>
<label>Error <select id="error"><option value="">All</option><option value="missed_small">Missed small objects</option>
<option value="missed">Missed objects</option><option value="wrong_class">Wrong class</option><option value="duplicate">Duplicates</option>
<option value="localization">Inaccurate boxes</option></select></label><span id="count"></span>
{''.join(cards)}<script>
function filter(){{let n=0;document.querySelectorAll('article').forEach(card=>{{const show=(!document.getElementById('class').value||card.dataset.class===document.getElementById('class').value)&&(!document.getElementById('error').value||card.dataset.errors.split(' ').includes(document.getElementById('error').value));card.hidden=!show;if(show)n++;}});document.getElementById('count').textContent=n+' examples';}}
document.querySelectorAll('select').forEach(select=>select.addEventListener('change',filter));filter();
</script></html>'''
    (args.out / 'index.html').write_text(page)
    print(json.dumps(summary, indent=2))
    print(f'Report: {args.out / "index.html"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', help='optional unique display names, one per run')
    parser.add_argument('--data', type=Path, default=Path('data/course-data'))
    parser.add_argument('--out', type=Path, default=Path('reports/detection-review'))
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--examples', type=int, default=48)
    write_report(parser.parse_args())


if __name__ == '__main__':
    main()
