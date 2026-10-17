"""Audit paired holdout detections; diagnostic matches are not COCO AP."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from inspect_detections import diagnose, draw_example, iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--data', type=Path, default=Path('data/course-data'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        raise ValueError('Provide one unique label per run')
    gt = json.loads((args.runs[0] / 'instances_val_holdout.json').read_text())
    truth = defaultdict(list)
    categories = {c['id']: c['name'] for c in gt['categories']}
    for ann in gt['annotations']:
        if not ann.get('iscrowd', 0):
            truth[ann['image_id']].append(ann)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'paired-images').mkdir(exist_ok=True)
    report = {'images': len(gt['images']), 'objects': sum(map(len, truth.values())),
              'note': 'All comparisons use identical holdout annotations. Greedy matching is diagnostic, '
                      'not COCO AP. Small means area below 256 pixels after resizing to 512x512. '
                      'Per-GT best overlap can reuse a prediction and is coverage, not one-to-one recall.',
              'models': {}, 'paired_examples': []}
    per_image_reviews = {}
    for label, run in zip(args.labels, args.runs):
        other_gt = json.loads((run / 'instances_val_holdout.json').read_text())
        if other_gt != gt:
            raise ValueError(f'Holdout differs: {run}')
        metrics = json.loads((run / 'metrics.json').read_text())
        predictions = defaultdict(list)
        for pred in json.loads((run / 'predictions_val.json').read_text()):
            predictions[pred['image_id']].append(pred)
        model = {'run': str(run), 'selected_metrics': metrics['val'], 'diagnostics': {}}
        per_image_reviews[label] = {}
        for threshold in (0.05, 0.3):
            for match_iou in (0.5, 0.75):
                totals, by_class = Counter(), defaultdict(Counter)
                for image in gt['images']:
                    annotations = truth[image['id']]
                    review = diagnose(image, annotations, predictions[image['id']], threshold, match_iou)
                    totals.update(review['counts'])
                    for index, ann in enumerate(annotations):
                        counts = by_class[categories[ann['category_id']]]
                        small = ann['bbox'][2] * ann['bbox'][3] * 512**2 / (image['width'] * image['height']) < 256
                        counts['objects'] += 1
                        counts['small_objects'] += int(small)
                        missed = index in review['missed_indices']
                        counts['missed'] += int(missed)
                        counts['missed_small'] += int(small and missed)
                    for pred in review['predictions']:
                        by_class[categories[pred['category_id']]][pred['kind']] += 1
                    if threshold == 0.3 and match_iou == 0.5:
                        per_image_reviews[label][image['id']] = review
                model['diagnostics'][f'score_{threshold}_iou_{match_iou}'] = {
                    'total': dict(totals), 'by_class': {k: dict(v) for k, v in by_class.items()}}
        coverage = defaultdict(Counter)
        for image in gt['images']:
            candidates = [p for p in predictions[image['id']] if p['score'] >= 0.05]
            for ann in truth[image['id']]:
                best = max((iou(ann['bbox'], p['bbox']) for p in candidates
                            if p['category_id'] == ann['category_id']), default=0.)
                key = 'no_overlap_below_0.1' if best < .1 else 'loose_0.1_to_0.5' if best < .5 else 'overlap_at_least_0.5'
                coverage[categories[ann['category_id']]][key] += 1
        model['best_same_class_overlap_at_score_0.05'] = {k: dict(v) for k, v in coverage.items()}
        report['models'][label] = model
        print(f'Analyzed {label}', flush=True)

    # Include one image where the first model wins and one where the final model
    # wins per class when available. Selection is illustrative, not a prevalence estimate.
    for category in categories.values():
        ranked = []
        for image in gt['images']:
            annotations = truth[image['id']]
            if not annotations or categories[annotations[0]['category_id']] != category:
                continue
            counts = {label: per_image_reviews[label][image['id']]['counts'] for label in args.labels}
            difference = counts[args.labels[0]].get('matched', 0) - counts[args.labels[-1]].get('matched', 0)
            ranked.append((difference, image))
        if not ranked:
            continue
        selected = [max(ranked, key=lambda row: row[0]), min(ranked, key=lambda row: row[0])]
        used = set()
        for difference, image in selected:
            if image['id'] in used:
                continue
            used.add(image['id'])
            row = {'class': category, 'image_id': image['id'], 'matched_difference_first_minus_last': difference,
                   'counts': {}, 'overlays': {}}
            annotations = truth[image['id']]
            gt_name = f'{category}_{image["id"]}_truth.jpg'
            draw_example(args.data / 'train' / image['file_name'], annotations, None, categories,
                         args.out / 'paired-images' / gt_name)
            row['ground_truth_overlay'] = f'paired-images/{gt_name}'
            for label in args.labels:
                review = per_image_reviews[label][image['id']]
                filename = f'{category}_{image["id"]}_{label}.jpg'
                draw_example(args.data / 'train' / image['file_name'], annotations, review, categories,
                             args.out / 'paired-images' / filename)
                row['counts'][label] = review['counts']
                row['overlays'][label] = f'paired-images/{filename}'
            report['paired_examples'].append(row)
    (args.out / 'paired_error_analysis.json').write_text(json.dumps(report, indent=2))
    print(f'Saved {args.out / "paired_error_analysis.json"}', flush=True)


if __name__ == '__main__':
    main()
