# job_l_boost: can raising scores help the bridges and playgrounds found only below score 0.001? (1 Nov, user: "for
# these cant i jus make the score higher".) Refit 340514's holdout boxes, job_l.py's outcomes, scored as submitted.
#  raise all   every sub-0.001 bridge / playground box moved up to just above 0.001 (ranked right after the real
#              originals and hedged like them): what "make the score higher" can do without knowing which are right.
#  ORACLE      only the correct ones (job_l.py's "found low" boxes) raised to 0.5: needs the labels, so impossible;
#              it shows what a perfect re-ranking signal would be worth.
#   python job_l_boost.py
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo'); sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import GT
from job_k import LOW, Scorer, split, submitted
from job_l import CLASSES, outcome


def main():
    from sarbench.twins import clean_image_ids
    gt = json.load(open(GT)); clean = set(clean_image_ids(gt['images']))
    boxes = json.load(open(LOW)); by_image = defaultdict(list)
    for p in boxes:
        by_image[p['image_id']].append(p)
    right = set()
    for g in gt['annotations']:
        if g['category_id'] in CLASSES and g['image_id'] in clean:
            oc, _, p = outcome(g, by_image[g['image_id']])
            if oc == 'found low':
                right.add(id(p))
    for cls, name in CLASSES.items():
        low = [p for p in boxes if p['category_id'] == cls and p['score'] < 0.001 and p['image_id'] in clean]
        lo = min((p['score'] for p in low if id(p) in right), default=0.0)
        band = [p for p in low if p['score'] >= lo]
        hits = sum(id(p) in right for p in band)
        print(f'{name}: {len(band)} boxes on the clean holdout score between {lo:.1e} and 0.001; {hits} of them are the '
              f'correct "found low" ones ({100 * hits / max(1, len(band)):.1f}%)', flush=True)
    s = Scorer()
    raw, low = split(boxes)
    s.row('baseline: round + hedge + tail', submitted(raw, low, s.widths))
    up = [{**p, 'score': 0.001 + p['score']} for p in low if p['category_id'] in CLASSES]
    s.row('raise all low bridge/playground boxes', submitted(raw + up, [p for p in low if p['category_id'] not in CLASSES], s.widths))
    oracle = [{**p, 'score': 0.5} for p in low if id(p) in right]
    s.row('ORACLE: raise only the correct ones', submitted(raw + oracle, [p for p in low if id(p) not in right], s.widths))
    print('JOB_L_BOOST_DONE', flush=True)


if __name__ == '__main__':
    main()
