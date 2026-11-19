"""Lay out the course data by the leak-free split, in exactly the official layout, so every script runs unchanged.

    python tools/leakfree/make_dataset.py results/leakfree/split.csv data/course-data OUT_DIR

OUT_DIR gets, like data/course-data:
  train/images/, train/instances.json, train/labels.csv
  val/images/, val/images.json          test/images/, test/images.json      (no labels, as the TA gave them)
  released-labels/{val,test}_instances.json, {val,test}_labels.csv, classes.json (for scripts/score_released.py)
  mapping.csv: new split, new file  <-  official split, official file
Images in each new split are numbered 1, 2, ... (images/00001.jpg, ...) in the order of split.csv (official train,
then val, then test, each by file name). The JPEG bytes are copied unchanged. Then tar the three folders on the node
(see sbatch/staging/leakfree_env.sh).
"""
import csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path

SPLITS = ('train', 'val', 'test')


def coco(data):
    """Official images and boxes: {(split, file): (image dict, [annotations])}, and the category list."""
    out, cats = {}, None
    for split, p in (('train', data / 'train' / 'instances.json'), ('val', data / 'released-labels' / 'val_instances.json'),
                     ('test', data / 'released-labels' / 'test_instances.json')):
        d = json.loads(p.read_text()); cats = cats or d['categories']
        assert d['categories'] == cats
        per = defaultdict(list)
        for a in d['annotations']: per[a['image_id']].append(a)
        for im in d['images']: out[(split, im['file_name'])] = (im, per[im['id']])
    return out, cats


def main(split_csv, data, out):
    data, out = Path(data), Path(out)
    rows = list(csv.DictReader(open(split_csv)))
    src, cats = coco(data)
    cat_index = {c['id']: k for k, c in enumerate(cats)}         # labels.csv label = category position (0-8)
    for s in SPLITS: (out / s / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'released-labels').mkdir(exist_ok=True)
    images, anns, labels, mapping, n, aid = defaultdict(list), defaultdict(list), defaultdict(list), [], defaultdict(int), defaultdict(int)
    for r in rows:
        off, file = r['official_split'], r['file'].split('/', 1)[1]          # file = '<split>/images/xxxxx.jpg'
        im, boxes = src[(off, file)]
        s = r['new_split']; n[s] += 1; i = n[s]
        new = f'images/{i:05d}{Path(file).suffix}'
        shutil.copyfile(data / off / file, out / s / new)
        images[s].append({'id': i, 'file_name': new, 'width': im['width'], 'height': im['height']})
        for a in boxes:
            aid[s] += 1; anns[s].append(dict(a, id=aid[s], image_id=i))
        cls = {a['category_id'] for a in boxes}; assert len(cls) == 1, (off, file)
        labels[s].append((i, new, cat_index[cls.pop()]))
        mapping.append((s, new, off, file))
    for s in SPLITS:
        inst = {'images': images[s], 'annotations': anns[s], 'categories': cats}
        lab = [('image_id', 'file_name', 'label')] + labels[s]
        if s == 'train':
            (out / s / 'instances.json').write_text(json.dumps(inst))
            with open(out / s / 'labels.csv', 'w', newline='') as fh: csv.writer(fh, lineterminator='\n').writerows(lab)
        else:
            (out / s / 'images.json').write_text(json.dumps({'images': images[s], 'categories': cats}))
            (out / 'released-labels' / f'{s}_instances.json').write_text(json.dumps(inst))
            with open(out / 'released-labels' / f'{s}_labels.csv', 'w', newline='') as fh: csv.writer(fh, lineterminator='\n').writerows(lab)
    if (data / 'released-labels' / 'classes.json').exists():
        shutil.copyfile(data / 'released-labels' / 'classes.json', out / 'released-labels' / 'classes.json')
    with open(out / 'mapping.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['new_split', 'new_file', 'official_split', 'official_file']); w.writerows(mapping)
    print(json.dumps({s: {'images': len(images[s]), 'boxes': len(anns[s])} for s in SPLITS}))


if __name__ == '__main__':
    main(*sys.argv[1:4])
