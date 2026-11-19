"""Pack every labelled course image (train, val, test) into one zip for the copy search, with one id per image.

The searches (pixsearch.py, scalesearch.py from the leakage audit) read images straight from a zip laid out as
course/images/<split>/<id>.<ext>. Ids run over train, then val, then test, each sorted by file name. The original
JPEG bytes are stored unchanged.

    python tools/leakfree/pack.py data/course-data OUT_DIR
    -> OUT_DIR/course.zip, OUT_DIR/ids.csv (id, split, file_name, width, height)
"""
import csv, json, sys, zipfile
from pathlib import Path

from PIL import Image


def main(data, out):
    data, out = Path(data), Path(out); out.mkdir(parents=True, exist_ok=True)
    rows, i = [], 0
    with zipfile.ZipFile(out / 'course.zip', 'w', zipfile.ZIP_STORED) as z:
        for split in ('train', 'val', 'test'):
            for f in sorted((data / split / 'images').iterdir()):
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'): continue
                i += 1
                with Image.open(f) as im: w, h = im.size
                zi = zipfile.ZipInfo(f'course/images/{split}/{i}{f.suffix.lower()}', date_time=(1980, 1, 1, 0, 0, 0))
                z.writestr(zi, f.read_bytes())   # fixed date: some files are dated before 1980, which zip cannot store
                rows.append((i, split, f'images/{f.name}', w, h))
    with open(out / 'ids.csv', 'w', newline='') as fh:
        wr = csv.writer(fh, lineterminator='\n'); wr.writerow(['id', 'split', 'file_name', 'width', 'height']); wr.writerows(rows)
    print(json.dumps({s: sum(r[1] == s for r in rows) for s in ('train', 'val', 'test')}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
