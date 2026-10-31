#!/usr/bin/env python3
"""job_m step 2: copy up to 25 Phase 1 images of each image size (60 in all, 224 to 512 px) into a separate folder
laid out the way Phase 2's images will come (images.json + the image files), and cut the uploaded ZIP down to those
images to compare with.

    python job_m_subset.py REFERENCE_ZIP OUT_DIR OUT_REFERENCE_ZIP      (from the repository root)
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

reference, out_dir, out_reference = map(Path, sys.argv[1:4])
source = Path('data/course-data/val')
index = json.loads((source / 'images.json').read_text())
by_id = sorted(index['images'], key=lambda im: int(im['id']))
picked = set()
for size in sorted({im['width'] for im in by_id}):
    same = [int(im['id']) for im in by_id if im['width'] == size]
    picked.update(same[::max(1, len(same) // 25)][:25])
images = [im for im in index['images'] if int(im['id']) in picked]  # the original order, as in the full run

out_dir.mkdir(parents=True)
for im in images:
    target = out_dir / im['file_name']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / im['file_name'], target)
(out_dir / 'images.json').write_text(json.dumps({**index, 'images': images}))

with zipfile.ZipFile(reference) as archive:
    boxes = json.loads(archive.read('detection.json'))
    rows = archive.read('classification.csv').decode().strip().splitlines()
kept = [b for b in boxes if b['image_id'] in picked]
with zipfile.ZipFile(out_reference, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('detection.json', json.dumps(kept, separators=(',', ':')))
    archive.writestr('classification.csv', '\n'.join([rows[0]] + [r for r in rows[1:] if int(r.split(',')[0]) in picked]) + '\n')
sizes = sorted({im['width'] for im in images})
print(f'{len(images)} images ({", ".join(f"{s} px" for s in sizes)}) copied to {out_dir}; '
      f'the reference keeps {len(kept)} of {len(boxes)} boxes', flush=True)
