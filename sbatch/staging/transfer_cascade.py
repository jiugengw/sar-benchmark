# Cascade form of label transfer (30 Oct; NOT ALLOWED, kept as a record). Matcher first: a test image whose
# best full-size match (similarity 0.45-1.0) is a labelled image of a static class (airport, wind turbine, playground,
# bridge) gets only that image's boxes and its class; every other image keeps the detector + classifier output. So
# each image's predictions come from one predictor, never merged. Built from transfer_full.py's ZIP, whose copied
# boxes are the only ones at score 1.0 (the detector's top score is 0.98).
#   python transfer_cascade.py TRANSFER.zip OUT.zip
import json, sys, zipfile

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src); boxes = json.loads(z.read('detection.json'))
copied = {b['image_id']: b['category_id'] for b in boxes if b['score'] >= 1.0}
out = [b for b in boxes if b['image_id'] not in copied or b['score'] >= 1.0]
rows = z.read('classification.csv').decode().splitlines()
changed = 0
for n, row in enumerate(rows[1:], 1):
    i, label = map(int, row.split(','))
    if i in copied and label != copied[i] - 1:  # classification labels are category ids - 1
        rows[n] = f'{i},{copied[i] - 1}'; changed += 1
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    o.writestr('detection.json', json.dumps(out)); o.writestr('classification.csv', '\n'.join(rows) + '\n')
print(f'{dst}: {len(copied)} images answered by the matcher alone ({len(boxes) - len(out)} detector boxes dropped, '
      f'{changed} class labels changed); {len(out)} boxes')
