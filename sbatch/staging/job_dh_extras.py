# Put the density-boosted scores of the +65.80 boxes onto the +65.85 extras ZIP (its first boxes are those same boxes).
import json, sys, zipfile
base, boosted, extras, out = sys.argv[1:5]
read = lambda p: {n: zipfile.ZipFile(p).read(n) for n in zipfile.ZipFile(p).namelist()}
B, D, E = read(base), read(boosted), read(extras)
b, d, e = (json.loads(x['detection.json']) for x in (B, D, E))
n = len(b)
assert len(d) == n and e[:n] == b, 'extras ZIP does not start with the +65.80 boxes'
assert all({k: v for k, v in p.items() if k != 'score'} == {k: v for k, v in q.items() if k != 'score'} for p, q in zip(b, d))
assert B['classification.csv'] == D['classification.csv'] == E['classification.csv']
assert max(p['score'] for p in e[n:]) < 0.001
new = d + e[n:]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('detection.json', json.dumps(new)); z.writestr('classification.csv', E['classification.csv'])
print(out, len(new), 'boxes;', sum(p['score'] != q['score'] for p, q in zip(b, d)), 'scores changed')
