"""Run the leakage audit's copy searches on the course images (radar only).

pixsearch.py (same-zoom copies: exact, flipped, shifted, overlapping) and scalesearch.py (copies at zoom 1.1-2, e.g. a
256 px chip cut from a 512 px tile) shortlist on an optical image and check the SAR one. The course data has radar only,
so the radar image stands in for both: r_opt and r_sar are then the same radar correlation, once at the shortlist
alignment and once checked again. Every image is resized to 512 px as the searches expect (a 256 px image of the same
place at the same pixel size becomes a zoom-2 copy, which scalesearch.py finds). scalesearch.py runs with ALL_PAIRS=1:
every image against every other, so train-train and val-test links are found too (needed to group images for the split).

    python coursesearch.py pix|scale course.zip OUT_DIR       (SEARCH_DIR: folder holding pixsearch.py, scalesearch.py)
"""
import os, sys, zipfile

sys.path.insert(0, os.environ.get('SEARCH_DIR', os.path.dirname(os.path.abspath(__file__))))
import pixsearch as px


def index(zpath):
    m = {}
    for x in zipfile.ZipFile(zpath).namelist():
        p = x.split('/')
        if len(p) == 4 and p[1] == 'images' and '.' in p[3]:
            m[int(p[3].split('.')[0])] = {'optical': x, 'sar': x, 'split': p[2]}
    return m


px.index = index

if __name__ == '__main__':
    mode, zpath, out = sys.argv[1:4]
    if mode == 'pix':
        px.main(zpath, out)
    else:
        os.environ.setdefault('ALL_PAIRS', '1')
        import scalesearch as ss
        ss.main(zpath, out)
