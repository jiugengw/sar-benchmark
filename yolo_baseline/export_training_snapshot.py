"""Fit the fixed spatial-head recipe on a private copy of a still-training YOLO checkpoint."""
import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-batch', type=int, default=4)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / 'detector_snapshot.pt'
    for _attempt in range(30):
        try:
            if time.time() - args.weights.stat().st_mtime > 5:
                shutil.copy2(args.weights, snapshot)
                break
        except FileNotFoundError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError('No stable trained checkpoint was available')
    from yolo_baseline.deadline_recovery import candidate, write_json
    result = candidate(snapshot, 'snapshot', args.output, args.output.name, feature_batch=args.feature_batch)
    write_json(args.output / 'complete.json', result)


if __name__ == '__main__':
    main()
