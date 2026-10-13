"""Run training, then generate a submission ZIP only if training succeeds."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args
    if training_args and training_args[0] == '--':
        training_args = training_args[1:]
    result = subprocess.run([sys.executable, '-u', 'train.py', *training_args])
    if result.returncode:
        return result.returncode
    result = subprocess.run([
        sys.executable, '-u', 'make_submission.py', '--run', str(args.checkpoint_dir),
        '--data', args.data, '--split', 'val', '--out', str(args.submission), '--workers', '4',
    ])
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
