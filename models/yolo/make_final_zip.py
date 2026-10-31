#!/usr/bin/env python3
"""The same as `python models/yolo/make_submission.py --final ...`: make_submission.py now holds the five steps of our
final submission (see the top of that file). Kept so commands written before 31 Oct still work."""
import os
import sys
from pathlib import Path

script = Path(__file__).with_name('make_submission.py')
os.execv(sys.executable, [sys.executable, '-u', str(script), '--final', *sys.argv[1:]])
