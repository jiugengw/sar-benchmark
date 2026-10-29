#!/bin/bash
# ZIP for M stride 1 (training-split memorise model 337666): its own boxes on the 1,426 official validation images
# (same predictor settings as make_submission: conf 0.001, NMS 0.7, 300 boxes), image labels copied from the best
# upload (two-head soup classifier), then rounded and hedged like the 54.88 upload. CPU on the login node.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
D=nonsubmissions/241026/s1-memorise-337666; mkdir -p $D /tmp/s1zip
python models/yolo/raw_candidates.py --weights runs/ft40-s1r-memorise-337666/train/weights/best.pt --images data/course-data/val/images.json --split-root data/course-data/val --out /tmp/s1zip/detection.json --iou 0.7 --max-det 300 --device cpu
unzip -o -q submissions/alldata-soup6-rounded-hedged-2head/alldata-soup6-rounded-hedged-2head.zip classification.csv -d /tmp/s1zip
(cd /tmp/s1zip && rm -f s1.zip && zip -q s1.zip detection.json classification.csv)
cp /tmp/s1zip/s1.zip $D/s1-memorise-337666.zip
python models/yolo/round_boxes.py zip $D/s1-memorise-337666.zip $D/s1-memorise-337666-rounded.zip --images data/course-data/val/images.json
python models/yolo/hedge_study.py zip $D/s1-memorise-337666-rounded.zip $D/s1-memorise-337666-rounded-hedged.zip
ls -la $D; echo S1_ZIP_READY
