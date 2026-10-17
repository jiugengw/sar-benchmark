"""Compare the two trained YOLO26 prediction branches on the saved holdout."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for branch in ['one2one', 'one2many']:
        detector = YOLO(str(args.weights))
        head = detector.model.model[-1]
        if head.cv2 is None or getattr(head, 'one2one_cv2', None) is None:
            raise RuntimeError('Checkpoint does not retain both trained branches')
        detector.model.end2end = branch == 'one2one'
        metrics = detector.val(
            data='data/yolo11s-sar-512/dataset.yaml', imgsz=512,
            batch=16, device=0, workers=4, conf=0.001, iou=0.7,
            max_det=300, plots=False, quantize=None, nms=branch == 'one2many',
            project=str(args.output.parent), name=branch,
        )
        results[branch] = {**{key: float(value) for key, value in metrics.results_dict.items()},
                           'per_class': metrics.summary()}
        args.output.write_text(json.dumps(results, indent=2, default=lambda value: value.item()) + '\n')
        print('BRANCH_RESULT', branch, results[branch], flush=True)
        del detector
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
