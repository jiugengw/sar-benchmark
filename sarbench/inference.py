"""Optional single-model confidence calibration shared by evaluation and submission."""
import math


def calibrate_detections(detections, logits, power=0.0):
    if not math.isfinite(power) or power < 0:
        raise ValueError('Classifier score power must be finite and nonnegative')
    if power == 0:
        return detections
    probabilities = logits.float().softmax(-1)
    return [{**d, 'scores': d['scores'] * probabilities[i][d['labels'] - 1].pow(power)}
            for i, d in enumerate(detections)]
