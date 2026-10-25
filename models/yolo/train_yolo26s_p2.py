#!/usr/bin/env python3
"""Train a scale of YOLO26 with a new stride-4 P2 detection level."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_fingerprint(model) -> str:
    """Fingerprint parameter values before any optimizer step, independent of device."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preserve_initialization(model, audit_path: Path) -> dict:
    """Register transferred weights with the wrapper and verify the trainer receives them.

    Layer state_dict loads alone leave a YAML wrapper without a checkpoint marker.
    The public load() API marks our prepared model as the initialization. Training
    must also use pretrained=True, or the wrapper explicitly discards those values.
    """
    expected = parameter_fingerprint(model.model)
    model.load(model.model)
    if not model.ckpt or parameter_fingerprint(model.model) != expected:
        raise RuntimeError('Could not register the transferred P2 model as training initialization')
    audit = {'prepared_parameters_sha256': expected, 'training_handoff_verified': False}
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')

    def verify(trainer):
        actual = parameter_fingerprint(trainer.model)
        audit.update(trainer_parameters_sha256=actual, training_handoff_verified=actual == expected)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        if actual != expected:
            raise RuntimeError('Trainer discarded or changed initialization before the first optimizer step')
        print(f'INITIALIZATION_HANDOFF_VERIFIED {actual}', flush=True)

    model.add_callback('on_pretrain_routine_end', verify)
    return audit


def transfer_standard_weights_to_p2(standard_model, p2_model, p4_attention=False, p1_level=False,
                                    p6_level=False) -> None:
    """Transfer semantically matching YOLO26 layers and P3-P5 detection heads."""
    source = standard_model.model
    target = p2_model.model

    # The new P2 branch occupies target layers 17-22, shifting the P4/P5
    # neck layers six positions later than in the standard YOLO26 model.
    # With the P4 attention block at target layer 14, every layer from 14 on moves one further, and the new block
    # starts as a copy of the backbone's attention block (layer 10) where the shapes match.
    shift = 1 if p4_attention else 0
    layer_pairs = [(index, index) for index in range(14)]
    layer_pairs.extend((source_index, source_index + shift) for source_index in range(14, 17))
    layer_pairs.extend((source_index, source_index + 6 + shift) for source_index in range(17, 23))
    if p4_attention:
        layer_pairs.append((10, P4_ATTENTION_INDEX))
    for source_index, target_index in layer_pairs:
        source_layer = source.model[source_index]
        target_layer = target.model[target_index]
        source_state = source_layer.state_dict()
        target_state = target_layer.state_dict()
        transferable = {
            key: value for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        target_layer.load_state_dict(transferable, strict=False)

    source_detect = source.model[-1]
    target_detect = target.model[-1]
    for attribute in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        source_layers = getattr(source_detect, attribute, None)
        target_layers = getattr(target_detect, attribute, None)
        if source_layers is None or target_layers is None:
            continue
        levels = 4 + p1_level + p6_level
        if len(source_layers) != 3 or len(target_layers) != levels:
            raise RuntimeError(
                f"Unexpected {attribute} levels: {len(source_layers)} -> {len(target_layers)}"
            )
        first_p3 = 2 if p1_level else 1  # the pretrained P3-P5 heads; P1 / P2 / P6 start fresh
        for source_index, target_index in enumerate(range(first_p3, first_p3 + 3)):
            source_module = source_layers[source_index]
            target_module = target_layers[target_index]
            source_state = source_module.state_dict()
            target_state = target_module.state_dict()
            transferable = {
                key: value for key, value in source_state.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            target_module.load_state_dict(transferable, strict=False)

    if hasattr(source_detect, "dfl") and hasattr(target_detect, "dfl"):
        target_detect.dfl.load_state_dict(source_detect.dfl.state_dict(), strict=False)


P4_ATTENTION_INDEX = 14


def add_p1_level(config: dict) -> dict:
    """Add a stride-2 detection level (P1) below P2, leaving every pretrained layer and grid unchanged.

    Stride 1 (a full-resolution first layer) helped small objects in native 512 px images but hurt large ones in the
    upscaled 256 px images, where every later layer then covers a quarter of the original image. Here only a new
    branch runs at 2 px per cell: the top-down P2 output (layer 19) is upsampled, joined with the first layer's
    stride-2 features and refined, then handed to the detection head as a fifth, finest level. Layers 0-28 and the
    P2-P5 levels are exactly the YOLO26-P2 ones; the new layers are appended, so no index moves."""
    config = copy.deepcopy(config)
    head = config["head"]
    detect = head[-1]
    if detect[2] != "Detect" or len(detect[0]) != 4 or config["backbone"][0][2] != "Conv":
        raise RuntimeError(f"Unexpected YOLO26-P2 layout: first layer {config['backbone'][0]}, detect {detect}")
    p2_output, channels = detect[0][0], head[detect[0][0] - len(config["backbone"])][3][0]
    head[-1:] = [
        [p2_output, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 0], 1, "Concat", [1]],
        [-1, 2, "C3k2", [channels, True]],
    ]
    p1_output = len(config["backbone"]) + len(head) - 1
    head.append([[p1_output, *detect[0]], *detect[1:]])
    return config


def add_p6_level(config: dict) -> dict:
    """Add a stride-64 detection level (P6) above P5: one more halving of the P5 output, refined, as a fifth level.

    With a stride-1 first layer every grid is twice as fine, so the coarsest level covers 16 px per cell instead of
    32, and large objects (bridges and harbours in upscaled 256 px images) lose the wide view they had. P6 brings a
    32 px-per-cell grid back while the finer levels keep stride 1's detail. Layers 0-28 and the P2-P5 levels are the
    YOLO26-P2 ones; the new layers are appended, so no index moves."""
    config = copy.deepcopy(config)
    head = config["head"]
    detect = head[-1]
    if detect[2] != "Detect" or len(detect[0]) != 4:
        raise RuntimeError(f"Unexpected YOLO26-P2 detect layer {detect}")
    p5_output = detect[0][-1]
    channels = head[p5_output - len(config["backbone"])][3][0]
    head[-1:] = [
        [p5_output, 1, "Conv", [channels, 3, 2]],
        [-1, 2, "C3k2", [channels, True]],
    ]
    p6_output = len(config["backbone"]) + len(head) - 1
    head.append([[*detect[0], p6_output], *detect[1:]])
    return config


def add_p4_attention(config: dict) -> dict:
    """Insert a C2PSA attention block after the top-down P4 layer (13), as YOLO26 has at P5 (layer 10).

    P4 is a 32 x 32 grid on a 512 image, so every position attends to the whole scene (rivers, roads, fields) at a
    resolution where those are shapes, and the result flows down into P3 and P2 and into the bottom-up P4 output.
    Absolute layer references from 14 on move one place; the bottom-up concat with layer 13 takes the block instead."""
    config = copy.deepcopy(config)
    backbone_size = len(config["backbone"])
    position = P4_ATTENTION_INDEX - backbone_size
    head = config["head"]
    if head[position - 1][2] != "C3k2" or head[position][2] != "nn.Upsample":
        raise RuntimeError(f"Unexpected YOLO26-P2 head layout around layer {P4_ATTENTION_INDEX}: {head[position - 1:position + 1]}")
    channels = head[position - 1][3][0]

    def move(reference):
        if isinstance(reference, list):
            return [move(r) for r in reference]
        if reference == P4_ATTENTION_INDEX - 1:
            return P4_ATTENTION_INDEX
        return reference + 1 if reference >= P4_ATTENTION_INDEX else reference

    for layer in head[position:]:
        layer[0] = move(layer[0])
    head.insert(position, [-1, 2, "C2PSA", [channels]])
    return config


def build_p2_model(source, scale, p2_yaml, detect_yaml, class_names, output, reg_max=None, p4_attention=False,
                   stem_stride=2, spd_stem=False, p1_level=False, p6_level=False):
    """YOLO26-P2 for our classes, initialised from a pretrained YOLO26 of any task.

    A detection checkpoint (COCO) is first loaded into the standard detection model. Another task's checkpoint, such
    as yolo26m-obb.pt (rotated boxes on DOTA aerial photos), goes into the standard detection architecture instead:
    every layer with the same name and shape transfers (backbone, neck, box branches); its class and angle outputs do
    not. The P2 model then takes the shared layers and the P3-P5 heads, as before.

    reg_max 16 swaps YOLO26's one-number box edges for YOLO11's distribution over 16 positions per edge (DFL); the last
    layer of each box branch then changes shape and starts from fresh weights, everything else transfers.

    p4_attention adds a second attention block at P4 (see add_p4_attention).

    p1_level adds a fifth, stride-2 detection level below P2 (see add_p1_level); P2-P5 stay as they are.

    p6_level adds a fifth detection level above P5 (see add_p6_level), meant for stem_stride 1.

    stem_stride 1 keeps the first convolution at full resolution, so every later grid is twice as fine (the finest
    detection level becomes 2 px per cell instead of 4) on the same 512 px input; no weight changes shape.

    spd_stem replaces the first stride-2 convolution with space-to-depth + a stride-1 convolution (spd.py): every pixel
    is kept, the grids stay as usual, and the layer starts as an exact copy of the pretrained one."""
    from ultralytics import YOLO

    if source.task == "detect":
        standard_config = copy.deepcopy(source.model.yaml)
    else:
        standard_config = yaml.safe_load(detect_yaml.read_text())
        standard_config["scale"] = scale
    standard_config["nc"] = len(class_names)
    standard_config_path = output / f"yolo26{scale}_9class.yaml"
    standard_config_path.write_text(yaml.safe_dump(standard_config, sort_keys=False))
    standard_model = YOLO(str(standard_config_path), task="detect")
    standard_model.model.names = dict(enumerate(class_names))
    standard_model.model.load(source.model, verbose=True)

    p2_config = yaml.safe_load(p2_yaml.read_text())
    p2_config["nc"] = len(class_names)
    p2_config["scale"] = scale
    if reg_max is not None:
        p2_config["reg_max"] = reg_max
    if p4_attention:
        p2_config = add_p4_attention(p2_config)
    if p1_level:
        if p4_attention:
            raise RuntimeError("p1_level and p4_attention together are not supported")
        p2_config = add_p1_level(p2_config)
    if p6_level:
        if p4_attention or p1_level:
            raise RuntimeError("p6_level with p4_attention or p1_level is not supported")
        p2_config = add_p6_level(p2_config)
    if stem_stride != 2:
        stem = p2_config["backbone"][0]
        if stem[2] != "Conv" or stem[3][2] != 2:
            raise RuntimeError(f"Unexpected first layer {stem}")
        stem[3][2] = stem_stride
    if spd_stem:
        import spd

        stem = p2_config["backbone"][0]
        if stem[2] != "Conv" or stem[3][1:3] != [3, 2]:
            raise RuntimeError(f"Unexpected first layer {stem}")
        stem[3][2] = -2  # SPDConv: space-to-depth + stride-1 conv
        spd.install()  # the trainer rebuilds the model from this YAML, so the name 'Conv' must stay patched
    p2_config_path = output / f"yolo26{scale}_p2.yaml"
    p2_config_path.write_text(yaml.safe_dump(p2_config, sort_keys=False))
    model = YOLO(str(p2_config_path), task="detect")
    model.model.names = dict(enumerate(class_names))
    transfer_standard_weights_to_p2(standard_model, model, p4_attention, p1_level, p6_level)
    if spd_stem:
        import spd

        spd.exact_copy(model.model.model[0], standard_model.model.model[0])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("n", "s", "m", "l"), default="s")
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--p2-yaml",
        type=Path,
        default=Path(".venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-p2.yaml"),
    )
    parser.add_argument(
        "--detect-yaml",
        type=Path,
        default=Path(".venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26.yaml"),
        help="detection architecture used when --weights is another task, e.g. yolo26m-obb.pt (DOTA aerial photos)",
    )
    parser.add_argument("--classes", type=Path, default=Path("data/yolo11s-sar-512/conversion_manifest.json"))
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flipud", type=float, default=0.0,
                        help="vertical-flip probability; overhead images have no up, so 0.5 is a free augmentation")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="left-right flip probability; 0 keeps the radar's fixed shadow side (Ultralytics default 0.5)")
    parser.add_argument("--reg-max", type=int,
                        help="box-edge bins: 1 is YOLO26's direct regression (the YAML default), 16 is YOLO11's DFL")
    parser.add_argument("--p4-attention", action="store_true",
                        help="add a whole-image attention block at P4, feeding P3 and P2 with scene context")
    parser.add_argument("--close-mosaic", type=int, default=10, help="turn mosaic off for the last N epochs")
    parser.add_argument("--spd-stem", action="store_true",
                        help="first layer as space-to-depth + stride-1 conv (keeps every pixel), from the pretrained layer")
    parser.add_argument("--p1-level", action="store_true",
                        help="add a stride-2 detection level below P2; the pretrained layers and grids stay unchanged")
    parser.add_argument("--p6-level", action="store_true",
                        help="add a detection level above P5 (with --stem-stride 1 it restores the 32 px-per-cell grid)")
    parser.add_argument("--lr0", type=float,
                        help="starting learning rate with MuSGD (default: Ultralytics' automatic choice, MuSGD 0.01 here)")
    parser.add_argument("--seed", type=int, default=0, help="0 reproduces the verified runs; others measure seed noise")
    parser.add_argument("--stem-stride", type=int, choices=(1, 2), default=2,
                        help="1: first convolution at full resolution, every grid twice as fine (about 4x compute)")
    parser.add_argument("--data", type=Path, default=Path("data/yolo11s-sar-512/dataset.yaml"),
                        help="dataset YAML; a variant from build_variant_dataset.py for the context-box or filter experiments")
    parser.add_argument("--no-colour-jitter", action="store_true",
                        help="hue and saturation jitter off (needed when the 3 channels carry different filters)")
    args = parser.parse_args()
    if args.weights is None:
        args.weights = Path(f"pretrained/yolo26{args.scale}.pt")

    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("YOLO training requires the allocated CUDA GPU")
    if not args.p2_yaml.is_file():
        raise FileNotFoundError(args.p2_yaml)
    class_names = json.loads(args.classes.read_text())["classes"]
    if len(class_names) != 9:
        raise ValueError(f"Expected the nine original classes, found {class_names}")
    args.train_output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.weights.is_file():
        source = YOLO(str(args.weights))
    elif args.scale == "n" and args.weights.name == "yolo26n.pt":
        # YOLO26n is available from Ultralytics but is not staged in this repo.
        # Resolve the official asset once and keep a local copy for evaluation.
        source = YOLO("yolo26n.pt")
        downloaded_weights = Path(source.ckpt_path)
        args.weights = args.train_output / "pretrained_yolo26n.pt"
        shutil.copy2(downloaded_weights, args.weights)
    else:
        raise FileNotFoundError(args.weights)
    from input_filters import install_training_filter
    install_training_filter()  # SAR_INPUT_FILTER, if set
    model = build_p2_model(source, args.scale, args.p2_yaml, args.detect_yaml, class_names, args.train_output, args.reg_max,
                           args.p4_attention, args.stem_stride, args.spd_stem, args.p1_level, args.p6_level)
    if args.spd_stem:  # the SPD layer must compute exactly what the pretrained first layer computed
        probe = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            first = model.model.model[0].eval(), source.model.model[0].float().eval()
            difference = (first[0](probe) - first[1](probe)).abs().max().item()
        if difference > 1e-3:
            raise RuntimeError(f'SPD first layer differs from the pretrained one by {difference}')
        print(f'SPD first layer reproduces the pretrained layer (max difference {difference:.2e})', flush=True)
    else:
        source_first = source.model.model[0].state_dict()
        target_first = model.model.model[0].state_dict()
        if any(not torch.equal(value, target_first[key]) for key, value in source_first.items()):
            raise RuntimeError('Source backbone weights did not reach the P2 model')
    initialization = preserve_initialization(model, args.train_output / 'initialization_audit.json')
    source_task = source.task
    del source

    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    training_settings = {
        "variant": "p2",
        "architecture": f"YOLO26{args.scale}-P2 (stride 4/8/16/32)",
        "scale": args.scale,
        "pretrained_weights": str(args.weights),
        "pretrained_task": source_task,
        "pretrained_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "initialization_handoff": initialization,
        "p2_architecture_yaml": str(args.p2_yaml),
        "classes": class_names,
        "epochs": args.epochs,
        "image_size": 512,
        "batch_size": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "reg_max": model.model.model[-1].reg_max,
        "p4_attention": args.p4_attention,
        "stem_stride": args.stem_stride,
        "spd_stem": args.spd_stem,
        "p1_level": args.p1_level,
        "p6_level": args.p6_level,
        "close_mosaic": args.close_mosaic,
        "optimizer": f"MuSGD lr0 {args.lr0}" if args.lr0 else "auto (Ultralytics default)",
        "augmentation": "Ultralytics YOLO26 defaults",
        "trainable_parameters": parameter_count,
        "split": "runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json",
        "initialization": f"pretrained YOLO26{args.scale} ({source_task}) shared layers and P3-P5 heads; "
                          "new P2 branch initialized randomly",
    }
    (args.train_output / "benchmark_settings.json").write_text(json.dumps(training_settings, indent=2) + "\n")
    print(json.dumps(training_settings, indent=2), flush=True)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(
        data=str(args.data),
        **({"hsv_h": 0.0, "hsv_s": 0.0} if args.no_colour_jitter else {}),
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch,
        workers=args.workers,
        device=0,
        pretrained=True,
        **({"optimizer": "MuSGD", "lr0": args.lr0} if args.lr0 else {"optimizer": "auto"}),
        patience=args.epochs + 1,
        seed=args.seed,
        deterministic=True,
        project=str(args.train_output.resolve()),
        name="train",
        exist_ok=False,
        plots=True,
        val=True,
        cache=False,
        flipud=args.flipud,
        fliplr=args.fliplr,
        close_mosaic=args.close_mosaic,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    compute = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(0) / 1024**3,
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters()
                                     if parameter.requires_grad),
        "best_weights": str(args.train_output / "train/weights/best.pt"),
    }
    (args.train_output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
