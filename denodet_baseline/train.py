#!/usr/bin/env python3
"""Train DenoDet V2 on course data with ImageNet-only initialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/GrokSAR"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/course-data")
    parser.add_argument("--converted", type=Path, default=ROOT / "data/yolo11s-sar-512")
    parser.add_argument("--split", type=Path, default=ROOT / "results/yolo26m-100-p2-328339/seed_0/split.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "pretrained/resnet50-0676ba61.pth")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop at this total epoch while preserving the original LR schedule on resume")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-settings", type=Path,
                        help="JSON settings overriding CLI epochs; defaults to OUTPUT/run_settings.json if present")
    args = parser.parse_args()
    settings_path = args.run_settings or args.output / "run_settings.json"
    if settings_path.is_file():
        overrides = json.loads(settings_path.read_text())
        if set(overrides) != {"epochs"} or type(overrides["epochs"]) is not int:
            raise ValueError("Run settings must contain exactly one integer field: epochs")
        args.epochs = overrides["epochs"]
        print(f"Using run settings {settings_path}: epochs={args.epochs}", flush=True)
    elif args.run_settings is not None:
        raise FileNotFoundError(settings_path)
    if args.epochs < 3 or args.batch < 1:
        raise ValueError("Need at least three epochs and a positive batch size")
    if args.stop_after_epoch is not None:
        if not args.resume or not 1 <= args.stop_after_epoch <= args.epochs:
            raise ValueError("--stop-after-epoch requires --resume and must be within the original schedule")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Run scripts/setup_denodet.sh first: missing {args.weights}")
    # The official ImageNet-1K ResNet50 V1 URL includes this SHA256 prefix.
    weight_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if not weight_hash.startswith("0676ba61"):
        raise ValueError("Expected the unmodified torchvision ImageNet ResNet50 V1 checkpoint")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from denodet_baseline.prepare_dataset import prepare
    dataset = args.output / "dataset"
    metadata = prepare(args.data, args.converted, args.split, dataset)

    import torch
    from mmcv.ops import nms
    from mmengine.runner import Runner, set_random_seed
    from denodet_baseline.config import build_config
    import groksar.models  # register the authors' unchanged FFTResNet
    import denodet_baseline.model  # register the course task adapter
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU allocation is required")
    # Test the compiled CUDA operator on the actual allocated GPU.
    boxes = torch.tensor([[0., 0., 8., 8.], [0., 0., 8., 8.]], device="cuda")
    _, keep = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    if keep.tolist() != [0]:
        raise RuntimeError("MMCV CUDA NMS preflight failed")
    cfg = build_config(ROOT, dataset, metadata, args.output, args.weights,
                       args.epochs, args.batch, args.workers)
    cfg.resume = args.resume
    if args.stop_after_epoch is not None:
        cfg.train_cfg.max_epochs = args.stop_after_epoch
    cfg.dump(str(args.output / "config.py"))
    runner = Runner.from_cfg(cfg)
    model = runner.model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total > 130_000_000 or trainable > 40_000_000:
        raise RuntimeError(f"Model exceeds benchmark budget: total={total}, trainable={trainable}")
    model.init_weights()
    preflight_loader = runner.build_dataloader({**cfg.train_dataloader, "num_workers": 0, "persistent_workers": False})
    batch = next(iter(preflight_loader))
    batch = {key: value[:2] for key, value in batch.items()}
    model.train()
    norm_buffers = {name: value.detach().clone() for name, value in model.named_buffers()
                    if "running_" in name or "num_batches_tracked" in name}
    inputs = model.data_preprocessor(batch, training=True)
    losses = model(**inputs, mode="loss")
    loss, log_vars = model.parse_losses(losses)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite preflight loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite preflight gradient: {name}")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, value in model.named_buffers():
            if name in norm_buffers:
                value.copy_(norm_buffers[name])
    set_random_seed(0)
    upstream = subprocess.check_output(["git", "-C", str(ROOT / "third_party/GrokSAR"), "rev-parse", "HEAD"], text=True).strip()
    settings = dict(architecture="DenoDet V2 FFTResNet50 + FPN + GFL with shared image head",
                    upstream_commit=upstream, initialization="torchvision ImageNet-1K ResNet50 V1 only",
                    pretrained_checkpoint=str(args.weights), pretrained_sha256=weight_hash,
                    sar_pretrained_checkpoint=False, seed=0, epochs=args.epochs, batch=args.batch,
                    stop_after_epoch=args.stop_after_epoch, resumed=args.resume,
                    run_settings=str(settings_path.resolve()) if settings_path.is_file() else None,
                    precision="float32 (including FFT denoising)", parameters=total, trainable_parameters=trainable,
                    gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), dataset=metadata,
                    preflight_losses={key: float(value.detach()) if torch.is_tensor(value) else float(value)
                                      for key, value in log_vars.items()})
    (args.output / "benchmark_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"PREFLIGHT PASSED: CUDA NMS, real-image forward/backward; parameters={total:,}, trainable={trainable:,}", flush=True)
    if args.preflight_only:
        return
    del inputs, losses, loss, batch, preflight_loader, norm_buffers
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    runner.train()
    torch.cuda.synchronize()
    compute = dict(training_wall_seconds=time.monotonic() - started,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), **settings)
    (args.output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    best = sorted(args.output.glob("best_*.pth"))
    if len(best) != 1:
        raise RuntimeError(f"Expected one best checkpoint, found {best}")
    # Only load checkpoints generated by this local run.
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, str(best[0]), map_location="cpu")
    from denodet_baseline.export import export_run
    export_run(model, args, metadata, dataset, best[0], settings)


if __name__ == "__main__":
    main()
