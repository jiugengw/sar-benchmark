"""Persistent compute and provenance records for report preparation."""
import importlib.metadata
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def provenance(config):
    """Record model sources, actual library versions, and input conventions."""
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'terratorch', 'torchgeo', 'pycocotools', 'Pillow', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    sources = {
        'resnet34': {'name': 'torchvision ResNet34_Weights.IMAGENET1K_V1',
                     'source': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
                     'pretraining_data': 'ImageNet-1k',
                     'input_adaptation': 'sum RGB first-convolution weights into one grayscale channel'},
        'vit': {'name': 'vit_base_patch16_224.augreg_in21k',
                'source': 'https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k',
                'pretraining_data': 'ImageNet-21k'},
        'terramind': {'name': 'TerraMind-1.0-base',
                     'source': 'https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base',
                     'input_modality': 'S1GRD; grayscale repeated as VV and VH'},
    }
    weights_path = config.get('backbone_weights')
    digest = None
    if weights_path is not None and Path(weights_path).is_file():
        with Path(weights_path).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'python': platform.python_version(), 'libraries': versions,
        'git_commit': commit, 'tracked_files_modified': dirty,
        'pretrained_models': [dict(sources[config['backbone']],
                                  local_weights=weights_path, local_weights_sha256=digest)]
                            if config.get('init') == 'pretrained' else [],
        'input': {'size': [1, 512, 512], 'geometry': config.get('geometry', 'stretch'),
                  'pixel_scale': 'uint8 / 255', 'mean': 0.2526, 'std': 0.2133,
                  'training_flips': 'independent horizontal/vertical, each probability 0.5',
                  'training_photometric': bool(config.get('photometric', False))},
        'configuration': config,
    }


class ComputeTracker:
    """Write at start, every 30 seconds, and at exit; preserve earlier attempts.

    Process hours are separate from Slurm allocation hours. A hard kill can leave
    a running record; scheduler accounting is authoritative for those attempts.
    """
    def __init__(self, path, phase, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.record = {'record_id': uuid.uuid4().hex, 'phase': phase, 'status': 'running',
                       'started_utc': utc_now(), 'hostname': socket.gethostname(),
                       'pid': os.getpid(), 'command': sys.argv,
                       'slurm': {key: os.environ.get(key) for key in
                                 ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
                                  'SLURM_JOB_PARTITION', 'SLURM_JOB_GPUS', 'SLURM_GPUS_ON_NODE')},
                       'metadata': metadata or {}}

    def _write(self):
        import torch
        elapsed = time.monotonic() - self.started
        self.record.update(updated_utc=utc_now(), process_wall_hours=elapsed / 3600)
        # These entry points use one CUDA device, even if more are visible.
        initialized = torch.cuda.is_initialized()
        self.record['model_gpu_count'] = 1 if initialized else 0
        self.record['process_gpu_hours'] = elapsed / 3600 if initialized else 0
        self.record['gpu_hours_definition'] = 'process wall time × model GPU count; includes CPU setup, excludes queue time'
        if initialized:
            device = torch.cuda.current_device()
            self.record.update(gpu_type=torch.cuda.get_device_name(device),
                               cuda_version=torch.version.cuda,
                               peak_gpu_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                               peak_gpu_memory_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.record, indent=2, default=str))
        temporary.replace(self.path)

    def update(self, **values):
        with self.lock:
            self.record.update(values)
            self._write()

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.update()

    def __enter__(self):
        if self.path.exists():
            self.path.rename(self.path.with_name(f'{self.path.stem}.previous-{uuid.uuid4().hex}.json'))
        self.update()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        self.update(status='completed' if kind is None else 'failed', ended_utc=utc_now(),
                    error=None if kind is None else f'{kind.__name__}: {error}')
        return False
