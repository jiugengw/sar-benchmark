"""Make the pinned MMCV 2.0.1 sources compile with PyTorch 2.9/CUDA 12.8."""
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2] / "third_party/mmcv"
    setup = root / "setup.py"
    text = setup.read_text().replace("-std=c++14", "-std=c++17")
    setup.write_text(text)
    path = root / "mmcv/ops/csrc/pytorch/spconv_utils.h"
    path.write_text(path.read_text().replace("tensor.type().scalarType()", "tensor.scalar_type()"))
    # PyTorch now includes Adafactor; MMEngine 0.10.7 also registers the
    # Transformers implementation under that name. Patch only our private env.
    packages = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    builder = packages / "mmengine/optim/optimizer/builder.py"
    text = builder.read_text()
    text = text.replace("OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)",
                        "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor, force=True)")
    builder.write_text(text)


if __name__ == "__main__":
    main()
