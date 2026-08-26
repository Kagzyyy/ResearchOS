"""Stage 6a - export the hard-negative reranker to ONNX and quantize to INT8.

    python scripts/export_onnx.py --config configs/onnx.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, seed_everything
from src.rerank.onnx_export import export_onnx, quantize_int8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    ec = cfg["export"]

    source = Path(ec["source_model"])
    if not source.exists():
        raise SystemExit(
            f"{source} not found. Train the hard-negative model first - stages are sequential."
        )

    fp32_path, _inputs = export_onnx(
        model_path=str(source),
        output_path=ec["fp32_path"],
        max_length=ec["max_length"],
        opset=ec["opset"],
    )
    quantize_int8(fp32_path, ec["int8_path"])


if __name__ == "__main__":
    main()
