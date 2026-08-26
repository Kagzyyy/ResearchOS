"""Export a fine-tuned cross-encoder to ONNX and quantize it to INT8.

Two artifacts come out of here:

  reranker.onnx        fp32 graph, no PyTorch runtime needed
  reranker.int8.onnx   the same graph with weights dynamically quantized to int8

Dynamic quantization stores weights as int8 and computes activations in fp32,
picking activation scales at runtime. It is the right tradeoff here: no
calibration set is required, the file shrinks roughly 4x, and CPU inference gets
materially faster. The cost is a small, measurable quality drop - which is why
row 6 of the results table re-evaluates quality instead of only reporting speed.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer


def export_onnx(
    model_path: str,
    output_path: str,
    max_length: int = 384,
    opset: int = 17,
) -> tuple[str, list[str]]:
    """Trace the cross-encoder to an fp32 ONNX graph. Returns (path, input_names)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cross_encoder = CrossEncoder(model_path, num_labels=1, max_length=max_length, device="cpu")
    model = cross_encoder.model.eval().float()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sample = tokenizer(
        ["a claim about vitamin d"],
        ["an abstract about vitamin d and respiratory infection"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in sample]
    args = tuple(sample[name] for name in input_names)

    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            str(out),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    tokenizer.save_pretrained(out.parent)
    size_mb = out.stat().st_size / 1e6
    print(f"[onnx] exported fp32 -> {out} ({size_mb:.1f} MB) inputs={input_names}")
    return str(out), input_names


def quantize_int8(fp32_path: str, int8_path: str) -> str:
    """Apply INT8 dynamic quantization to an exported ONNX graph."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(fp32_path)
    target = Path(int8_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
    )

    before = source.stat().st_size / 1e6
    after = target.stat().st_size / 1e6
    print(f"[onnx] quantized int8 -> {target} ({before:.1f} MB -> {after:.1f} MB, {before/after:.1f}x smaller)")
    return str(target)
