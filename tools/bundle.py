#!/usr/bin/env python3
"""Bundle an ai-toolkit YuE2 LoRA with a Mothersuperior NAR adapter.

The output is one safetensors carrying both payloads under their own key
namespaces, so a single ``--lora`` file reproduces the full training stack
(user LoRA trained on the adapter-merged base). Examples::

    python tools/bundle.py my_lora.safetensors nar_lora_joint_v9.safetensors \
        -o my_style_v9.safetensors

    python examples/generate.py --output out --lora my_style_v9.safetensors ...

Both inputs must be safetensors: every released adapter version (v4 on) has a
safetensors twin with the same layout, so the ``.pt`` releases are never
needed. The two metadata blocks are kept under ``lora.``/``nar.`` prefixes —
``nar.pair_with`` records which adapter version the bundle carries; bundle the
adapter version the LoRA was trained with.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from yue2.lora import _IO_KEYS, _KEY, _NAR_KEY


def _namespace_counts(keys):
    counts = {"lora": 0, "nar": 0}
    for key in keys:
        if _KEY.match(key):
            counts["lora"] += 1
        elif _NAR_KEY.match(key) or key in _IO_KEYS:
            counts["nar"] += 1
    return counts


def _metadata(path, prefix):
    with safe_open(path, framework="pt") as handle:
        return {f"{prefix}.{k}": str(v) for k, v in (handle.metadata() or {}).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lora", type=Path, help="ai-toolkit YuE2 LoRA (.safetensors)")
    parser.add_argument("adapter", type=Path, help="Mothersuperior nar_lora_joint_*.safetensors")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [p.resolve() for p in (args.lora, args.adapter)]
    if args.output.resolve() in inputs:
        parser.error("output path must differ from the input paths")

    lora_tensors = load_file(args.lora)
    adapter_tensors = load_file(args.adapter)

    lora_counts = _namespace_counts(lora_tensors)
    adapter_counts = _namespace_counts(adapter_tensors)
    bad_lora = [k for k in lora_tensors if not _KEY.match(k)]
    bad_adapter = [k for k in adapter_tensors
                   if not (_NAR_KEY.match(k) or k in _IO_KEYS)]
    if bad_lora or lora_counts["lora"] == 0 or lora_counts["nar"]:
        parser.error(f"{args.lora} is not a pure ai-toolkit YuE2 LoRA "
                     f"(offending keys: {(bad_lora or list(lora_tensors))[:2]})")
    if bad_adapter or adapter_counts["nar"] == 0 or adapter_counts["lora"]:
        parser.error(f"{args.adapter} is not a pure nar_lora_joint adapter "
                     f"(offending keys: {(bad_adapter or list(adapter_tensors))[:2]})")

    overlap = set(lora_tensors) & set(adapter_tensors)
    if overlap:
        parser.error(f"inputs share keys ({sorted(overlap)[:2]}); refusing to bundle")

    metadata = _metadata(args.lora, "lora") | _metadata(args.adapter, "nar")
    metadata["bundle"] = "ai-toolkit YuE2 LoRA + Mothersuperior nar_lora_joint adapter"
    save_file({**lora_tensors, **adapter_tensors}, args.output, metadata=metadata)

    print({"output": str(args.output),
           "lora_keys": lora_counts["lora"],
           "adapter_keys": adapter_counts["nar"],
           "adapter_pair_with": metadata.get("nar.pair_with", "unknown")})


if __name__ == "__main__":
    main()
