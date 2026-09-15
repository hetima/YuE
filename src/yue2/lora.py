"""Merge ai-toolkit YuE2 LoRA checkpoints into the official YuE2 weights.

ai-toolkit's YuE2 extension wraps each expert with fused projections and
swapped branch names before saving: ``text_encoders.*`` is the AR expert and
``diffusion_model.*`` the NAR expert, whose ``qkv_proj``/``gate_up_proj``
linears hold q|k|v and gate|up rows in that order. The saved pairs use no
alpha correction (linear_alpha equals rank), so the applied delta is
``B @ A * strength`` exactly as the trainer's own merge helper computes it.
Merging keeps standard ``nn.Linear`` modules, so the CUDA-graph decode path
stays valid.
"""
from __future__ import annotations

import re

import torch
from safetensors.torch import load_file

_KEY = re.compile(
    r"^(?P<branch>text_encoders|diffusion_model)\.model\.layers\.(?P<layer>\d+)\."
    r"(?P<module>self_attn\.qkv_proj|self_attn\.o_proj|mlp\.gate_up_proj|mlp\.down_proj)\."
    r"lora_(?P<side>[AB])\.weight$")


@torch.no_grad()
def merge_lora(model, path, strength=1.0):
    """Fold one ai-toolkit YuE2 LoRA file into a loaded YuE2ForCausalLM.

    Returns a ``{branch: merged module count}`` summary. Raises ValueError on
    keys that do not match the expected layout or shapes that disagree with
    the target weights.
    """
    if model.training:
        raise ValueError("merge_lora requires model.eval()")
    if not isinstance(strength, float) and not isinstance(strength, int):
        raise TypeError("strength must be a number")
    tensors = load_file(path)
    config = model.config
    inner = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    row_splits = {
        "self_attn.qkv_proj": {"q_proj": (0, inner), "k_proj": (inner, inner + kv),
                               "v_proj": (inner + kv, inner + 2 * kv)},
        "mlp.gate_up_proj": {"gate_proj": (0, config.intermediate_size),
                             "up_proj": (config.intermediate_size, 2 * config.intermediate_size)},
    }
    pairs, counts = {}, {"text_encoders": 0, "diffusion_model": 0}
    for key, value in tensors.items():
        match = _KEY.match(key)
        if match is None:
            raise ValueError(f"Unexpected LoRA key {key!r}; this merger handles the "
                             "ai-toolkit YuE2 layout only")
        ident = (match["branch"], int(match["layer"]), match["module"], match["side"])
        pairs.setdefault(ident[:-1], {})[match["side"]] = value
    for (branch, layer_index, module), sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            raise ValueError(f"LoRA module {branch} layer {layer_index} {module} misses a side")
        if not 0 <= layer_index < len(model.model.layers):
            raise ValueError(f"LoRA layer {layer_index} is outside the model")
        attention_name, mlp_name = ("self_attn", "mlp") if branch == "text_encoders" \
            else ("nar_self_attn", "nar_mlp")
        layer = model.model.layers[layer_index]
        delta = (sides["B"].float() @ sides["A"].float()) * strength
        if module == "self_attn.o_proj":
            targets = {f"{attention_name}.o_proj": (0, None)}
        elif module == "mlp.down_proj":
            targets = {f"{mlp_name}.down_proj": (0, None)}
        else:
            attention_or_mlp = attention_name if module.startswith("self_attn") else mlp_name
            targets = {f"{attention_or_mlp}.{name}": bounds
                       for name, bounds in row_splits[module].items()}
        for target_name, (low, high) in targets.items():
            parent, _, leaf = target_name.rpartition(".")
            weight = getattr(getattr(layer, parent), leaf).weight
            rows = delta if high is None else delta[low:high]
            if tuple(rows.shape) != tuple(weight.shape):
                raise ValueError(f"{branch} layer {layer_index} {target_name}: "
                                 f"delta {tuple(rows.shape)} vs weight {tuple(weight.shape)}")
            weight.add_(rows.to(weight.device, weight.dtype))
        counts[branch] += 1
    return counts
