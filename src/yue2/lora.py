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

_NAR_KEY = re.compile(
    r"^layers\.(?P<layer>\d+)\.(?P<mod>nar_self_attn|nar_mlp)\."
    r"(?P<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_(?P<side>[AB])$")


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


@torch.no_grad()
def merge_nar_adapter(model, path, strength=1.0):
    """Fold a Mothersuperior joint NAR adapter (nar_lora_joint_*.safetensors)
    into a loaded official YuE2ForCausalLM.

    The file carries LoRA pairs under ``layers.N.nar_self_attn/nar_mlp.<proj>
    .lora_A/B`` — the official model's separate projections, so no row
    splitting — plus complete replacement weights for the top-level
    ``vae2llm``/``llm2vae`` Linears. Returns a ``{"nar_projections": n,
    "io": [...]}`` summary. Raises ValueError on keys outside the expected
    layout, missing sides, or shapes that disagree with the target weights.
    """
    if model.training:
        raise ValueError("merge_nar_adapter requires model.eval()")
    if not isinstance(strength, (float, int)):
        raise TypeError("strength must be a number")
    tensors = load_file(path)
    pairs, io_source = {}, {}
    for key, value in tensors.items():
        match = _NAR_KEY.match(key)
        if match is not None:
            pairs.setdefault((int(match["layer"]), match["mod"], match["proj"]), {})[match["side"]] = value
        elif key in ("vae2llm.weight", "vae2llm.bias", "llm2vae.weight", "llm2vae.bias"):
            io_source[key] = value
        else:
            raise ValueError(f"Unexpected adapter key {key!r}; this merger handles the "
                             "Mothersuperior nar_lora_joint layout only")
    merged = 0
    for (layer_index, mod, proj), sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            raise ValueError(f"Adapter layer {layer_index} {mod}.{proj} misses a side")
        if not 0 <= layer_index < len(model.model.layers):
            raise ValueError(f"Adapter layer {layer_index} is outside the model")
        weight = getattr(getattr(model.model.layers[layer_index], mod), proj).weight
        delta = (sides["B"].float() @ sides["A"].float()) * strength
        if tuple(delta.shape) != tuple(weight.shape):
            raise ValueError(f"Layer {layer_index} {mod}.{proj}: delta {tuple(delta.shape)} "
                             f"vs weight {tuple(weight.shape)}")
        weight.add_(delta.to(weight.device, weight.dtype))
        merged += 1
    io = []
    for name in ("vae2llm", "llm2vae"):
        module = getattr(model, name)
        for pname in ("weight", "bias"):
            key = f"{name}.{pname}"
            if key in io_source:
                target = getattr(module, pname)
                if tuple(io_source[key].shape) != tuple(target.shape):
                    raise ValueError(f"{key}: file {tuple(io_source[key].shape)} "
                                     f"vs model {tuple(target.shape)}")
                target.copy_(io_source[key].to(target.device, target.dtype))
                if pname == "weight":
                    io.append(name)
            elif pname == "weight":
                raise ValueError(f"Adapter has no {key}; the joint layout replaces io weights outright")
    return {"nar_projections": merged, "io": io}
