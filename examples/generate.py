#!/usr/bin/env python3
"""Generate from original example lyrics, optionally with a supplied ABC score.

Request JSON handling on top of the upstream SongRequest fields:
- seed: -1 draws a random seed from SongRequest's [0, 2**63) domain; the drawn
  value is echoed in the summary JSON and recorded in request.json so the run
  can be reproduced exactly.
- style_key: when its value is a nonblank string naming another key in the
  same JSON, that key's value replaces style (style presets in one file);
  a missing key, blank value, or absent style_key keeps style as-is.
- id: also names the output flac; a missing or blank id keeps the "audio" base.
Unknown JSON fields are ignored rather than rejected.

Request JSON can also stand in for CLI flags that were not passed; the command
line always wins and blank values count as unset: output, abc_file,
lyrics_file (a file fallback for the lyrics key), nar_lora, lora (a path or a
list of paths), lora_strength.

--num N generates N songs on one model load: songs run with seed, seed-1, ...
(the start seed is lifted above N when seed < N so the countdown stays >= 0).
A supplied ABC score is reused for every song; otherwise each song plans a
fresh score. A failed song is reported and the batch continues.

--lora PATH [--lora-strength S] merges ai-toolkit YuE2 LoRA files, Mothersuperior
NAR adapters, or bundles of both (tools/bundle.py) into the model weights before
generation; repeatable for stacking. A file may carry at most one adapter
source alongside the --nar-lora flag.
--nar-lora PATH merges a Mothersuperior nar_lora_joint adapter (LoRA pairs on
the NAR branch + full vae2llm/llm2vae replacement) before the --lora merges.
"""

import argparse
import json
import random
import re
from pathlib import Path


def _blank(value):
    """None or a whitespace-only string counts as unset (request JSON values)."""
    return value is None or (isinstance(value, str) and not value.strip())


def next_audio_path(out_dir, base):
    # <base>.flac first, then <base>_<max existing suffix + 1>.flac.
    # if not (out_dir / f"{base}.flac").exists():
    #     return out_dir / f"{base}.flac"
    numbered = re.compile(rf"{re.escape(base)}_(\d+)\.flac")
    numbers = [int(m.group(1)) for p in out_dir.iterdir()
               if (m := numbered.fullmatch(p.name))]
    return out_dir / f"{base}_{max(numbers, default=0) + 1}.flac"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("song.json"))
    parser.add_argument("--abc-file", type=Path)
    parser.add_argument("--lyrics", type=Path,
                        help="Read lyrics from a UTF-8 text file, overriding the request JSON")
    parser.add_argument("--cot", choices=("full", "melody", "off"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (fallback: request JSON 'output')")
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--revision")
    parser.add_argument("--vae-revision")
    parser.add_argument("--low-vram", action="store_true",
                        help="Keep only the active generation path on the GPU")
    parser.add_argument("--save-abc", action="store_true")
    parser.add_argument("--num", type=int, default=1, metavar="N",
                        help="Generate N songs on one model load; seeds count down")
    parser.add_argument("--lora", action="append", default=[], metavar="PATH",
                        help="Merge an ai-toolkit LoRA, a NAR adapter, or a bundle; repeatable")
    parser.add_argument("--lora-strength", type=float, default=None,
                        help="Multiplier applied to every --lora delta (default: 1.0 or request JSON 'lora_strength')")
    parser.add_argument("--nar-lora", type=Path, metavar="PATH",
                        help="Merge a Mothersuperior nar_lora_joint adapter into the NAR branch")
    args = parser.parse_args()
    # if args.output.exists():
    #     parser.error("Choose a fresh output directory to retain each version.")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    style_key = request.get("style_key")
    if isinstance(style_key, str) and style_key.strip() and style_key in request:
        # A nonblank style_key selects a style preset from the same request.
        request["style"] = request[style_key]
    # CLI flags win; request JSON stands in for anything left unset (blank = unset).
    args.output = args.output if args.output is not None else (
        None if _blank(request.get("output")) else Path(request["output"]))
    if args.output is None:
        parser.error("--output is required: pass --output or set 'output' in the request JSON")
    for name, key in (("--abc-file", "abc_file"), ("--nar-lora", "nar_lora")):
        value = getattr(args, name.lstrip("-").replace("-", "_"))
        if value is None and not _blank(request.get(key)):
            setattr(args, name.lstrip("-").replace("-", "_"), Path(request[key]))
    # lyrics_file only stands in when the request carries no inline lyrics
    if args.lyrics is None and _blank(request.get("lyrics")) and not _blank(request.get("lyrics_file")):
        args.lyrics = Path(request["lyrics_file"])
    if not args.lora:
        entries = request.get("lora")
        if isinstance(entries, str):
            entries = [entries]
        if isinstance(entries, list):
            try:
                args.lora = [Path(e) for e in entries if not _blank(e)]
            except TypeError:
                parser.error("request JSON 'lora' must be a path or a list of paths")
    if args.lora_strength is None:
        strength = request.get("lora_strength")
        if _blank(strength):
            args.lora_strength = 1.0
        else:
            try:
                args.lora_strength = float(strength)
            except (TypeError, ValueError):
                parser.error("request JSON 'lora_strength' must be a number")
    # Only SongRequest fields are forwarded; unknown JSON fields are ignored.
    fields = {key: request[key] for key in
              ("style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id") if key in request}
    if fields.get("seed") == -1:
        # -1 requests a random seed from SongRequest's [0, 2**63) domain.
        fields["seed"] = random.randrange(2**63)
    if args.lyrics and args.lyrics.is_file():
        # A UTF-8 lyrics file overrides the request lyrics; CRLF is normalized to LF.
        fields["lyrics"] = args.lyrics.read_text(encoding="utf-8").replace("\r\n", "\n")
    if args.abc_file:
        fields["abc"] = args.abc_file.read_text(encoding="utf-8")
    if args.cot:
        fields["cot"] = args.cot
    if fields.get("abc") is not None and fields.get("cot", "full") == "off":
        parser.error("A supplied score requires full or melody mode.")
    if args.num < 1:
        parser.error("--num must be at least 1")
    # Songs run with seed, seed-1, ...; lift the start above num so the countdown
    # never dips below zero (seed=3, num=5 -> 8,7,6,5,4). A missing seed starts
    # from SongRequest's default.
    start_seed = fields.setdefault("seed", 831001)
    if args.num > start_seed:
        start_seed += args.num
    fields["seed"] = start_seed
    from yue2 import YuE2Pipeline

    with YuE2Pipeline.from_pretrained(
        args.model, vae=args.vae, revision=args.revision,
        vae_revision=args.vae_revision, device="cuda", low_vram=args.low_vram,
    ) as pipe:
        if args.lora or args.nar_lora:
            from yue2.lora import merge_lora, merge_nar_adapter
            pipe._load_model()  # the LLM loads lazily; materialize it before merging
        if args.nar_lora:
            # the adapter is part of the training base, so fold it before any --lora
            summary = merge_nar_adapter(pipe._model, args.nar_lora)
            print(json.dumps({"nar_lora": str(args.nar_lora), "modules": summary}))
        if args.lora:
            for path in args.lora:
                counts = merge_lora(pipe._model, path, args.lora_strength)
                print(json.dumps({"lora": str(path), "modules": counts}))
        args.output.mkdir(parents=True, exist_ok=True)
        # The flac basename follows the request id; missing or blank id means "audio".
        base = str(fields.get("id") or "").strip() or "audio"
        base = re.sub(r'[\\/:*?"<>|]', "_", base)  # keep it a legal Windows filename
        generated = failed = 0
        truncated_any = False
        for index in range(args.num):
            # a supplied score is reused every song; otherwise pipe plans a fresh one
            try:
                song = pipe(**{**fields, "seed": start_seed - index})
                audio_path = next_audio_path(args.output, base)
                song.save(audio_path)
                # song.save_artifacts(args.output)
                if args.save_abc and song.semantic.plan.abc is not None:
                    (audio_path.with_suffix(".abc")).write_bytes(song.semantic.plan.abc.encode("utf-8"))
                truncated_any |= any(song.truncated.values())
                generated += 1
                print(json.dumps({"song": f"{index + 1}/{args.num}", "audio": str(audio_path),
                                  "seed": song.semantic.plan.request.seed, "truncated": song.truncated}))
            except Exception as exc:  # one bad take must not waste the batch
                failed += 1
                print(json.dumps({"song": f"{index + 1}/{args.num}", "seed": start_seed - index,
                                  "error": f"{type(exc).__name__}: {exc}"}))
        print(json.dumps({"generated": generated, "failed": failed, "output": str(args.output)}))
        return 1 if failed or truncated_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
