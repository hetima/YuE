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
"""

import argparse
import json
import random
import re
from pathlib import Path


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--revision")
    parser.add_argument("--vae-revision")
    parser.add_argument("--low-vram", action="store_true",
                        help="Keep only the active generation path on the GPU")
    parser.add_argument("--save-abc", action="store_true")
    args = parser.parse_args()
    # if args.output.exists():
    #     parser.error("Choose a fresh output directory to retain each version.")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    style_key = request.get("style_key")
    if isinstance(style_key, str) and style_key.strip() and style_key in request:
        # A nonblank style_key selects a style preset from the same request.
        request["style"] = request[style_key]
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
    from yue2 import YuE2Pipeline

    with YuE2Pipeline.from_pretrained(
        args.model, vae=args.vae, revision=args.revision,
        vae_revision=args.vae_revision, device="cuda", low_vram=args.low_vram,
    ) as pipe:
        song = pipe(**fields)
        args.output.mkdir(parents=True, exist_ok=True)
        # The flac basename follows the request id; missing or blank id means "audio".
        base = str(fields.get("id") or "").strip() or "audio"
        base = re.sub(r'[\\/:*?"<>|]', "_", base)  # keep it a legal Windows filename
        audio_path = next_audio_path(args.output, base)
        song.save(audio_path)
        # song.save_artifacts(args.output)
        if args.save_abc and song.semantic.plan.abc is not None:
           (audio_path.with_suffix(".abc")).write_bytes(song.semantic.plan.abc.encode("utf-8"))
        print(json.dumps({"audio": str(audio_path), "seed": song.semantic.plan.request.seed,
                          "truncated": song.truncated}))
        return 1 if any(song.truncated.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
