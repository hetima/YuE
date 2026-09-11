#!/usr/bin/env python3
"""Generate from original example lyrics, optionally with a supplied ABC score."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("song.json"))
    parser.add_argument("--abc-file", type=Path)
    parser.add_argument("--cot", choices=("full", "melody", "off"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--revision")
    parser.add_argument("--vae-revision")
    parser.add_argument("--low-vram", action="store_true",
                        help="Keep only the active generation path on the GPU")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory to retain each version.")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    # Only SongRequest fields are forwarded; unknown JSON fields are ignored.
    fields = {key: request[key] for key in
              ("style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id") if key in request}
    if fields.get("seed") == -1:
        # -1 requests a random seed from SongRequest's [0, 2**63) domain.
        fields["seed"] = random.randrange(2**63)
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
        song.save_artifacts(args.output)
        print(json.dumps({"audio": str(args.output / "audio.flac"), "seed": song.semantic.plan.request.seed,
                          "truncated": song.truncated}))
        return 1 if any(song.truncated.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
