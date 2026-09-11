#!/usr/bin/env python3
"""Cover a song: reimagine a transcribed melody score in a new style.

Reads style, lyrics, and seed from a request JSON in the examples/generate.py
format and the melody ABC from a separate file. Covers always run cot="melody"
so the accompaniment can adapt to the new style; any other JSON fields,
including cot, are ignored rather than rejected.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--abc-file", type=Path, required=True)
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
    if "style" not in request or "lyrics" not in request:
        parser.error("The request JSON must provide style and lyrics.")
    abc = args.abc_file.read_text(encoding="utf-8")
    from yue2 import YuE2Pipeline

    with YuE2Pipeline.from_pretrained(
        args.model, vae=args.vae, revision=args.revision,
        vae_revision=args.vae_revision, device="cuda", low_vram=args.low_vram,
    ) as pipe:
        fields = {key: request[key] for key in ("style", "lyrics", "seed") if key in request}
        cover = pipe(abc=abc, cot="melody", **fields)
        cover.save_artifacts(args.output)
        print(json.dumps({"audio": str(args.output / "audio.flac"), "truncated": cover.truncated}))
        return 1 if any(cover.truncated.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
