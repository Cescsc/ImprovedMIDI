#!/usr/bin/env python3
"""
Walk a MIDI dataset folder and convert each MIDI file separately.

Behavior
--------
- Recursively scans an input directory for .mid / .midi files.
- Mirrors the input folder structure under the output directory.
- For each MIDI file, creates a dedicated output folder containing:
    tokens.npy
    events.json
    meta.json
    vocab.json
- Writes dataset-level manifest files so it is easy to resume and debug.

This script deliberately does NOT merge the whole database into one giant file.
That keeps failures local, makes inspection easier, and lets you re-run only the
broken files later.

Example
-------
python batch_convert_midi_dataset.py \
  --input-root /data/maestro-v3.0.0 \
  --output-root /data/maestro_tokens
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List

from midi_to_npy import ConverterConfig, convert_midi_file


def find_midi_files(input_root: Path) -> List[Path]:
    midi_files: List[Path] = []
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}:
            midi_files.append(path)
    return sorted(midi_files)


def safe_output_dir(input_root: Path, midi_path: Path, output_root: Path) -> Path:
    """
    Mirror the relative path, but make the file itself a directory.

    Example:
      input_root/foo/bar/example.mid
    becomes
      output_root/foo/bar/example/
    """
    rel = midi_path.relative_to(input_root)
    rel_without_suffix = rel.with_suffix("")
    return output_root / rel_without_suffix


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing MIDI files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory where per-file outputs will be written.")
    parser.add_argument("--time-bin-ms", type=int, default=10, help="Milliseconds per TIME_SHIFT step.")
    parser.add_argument("--max-time-shift-steps", type=int, default=100, help="Largest TIME_SHIFT token before splitting into multiple tokens.")
    parser.add_argument("--velocity-bins", type=int, default=32, help="Number of velocity bins.")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend BOS.")
    parser.add_argument("--no-eos", action="store_true", help="Do not append EOS.")
    parser.add_argument("--no-events-json", action="store_true", help="Do not write readable events.json for each file.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have tokens.npy in the output folder.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    config = ConverterConfig(
        time_bin_ms=args.time_bin_ms,
        max_time_shift_steps=args.max_time_shift_steps,
        velocity_bins=args.velocity_bins,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
        write_events_json=not args.no_events_json,
    )

    midi_files = find_midi_files(args.input_root)
    manifest: Dict[str, object] = {
        "format": "expressive_midi_event_tokens_dataset_v1",
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "config": {
            "time_bin_ms": config.time_bin_ms,
            "max_time_shift_steps": config.max_time_shift_steps,
            "velocity_bins": config.velocity_bins,
            "add_bos": config.add_bos,
            "add_eos": config.add_eos,
            "write_events_json": config.write_events_json,
        },
        "totals": {
            "discovered_midis": len(midi_files),
            "converted": 0,
            "skipped_existing": 0,
            "failed": 0,
        },
        "files": [],
        "failures": [],
    }

    for index, midi_path in enumerate(midi_files, start=1):
        out_dir = safe_output_dir(args.input_root, midi_path, args.output_root)
        tokens_path = out_dir / "tokens.npy"
        rel_midi = midi_path.relative_to(args.input_root)

        print(f"[{index}/{len(midi_files)}] {rel_midi}")

        if args.skip_existing and tokens_path.exists():
            manifest["totals"]["skipped_existing"] += 1
            manifest["files"].append({
                "source_midi": str(rel_midi),
                "output_dir": str(out_dir.relative_to(args.output_root)),
                "status": "skipped_existing",
            })
            continue

        try:
            meta = convert_midi_file(midi_path, out_dir, config)
            manifest["totals"]["converted"] += 1
            manifest["files"].append({
                "source_midi": str(rel_midi),
                "output_dir": str(out_dir.relative_to(args.output_root)),
                "status": "converted",
                "num_tokens": meta["stats"]["num_tokens"],
                "num_events": meta["stats"]["num_events"],
            })
        except Exception as exc:
            manifest["totals"]["failed"] += 1
            failure = {
                "source_midi": str(rel_midi),
                "output_dir": str(out_dir.relative_to(args.output_root)),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            manifest["failures"].append(failure)
            manifest["files"].append({
                "source_midi": str(rel_midi),
                "output_dir": str(out_dir.relative_to(args.output_root)),
                "status": "failed",
            })
            write_json(out_dir / "error.json", failure)

    write_json(args.output_root / "dataset_manifest.json", manifest)
    write_json(args.output_root / "dataset_failures.json", {"failures": manifest["failures"]})

    print("Done.")
    print(json.dumps(manifest["totals"], indent=2))


if __name__ == "__main__":
    main()
