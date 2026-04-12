#!/usr/bin/env python3
"""
Convert a single MIDI file into a debug-friendly token representation.

Why this format
---------------
- One MIDI file in -> one output folder out.
- Main token data is saved as `tokens.npy` for simple, fast loading later.
- Human-readable `events.json` and `meta.json` are also written so debugging is
  easy and you do not have to inspect raw numpy arrays only.

Output folder structure
-----------------------
<output_dir>/
    tokens.npy          # int32 token ids
    events.json         # readable event sequence with times and token strings
    meta.json           # source path, config, vocabulary summary, stats
    vocab.json          # token -> id mapping used for this file

Token scheme
------------
- NOTE_ON_<pitch>
- NOTE_OFF_<pitch>
- VELOCITY_<bin>
- TIME_SHIFT_<steps>
- PEDAL_64_{0,1}, PEDAL_66_{0,1}, PEDAL_67_{0,1}
- BOS / EOS / PAD

Example
-------
python midi_to_npy.py \
  --midi /path/to/file.mid \
  --output-dir /path/to/output/file_name
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mido
import numpy as np

SPECIAL_TOKENS = ["PAD", "BOS", "EOS"]
NOTE_ON_PREFIX = "NOTE_ON"
NOTE_OFF_PREFIX = "NOTE_OFF"
VELOCITY_PREFIX = "VELOCITY"
TIME_SHIFT_PREFIX = "TIME_SHIFT"
PEDAL_PREFIX = "PEDAL"
SUPPORTED_PEDALS = (64, 66, 67)


@dataclass
class ConverterConfig:
    time_bin_ms: int = 10
    max_time_shift_steps: int = 100
    velocity_bins: int = 32
    add_bos: bool = True
    add_eos: bool = True
    write_events_json: bool = True


@dataclass
class MidiEvent:
    time_sec: float
    sort_key: int
    token: str


class Vocabulary:
    def __init__(self, velocity_bins: int, max_time_shift_steps: int):
        self.tokens: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self._build(velocity_bins, max_time_shift_steps)

    def _add(self, token: str) -> None:
        if token in self.token_to_id:
            raise ValueError(f"Duplicate token: {token}")
        self.token_to_id[token] = len(self.tokens)
        self.tokens.append(token)

    def _build(self, velocity_bins: int, max_time_shift_steps: int) -> None:
        for token in SPECIAL_TOKENS:
            self._add(token)
        for pitch in range(128):
            self._add(f"{NOTE_ON_PREFIX}_{pitch}")
        for pitch in range(128):
            self._add(f"{NOTE_OFF_PREFIX}_{pitch}")
        for velocity_bin in range(velocity_bins):
            self._add(f"{VELOCITY_PREFIX}_{velocity_bin}")
        for step in range(1, max_time_shift_steps + 1):
            self._add(f"{TIME_SHIFT_PREFIX}_{step}")
        for cc in SUPPORTED_PEDALS:
            self._add(f"{PEDAL_PREFIX}_{cc}_0")
            self._add(f"{PEDAL_PREFIX}_{cc}_1")

    def encode(self, token: str) -> int:
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        return self.tokens[token_id]

    def to_dict(self) -> Dict[str, object]:
        return {
            "size": len(self.tokens),
            "tokens": self.tokens,
            "token_to_id": self.token_to_id,
        }


def quantize_velocity(velocity: int, velocity_bins: int) -> int:
    velocity = max(1, min(127, int(velocity)))
    return min(velocity_bins - 1, (velocity - 1) * velocity_bins // 127)


def pedal_state_from_value(value: int) -> int:
    return 1 if int(value) >= 64 else 0


def extract_events(midi_path: Path, config: ConverterConfig) -> List[MidiEvent]:
    """
    Read a MIDI and extract time-ordered performance events.

    This follows the same simple event representation across all tracks. It is
    intentionally conservative and keeps note timing, velocities, and pedals.
    """
    midi = mido.MidiFile(str(midi_path))
    events: List[MidiEvent] = []

    for track in midi.tracks:
        abs_ticks = 0
        tempo = 500000  # default MIDI tempo: 120 BPM

        for msg in track:
            abs_ticks += msg.time
            time_sec = mido.tick2second(abs_ticks, midi.ticks_per_beat, tempo)

            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue

            if msg.type == "note_on":
                if msg.velocity == 0:
                    events.append(MidiEvent(time_sec, 2, f"{NOTE_OFF_PREFIX}_{msg.note}"))
                else:
                    vel_bin = quantize_velocity(msg.velocity, config.velocity_bins)
                    events.append(MidiEvent(time_sec, 0, f"{VELOCITY_PREFIX}_{vel_bin}"))
                    events.append(MidiEvent(time_sec, 1, f"{NOTE_ON_PREFIX}_{msg.note}"))
            elif msg.type == "note_off":
                events.append(MidiEvent(time_sec, 2, f"{NOTE_OFF_PREFIX}_{msg.note}"))
            elif msg.type == "control_change" and msg.control in SUPPORTED_PEDALS:
                state = pedal_state_from_value(msg.value)
                events.append(MidiEvent(time_sec, 3, f"{PEDAL_PREFIX}_{msg.control}_{state}"))

    events.sort(key=lambda e: (e.time_sec, e.sort_key, e.token))
    return events


def events_to_token_ids(
    events: List[MidiEvent],
    vocab: Vocabulary,
    config: ConverterConfig,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    token_ids: List[int] = []
    debug_events: List[Dict[str, object]] = []

    if config.add_bos:
        bos_id = vocab.encode("BOS")
        token_ids.append(bos_id)
        debug_events.append({
            "time_sec": 0.0,
            "kind": "special",
            "token": "BOS",
            "token_id": bos_id,
        })

    prev_time = 0.0
    time_bin_sec = config.time_bin_ms / 1000.0

    for event in events:
        delta = max(0.0, event.time_sec - prev_time)
        steps = int(round(delta / time_bin_sec))

        while steps > 0:
            chunk = min(steps, config.max_time_shift_steps)
            token = f"{TIME_SHIFT_PREFIX}_{chunk}"
            token_id = vocab.encode(token)
            token_ids.append(token_id)
            debug_events.append({
                "time_sec": round(prev_time, 6),
                "kind": "time_shift",
                "delta_sec": round(chunk * time_bin_sec, 6),
                "token": token,
                "token_id": token_id,
            })
            prev_time += chunk * time_bin_sec
            steps -= chunk

        token_id = vocab.encode(event.token)
        token_ids.append(token_id)
        debug_events.append({
            "time_sec": round(event.time_sec, 6),
            "kind": "midi_event",
            "token": event.token,
            "token_id": token_id,
        })
        prev_time = max(prev_time, event.time_sec)

    if config.add_eos:
        eos_id = vocab.encode("EOS")
        token_ids.append(eos_id)
        debug_events.append({
            "time_sec": round(prev_time, 6),
            "kind": "special",
            "token": "EOS",
            "token_id": eos_id,
        })

    return np.asarray(token_ids, dtype=np.int32), debug_events


def convert_midi_file(
    midi_path: Path,
    output_dir: Path,
    config: ConverterConfig,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary(
        velocity_bins=config.velocity_bins,
        max_time_shift_steps=config.max_time_shift_steps,
    )

    events = extract_events(midi_path, config)
    token_ids, debug_events = events_to_token_ids(events, vocab, config)

    np.save(output_dir / "tokens.npy", token_ids)
    (output_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=2), encoding="utf-8")

    if config.write_events_json:
        (output_dir / "events.json").write_text(json.dumps(debug_events, indent=2), encoding="utf-8")

    meta = {
        "format": "expressive_midi_event_tokens_single_file_v1",
        "source_midi": str(midi_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "config": asdict(config),
        "stats": {
            "num_events": len(events),
            "num_tokens": int(token_ids.shape[0]),
            "first_tokens": [vocab.decode(int(x)) for x in token_ids[:32]],
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True, help="Path to a .mid or .midi file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs for this MIDI will be written.")
    parser.add_argument("--time-bin-ms", type=int, default=10, help="Milliseconds per TIME_SHIFT step.")
    parser.add_argument("--max-time-shift-steps", type=int, default=100, help="Largest TIME_SHIFT token before splitting into multiple tokens.")
    parser.add_argument("--velocity-bins", type=int, default=32, help="Number of velocity bins.")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend BOS.")
    parser.add_argument("--no-eos", action="store_true", help="Do not append EOS.")
    parser.add_argument("--no-events-json", action="store_true", help="Do not write readable events.json.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ConverterConfig(
        time_bin_ms=args.time_bin_ms,
        max_time_shift_steps=args.max_time_shift_steps,
        velocity_bins=args.velocity_bins,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
        write_events_json=not args.no_events_json,
    )

    if not args.midi.exists():
        raise FileNotFoundError(f"MIDI file not found: {args.midi}")

    if args.midi.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Expected a .mid or .midi file, got: {args.midi.name}")

    meta = convert_midi_file(args.midi, args.output_dir, config)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
