#!/usr/bin/env python3
"""
Convert one expressive MIDI file into a simplified, note-preserving robotic MIDI.

This version is designed to preserve the *same underlying notes* while removing
performance expression:
- keeps the same pitches and note count
- preserves note ordering
- removes control changes, pedal, pitch bend, aftertouch, etc.
- flattens velocity to one fixed value
- optionally snaps note starts/ends to a grid *without collapsing note order*
- prevents same-pitch overlaps introduced by quantization

This is better done from the original MIDI than from NPY tokens, because the
original MIDI contains the authoritative note starts and note ends.

Example:
    python midi_to_note_preserving_robotic_midi.py \
        --input expressive.mid \
        --output robotic.mid \
        --grid 16 \
        --velocity 80 \
        --bpm 120

Recommended conservative setting if you want to preserve the musical content
very closely while removing expression:
    --grid 0
This disables onset/duration quantization and only removes expressive controls,
forces a constant tempo meta event, and sets a fixed velocity.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import mido


@dataclass
class Note:
    pitch: int
    channel: int
    start_tick: int
    end_tick: int
    track_hint: int | None = None


@dataclass
class ConversionStats:
    input_file: str
    output_file: str
    ticks_per_beat: int
    original_note_count: int
    output_note_count: int
    grid_division: int
    grid_ticks: int
    fixed_velocity: int
    bpm: int
    force_4_4: bool
    same_pitch_overlap_repairs: int
    onset_order_repairs: int
    zero_length_repairs: int
    note_range_min: int | None
    note_range_max: int | None


@dataclass
class RawEvent:
    abs_tick: int
    msg: mido.Message
    track_index: int


def merged_raw_events(mid: mido.MidiFile) -> List[RawEvent]:
    """Merge all tracks into one global time-ordered event list.

    We intentionally ignore meta messages here when extracting notes.
    """
    per_track_events: List[List[RawEvent]] = []
    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        events: List[RawEvent] = []
        for msg in track:
            abs_tick += msg.time
            if not msg.is_meta:
                events.append(RawEvent(abs_tick=abs_tick, msg=msg, track_index=track_index))
        per_track_events.append(events)

    merged: List[RawEvent] = []
    indices = [0] * len(per_track_events)
    while True:
        candidates = []
        for i, events in enumerate(per_track_events):
            if indices[i] < len(events):
                candidates.append(events[indices[i]])
        if not candidates:
            break
        # Stable ordering: by tick, then note_off-like before note_on, then track.
        chosen = min(
            candidates,
            key=lambda e: (
                e.abs_tick,
                0 if (e.msg.type == 'note_off' or (e.msg.type == 'note_on' and e.msg.velocity == 0)) else 1,
                e.track_index,
            ),
        )
        merged.append(chosen)
        indices[chosen.track_index] += 1
    return merged


def collect_notes(mid: mido.MidiFile) -> List[Note]:
    """Collect note intervals globally using FIFO pairing by (channel, pitch)."""
    active: DefaultDict[Tuple[int, int], List[Tuple[int, int | None]]] = defaultdict(list)
    notes: List[Note] = []

    for event in merged_raw_events(mid):
        msg = event.msg
        if msg.type == 'note_on' and msg.velocity > 0:
            active[(msg.channel, msg.note)].append((event.abs_tick, event.track_index))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if active[key]:
                start_tick, track_hint = active[key].pop(0)
                end_tick = max(event.abs_tick, start_tick + 1)
                notes.append(
                    Note(
                        pitch=msg.note,
                        channel=msg.channel,
                        start_tick=start_tick,
                        end_tick=end_tick,
                        track_hint=track_hint,
                    )
                )

    notes.sort(key=lambda n: (n.start_tick, n.pitch, n.end_tick, n.channel))
    return notes


def round_to_grid(tick: int, grid_ticks: int) -> int:
    return int(round(tick / grid_ticks) * grid_ticks)


def quantize_onsets_without_collapsing(notes: Sequence[Note], grid_ticks: int) -> Tuple[List[int], int]:
    """Quantize onset clusters while preserving global onset order.

    Notes that originally started together remain together. Distinct onset clusters
    are never collapsed into the same quantized time.
    """
    unique_starts = sorted({n.start_tick for n in notes})
    if not unique_starts:
        return [], 0

    repairs = 0
    qstarts: List[int] = []
    prev_q = None
    for i, start in enumerate(unique_starts):
        q = round_to_grid(start, grid_ticks)
        if i == 0:
            q = max(0, q)
        else:
            # Preserve distinct onset ordering.
            min_allowed = prev_q if start == unique_starts[i - 1] else prev_q + 1
            if q < min_allowed:
                q = min_allowed
                repairs += 1
            elif q == prev_q and start != unique_starts[i - 1]:
                q = prev_q + 1
                repairs += 1
        qstarts.append(q)
        prev_q = q
    return qstarts, repairs


def normalize_notes(
    notes: Sequence[Note],
    grid_ticks: int,
    gate_fraction: float,
) -> Tuple[List[Note], int, int, int]:
    """Normalize notes while preserving note identity and ordering.

    If grid_ticks <= 0, timing is preserved exactly and only note overlaps of the
    same pitch are repaired if needed.
    """
    if not notes:
        return [], 0, 0, 0

    same_pitch_repairs = 0
    onset_repairs = 0
    zero_repairs = 0

    if grid_ticks > 0:
        unique_starts = sorted({n.start_tick for n in notes})
        qstarts, onset_repairs = quantize_onsets_without_collapsing(notes, grid_ticks)
        start_map = {orig: q for orig, q in zip(unique_starts, qstarts)}
    else:
        start_map = {s: s for s in sorted({n.start_tick for n in notes})}

    normalized: List[Note] = []
    for n in notes:
        start = start_map[n.start_tick]
        original_duration = max(1, n.end_tick - n.start_tick)

        if grid_ticks > 0:
            # Quantize duration conservatively. Gate fraction <1 shortens notes a bit,
            # which helps remove pedal-like smear while keeping note identity.
            target_duration = max(1, int(round(original_duration * gate_fraction)))
            qdur = max(grid_ticks, round_to_grid(target_duration, grid_ticks))
            end = start + qdur
        else:
            end = start + max(1, int(round(original_duration * gate_fraction)))

        if end <= start:
            end = start + (grid_ticks if grid_ticks > 0 else 1)
            zero_repairs += 1

        normalized.append(
            Note(
                pitch=n.pitch,
                channel=n.channel,
                start_tick=start,
                end_tick=end,
                track_hint=n.track_hint,
            )
        )

    # Prevent same-pitch overlaps introduced by quantization.
    by_key: DefaultDict[Tuple[int, int], List[Note]] = defaultdict(list)
    for n in normalized:
        by_key[(n.channel, n.pitch)].append(n)

    repaired: List[Note] = []
    for _, group in by_key.items():
        group.sort(key=lambda n: (n.start_tick, n.end_tick))
        prev_end = None
        for i, n in enumerate(group):
            start = n.start_tick
            end = n.end_tick
            if prev_end is not None and start < prev_end:
                start = prev_end
                same_pitch_repairs += 1
            if end <= start:
                # Keep note present, but non-overlapping.
                end = start + (grid_ticks if grid_ticks > 0 else 1)
                zero_repairs += 1
            repaired.append(
                Note(
                    pitch=n.pitch,
                    channel=n.channel,
                    start_tick=start,
                    end_tick=end,
                    track_hint=n.track_hint,
                )
            )
            prev_end = end

    repaired.sort(key=lambda n: (n.start_tick, n.pitch, n.end_tick, n.channel))
    return repaired, same_pitch_repairs, onset_repairs, zero_repairs


def notes_to_midi_track(
    notes: Iterable[Note],
    bpm: int,
    velocity: int,
    force_4_4: bool,
) -> mido.MidiTrack:
    """Build a single stripped-down MIDI track."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    if force_4_4:
        track.append(
            mido.MetaMessage(
                'time_signature',
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
    track.append(mido.Message('program_change', channel=0, program=0, time=0))

    events: List[Tuple[int, int, int, mido.Message]] = []
    for n in notes:
        events.append((n.start_tick, 1, n.pitch, mido.Message('note_on', note=n.pitch, velocity=velocity, channel=0, time=0)))
        events.append((n.end_tick, 0, n.pitch, mido.Message('note_off', note=n.pitch, velocity=0, channel=0, time=0)))

    # note_off before note_on at the same tick helps avoid spurious overlaps.
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    last_tick = 0
    for abs_tick, _, _, msg in events:
        msg.time = max(0, abs_tick - last_tick)
        last_tick = abs_tick
        track.append(msg)

    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def convert_one(
    input_path: Path,
    output_path: Path,
    grid_division: int,
    velocity: int,
    bpm: int,
    gate_fraction: float,
    force_4_4: bool,
    save_stats: bool,
) -> ConversionStats:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not (1 <= velocity <= 127):
        raise ValueError('velocity must be between 1 and 127')
    if bpm <= 0:
        raise ValueError('bpm must be > 0')
    if not (0 < gate_fraction <= 1.5):
        raise ValueError('gate_fraction must be > 0 and <= 1.5')

    mid = mido.MidiFile(input_path)
    ticks_per_beat = mid.ticks_per_beat
    grid_ticks = 0 if grid_division <= 0 else max(1, ticks_per_beat * 4 // grid_division)

    original_notes = collect_notes(mid)
    normalized_notes, same_pitch_repairs, onset_repairs, zero_repairs = normalize_notes(
        original_notes,
        grid_ticks=grid_ticks,
        gate_fraction=gate_fraction,
    )

    out_mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    out_mid.tracks.append(
        notes_to_midi_track(
            notes=normalized_notes,
            bpm=bpm,
            velocity=velocity,
            force_4_4=force_4_4,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_mid.save(output_path)

    note_range = [n.pitch for n in normalized_notes]
    stats = ConversionStats(
        input_file=str(input_path),
        output_file=str(output_path),
        ticks_per_beat=ticks_per_beat,
        original_note_count=len(original_notes),
        output_note_count=len(normalized_notes),
        grid_division=grid_division,
        grid_ticks=grid_ticks,
        fixed_velocity=velocity,
        bpm=bpm,
        force_4_4=force_4_4,
        same_pitch_overlap_repairs=same_pitch_repairs,
        onset_order_repairs=onset_repairs,
        zero_length_repairs=zero_repairs,
        note_range_min=min(note_range) if note_range else None,
        note_range_max=max(note_range) if note_range else None,
    )

    if save_stats:
        stats_path = output_path.with_suffix(output_path.suffix + '.stats.json')
        with stats_path.open('w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2)

    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Convert one expressive MIDI file into a simplified note-preserving robotic MIDI.'
    )
    parser.add_argument('--input', type=Path, required=True, help='Input .mid/.midi path')
    parser.add_argument('--output', type=Path, required=True, help='Output .mid path')
    parser.add_argument(
        '--grid',
        type=int,
        default=0,
        help='Grid per whole note: 0 disables quantization; 4, 8, 16, 32 are common. Default: 0',
    )
    parser.add_argument(
        '--velocity',
        type=int,
        default=80,
        help='Fixed velocity for all output note_on messages. Default: 80',
    )
    parser.add_argument(
        '--bpm',
        type=int,
        default=120,
        help='Constant tempo metadata for the output MIDI. Default: 120',
    )
    parser.add_argument(
        '--gate-fraction',
        type=float,
        default=0.95,
        help='Multiply note durations by this factor to reduce articulation/pedal smear. Default: 0.95',
    )
    parser.add_argument(
        '--keep-time-signature',
        action='store_true',
        help='Do not force 4/4 metadata in the output.',
    )
    parser.add_argument(
        '--no-stats-json',
        action='store_true',
        help='Do not write sidecar stats JSON.',
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.input.suffix.lower() not in {'.mid', '.midi'}:
        raise ValueError('Input file must end with .mid or .midi')

    stats = convert_one(
        input_path=args.input,
        output_path=args.output,
        grid_division=args.grid,
        velocity=args.velocity,
        bpm=args.bpm,
        gate_fraction=args.gate_fraction,
        force_4_4=not args.keep_time_signature,
        save_stats=not args.no_stats_json,
    )
    print('Done.')
    print(json.dumps(asdict(stats), indent=2))


if __name__ == '__main__':
    main()
