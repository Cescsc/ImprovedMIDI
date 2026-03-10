"""MIDI parsing utilities for the ImprovedMIDI project.

This module provides a simple function that reads a MIDI file and returns
an object representation (``symusic.Score``) that the rest of the project
can work with.

The goal is to keep the function lightweight; other modules can import
``parse_midi`` and manipulate the score, convert it to custom structures,
serialize it, etc.  The underlying implementation is powered by the
``symusic`` package already present in the virtual environment.
"""

from __future__ import annotations

from pathlib import Path

import symusic
from symusic import Score


def parse_midi(path: Path) -> symusic.Score:
    """Read a MIDI file and return a :class:`symusic.Score`.

    Parameters
    ----------
    path : Path
        Path to a ``.mid`` file.  If the file cannot be read or is not a MIDI
        file, an exception will propagate from the underlying I/O routines.

    Returns
    -------
    symusic.Score
        A score object representing the contents of the MIDI file.
    """

    data = path.read_bytes()
    return Score.from_midi(data)
