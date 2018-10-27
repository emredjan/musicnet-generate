from pathlib import Path

import click
from music21 import chord, converter, instrument, note


def get_notes(midi_path: Path):

    notes = []

    total_songs = len(list(midi_path.glob('*.mid')))

    with click.progressbar(
            midi_path.glob('*.mid'),
            length=total_songs,
            label='Parsing midi files..') as bar:
        for midi_file in bar:

            midi_data = converter.parse(midi_file)

            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi_data)

            if parts:
                notes_to_parse = parts.recurse()
            else:
                notes_to_parse = midi_data.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

            bar.update(1)

    return notes
