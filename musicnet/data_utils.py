import datetime
from pathlib import Path

import click
import joblib
import pandas as pd
from music21 import chord, converter, instrument, note
from typing import Any

from musicnet import params


def to_hms(sec: float):
    '''convert seconds to H:M:S'''
    return str(datetime.timedelta(seconds=round(sec, 0)))


def get_midi_elements(midi_path: Path, out_file: Path) -> pd.DataFrame:

    comps = []
    parts = []
    elements = []

    total_comps = len(list(midi_path.glob('*.mid')))

    with click.progressbar(
            midi_path.glob('*.mid'),
            length=total_comps,
            label=click.style('Parsing midi files..',
                              fg='bright_green')) as bar:
        for midi_file in bar:

            midi_data = converter.parse(midi_file)
            midi_parts = instrument.partitionByInstrument(midi_data)

            for part in midi_parts:

                part_contents = part.recurse()

                for element in part_contents:
                    elements.append(element)
                    parts.append(part.id)
                    comps.append(midi_file.stem)

            bar.update(1)

    element_data = pd.DataFrame(
        {
            'comp': comps,
            'part': parts,
            'element': elements
        }
    )

    # convert categorical columns to appropriate type
    categ_columns = ['comp', 'part']
    for col in categ_columns:
        element_data[col] = element_data[col].astype('category')

    return element_data


def get_midi_gm_mapping(
        midi_gm_mapping_file: Path = params.midi_gm_mapping_file):

    midi_gm_mapping = pd.read_csv(midi_gm_mapping_file)

    return midi_gm_mapping


def dump_data(data_object: Any, out_file: Path) -> bool:

    if out_file.exists():
        out_file.unlink()
    joblib.dump(data_object, out_file)

    return out_file.exists()
