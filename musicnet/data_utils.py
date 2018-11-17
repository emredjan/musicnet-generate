import datetime
from pathlib import Path
from typing import Any, List

import click
import joblib
import pandas as pd
from music21 import chord, converter, instrument, note
from tqdm import tqdm

from musicnet import params


def to_hms(sec: float):
    '''convert seconds to H:M:S'''
    return str(datetime.timedelta(seconds=round(sec, 0)))


def get_midi_elements(midi_path: Path) -> pd.DataFrame:

    comps = []
    parts = []
    elements = []

    total_comps = len(list(midi_path.rglob('*.mid')))
    click.secho(
        'Total MIDI files to parse: '+ str(total_comps), color='bright_blue')

    with click.progressbar(
            midi_path.rglob('*.mid'),
            length=total_comps,
            label=click.style('Parsing midi files..',
                              fg='bright_green')) as bar:
        for midi_file in bar:

            try:
                midi_data = converter.parse(midi_file)
            except:
                click.secho(
                    'Error parsing MIDI file: ' + str(midi_file),
                    color='bright_red')
                continue
            midi_parts = instrument.partitionByInstrument(midi_data)

            try:
                for part in midi_parts:

                    part_contents = part.recurse()

                    for element in part_contents:
                        elements.append(element)
                        parts.append(part.id)
                        comps.append(midi_file.stem)

            except TypeError:
                click.secho(
                    'Error in MIDI file: ' + str(midi_file), color='bright_red')
                continue

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


def process_element_data(element_data: pd.DataFrame,
                         classes: List[Any] = [note.Note, note.Rest],
                         durations_to_filter: List[str] = [
                             'breve', 'complex', 'inexpressible', 'longa',
                             'maxima', 'duplex-maxima', 'zero'
                         ],
                         note_filter_min: int = 48,
                         note_filter_max: int = 71):

    # filter for only selected classes
    class_filter = element_data['element'].apply(
        lambda x: x.isClassOrSubclass(classes))
    ed = element_data[class_filter].copy()

    # parse elements for new columns
    tqdm.pandas(desc="Parsing note/rest names")
    ed['note_name'] = ed['element'].progress_apply(
        lambda x: x.nameWithOctave if x.isClassOrSubclass([note.Note]) else 'R'
    )

    tqdm.pandas(desc="Parsing note/rest midi codes")
    ed['note_midi'] = ed['element'].progress_apply(
        lambda x: x.pitch.midi if x.isClassOrSubclass([note.Note]) else 0)

    tqdm.pandas(desc="Parsing note/rest durations")
    ed['duration'] = ed['element'].progress_apply(
        lambda x: x.duration.type)

    # filter out the selected durations
    duration_filter = ed['duration'].isin(durations_to_filter)
    ed2 = ed[~duration_filter].copy()

    # drop the music21 elements column
    ed2 = ed2.drop(columns=['element'])

    note_filter = (ed2['note_midi'] >= 48) & (ed2['note_midi'] <= 71)
    ed3 = ed2[note_filter].copy()

    # prepare training classes
    ed3['note_class'] = ed3['note_name'] + ' - ' + ed3['duration']

    # drop unused columns
    ed3 = ed3.drop(columns=['note_name', 'duration', 'note_midi'])

    # convert classes to category
    ed3['note_class'] = ed3['note_class'].astype('category')

    return ed3


def prepare_seq_data(element_data: pd.DataFrame,
                     seq_size: int = 50) -> pd.DataFrame:

    data = element_data[['comp', 'part', 'note_class']].copy()
    classes = data['note_class']

    # get all pitch names
    pitchnames = sorted(set(item for item in classes))
    # create a dictionary to map pitches to integers
    note_to_int = dict(
        (note, number) for number, note in enumerate(pitchnames))

    data['note'] = data['note_class'].map(note_to_int)
    data = data[['comp', 'part', 'note']]

    seq_size = 100

    columns = ['comp', 'part', 'note'
               ] + ['note_' + str(i).zfill(3) for i in range(1, seq_size + 1)]
    data_seq = pd.DataFrame(columns=columns)

    total_comps = data['comp'].nunique()

    for i, comp in enumerate(data['comp'].unique()):

        lagged_df_comp = data[data['comp'] == comp].copy()

        for part in lagged_df_comp['part'].unique():

            lagged_df_part = lagged_df_comp[lagged_df_comp['part'] ==
                                            part].copy()

            for lag in range(1, seq_size + 1):
                lagged_df_part['note_' + str(lag).zfill(
                    3)] = lagged_df_part['note'].shift(lag)

            data_seq = data_seq.append(lagged_df_part, sort=True)

    return data_seq


def get_midi_gm_mapping(
        midi_gm_mapping_file: Path = params.midi_gm_mapping_file):

    midi_gm_mapping = pd.read_csv(midi_gm_mapping_file)

    return midi_gm_mapping


def dump_data(data_object: Any, out_file: Path) -> bool:

    if out_file.exists():
        out_file.unlink()

    if isinstance(data_object, pd.DataFrame):
        out_file = out_file.parents[0] / (out_file.name + '.pandas_pickle')
        data_object.to_pickle(out_file)
    else:
        out_file = out_file.parents[0] / (out_file.name + '.joblib_dump')
        joblib.dump(data_object, out_file)

    return out_file.exists()
