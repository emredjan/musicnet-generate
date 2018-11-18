import datetime
from pathlib import Path
from typing import Any, List

import click
import joblib
import pandas as pd
from music21 import chord, converter, instrument, note, duration
from tqdm import tqdm

from musicnet import params


def to_hms(sec: float):
    '''convert seconds to H:M:S'''
    return str(datetime.timedelta(seconds=round(sec, 0)))


def get_midi_elements(midi_path: Path) -> pd.DataFrame:

    comps = []
    parts = []
    e_types = []
    e_names = []
    e_durations = []
    e_pitches = []
    e_velocities = []

    total_comps = len(list(midi_path.rglob('*.mid')))
    total_size = sum(m.stat().st_size for m in midi_path.rglob('*.mid'))
    click.secho(
        'Total MIDI files to parse: '+ str(total_comps), color='bright_blue')

    parse_errors = 0
    midi_errors = 0

    with click.progressbar(
            midi_path.rglob('*.mid'),
            length=total_size,
            label=click.style('Parsing midi files..',
                              fg='bright_green')) as bar:
        for midi_file in bar:

            try:
                midi_data = converter.parse(midi_file)
            except:
                parse_errors += 1
                continue

            midi_parts = instrument.partitionByInstrument(midi_data)

            try:
                for part in midi_parts:

                    part_contents = part.recurse()

                    for element in part_contents:

                        if element.isClassOrSubclass([note.Note]):
                            e_type = 'note'
                            e_name = element.nameWithOctave
                            e_pitch = element.pitch.midi
                            e_velocity = element.volume.velocity
                        elif element.isClassOrSubclass([chord.Chord]):
                            e_type = 'chord'
                            e_name = element.pitchedCommonName
                            e_pitch = ' '.join([str(p.midi) for p in element.pitches])
                            e_velocity = element.volume.velocity
                        elif element.isClassOrSubclass([note.Rest]):
                            e_type = 'rest'
                            e_name = 'R'
                            e_pitch = None
                            e_velocity = 0
                        else:
                            continue

                        try:
                            e_duration = element.duration.type
                        except duration.DurationException:
                            continue

                        # element specific
                        e_types.append(e_type)
                        e_names.append(e_name)
                        e_pitches.append(e_pitch)
                        e_velocities.append(e_velocity)

                        # common to all elements
                        e_durations.append(e_duration)
                        parts.append(part.id)
                        comps.append(midi_file.stem)

            except TypeError:
                midi_errors += 1
                continue

            bar.update(midi_file.stat().st_size)

    element_data = pd.DataFrame(
        {
            'comp': comps,
            'part': parts,
            'element_type': e_types,
            'element_name': e_names,
            'element_duration': e_durations,
            'element_pitch': e_pitches,
            'element_velocity': e_velocities
        }
    )

    # convert categorical columns to appropriate type
    categ_columns = element_data.columns
    for col in categ_columns:
        element_data[col] = element_data[col].astype('category')

    click.secho('# parsing errors: ' + str(parse_errors), color='bright_red')
    click.secho('# invalid midi files: ' + str(midi_errors), color='bright_red')
    click.secho(
        '# files parsed: ' + str(total_comps - parse_errors - midi_errors),
        color='bright_red')

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

    note_filter = (ed2['note_midi'] >= note_filter_min) & (ed2['note_midi'] <=
                                                           note_filter_max)
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

    for comp in data['comp'].unique():

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
