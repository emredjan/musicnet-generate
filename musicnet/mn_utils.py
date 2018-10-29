from pathlib import Path
from typing import List, Dict, Set, Any

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervaltree import Interval, IntervalTree

from musicnet import params, data_utils


def get_musicnet_metadata(
        metadata_file: Path = params.metadata_file) -> pd.DataFrame:

    metadata = pd.read_csv(metadata_file, index_col='id')

    return metadata


def load_musicnet_data(
        musicnet_data: Path = params.musicnet_data) -> pd.DataFrame:

    data = np.load(open(musicnet_data, 'rb'), encoding='latin1')

    return data


def plot_recording(audio_data: np.ndarray,
                   sample_rate: int = params.sample_rate,
                   seconds: int = None,
                   figwidth: int = 20,
                   figheight: int = 2) -> None:

    fig = plt.figure(figsize=(figwidth, figheight))

    if seconds:
        plt.plot(audio_data[0:seconds * sample_rate])
    else:
        plt.plot(audio_data)

    fig.axes[0].set_xlabel(f'Sample ({sample_rate} Hz)')
    fig.axes[0].set_ylabel('Amplitude')
    plt.show()


def get_instrument_list(label_data: IntervalTree) -> Set[int]:

    instruments = set()
    for interval in label_data:
        instruments.add(interval[2][0])

    return instruments


def plot_piano_roll(audio_data: np.ndarray,
                    label_data: IntervalTree,
                    sample_rate: int = params.sample_rate,
                    seconds: int = None,
                    resolution: int = params.resolution,
                    figwidth: int = 20,
                    figheight: int = 5) -> None:

    windows_ps = sample_rate / float(resolution)

    if seconds:
        plot_length = seconds
    else:
        plot_length = int(round(len(audio_data) / sample_rate, 0))

    # 128 distinct note labels
    window_range = np.zeros((int(plot_length * windows_ps), 128))

    instruments = get_instrument_list(label_data)
    colors = {}
    color_range = np.random.uniform(0.2, 1, len(instruments))
    for i, inst in enumerate(instruments):
        colors[inst] = color_range[i]

    for window in range(len(window_range)):
        labels = label_data[window * resolution]
        for label in labels:
            window_range[window, label.data[1]] = colors[label.data[0]]

    fig = plt.figure(figsize=(figwidth, figheight))

    plt.imshow(window_range.T, aspect='auto', cmap='ocean_r')

    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('Window')
    fig.axes[0].set_ylabel('Note (MIDI code)')

    plt.show()


def prepare_label_data(musicnet_data: np.lib.npyio.NpzFile) -> Dict[str, Any]:

    comps = []
    starts = []
    ends = []
    instruments = []
    notes = []
    measures = []
    beats = []
    note_values = []

    with click.progressbar(
            musicnet_data.items(), label='Processing labels..') as bar:
        for musicnet_id, comp in bar:

            labels = comp[1]

            for interval in sorted(labels):

                comps.append(musicnet_id)
                starts.append(interval[0])
                ends.append(interval[1])
                instruments.append(interval[2][0])
                notes.append(interval[2][1])
                measures.append(interval[2][2])
                beats.append(interval[2][3])
                note_values.append(interval[2][4])

    label_data = {
        'comp': comps,
        'start': starts,
        'end': ends,
        'instrument': instruments,
        'note': notes,
        'measure': measures,
        'beat': beats,
        'note_value': note_values,
        'event': 'note_on',
        'event_sample': starts
    }

    return label_data


def export_label_data(label_data_dict: Dict[str, Any],
                      sample_rate: int = params.sample_rate) -> pd.DataFrame:

    comp_df = pd.DataFrame(label_data_dict)

    # make a copy of all data, for 'note off' events
    comp_df_note_off = comp_df.copy()
    comp_df_note_off.loc[:, 'event'] = 'note_off'
    comp_df_note_off.loc[:, 'event_sample'] = comp_df_note_off['end']

    comp_df = comp_df.append(comp_df_note_off)
    comp_df = comp_df.reset_index(drop=True)

    # convert categorical columns to appropriate type
    categ_columns = ['event', 'note_value', 'comp']
    for col in categ_columns:
        comp_df[col] = comp_df[col].astype('category')

    comp_df = comp_df.sort_values(['comp', 'instrument', 'event_sample'])

    # calculate tick offsets
    comp_df['tick'] = (comp_df.event_sample * 1000 / sample_rate).diff()
    comp_df.loc[comp_df['tick'] < 0, 'tick'] = np.nan

    tick_filter = comp_df['tick'].isna()
    comp_df.loc[tick_filter, 'tick'] = comp_df.loc[
        tick_filter, 'event_sample'] * 1000 / sample_rate

    comp_df['tick'] = comp_df['tick'].round(0).astype(int)

    return comp_df
