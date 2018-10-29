from pathlib import Path

import click
import joblib
import pandas as pd

from musicnet import data_utils, params


@click.command()
@click.option('--midi-path', '-m', type=click.Path(exists=True))
@click.option('--out-file', '-o', type=click.Path())
def main(midi_path, out_file):

    midi_path = Path(midi_path) if midi_path else params.midi_musicnet
    out_file = Path(out_file) if out_file else params.element_data

    element_data = data_utils.get_midi_elements(midi_path, out_file)

    click.secho('Writing dataframe..', fg='bright_green')

    data_dumped = data_utils.dump_data(element_data, out_file)

    if data_dumped:
        click.secho('All done.', fg='bright_green')
    else:
        click.secho('Problem Writing DataFrame.', fg='bright_red')

if __name__ == '__main__':
    main()  #pylint: disable=E1120
