from pathlib import Path

import click
import joblib
import pandas as pd

from musicnet import data_utils, params


@click.command()
@click.option('--midi-path', '-m', type=click.Path(exists=True))
@click.option('--out-file-ed', '-o1', type=click.Path())
@click.option('--out-file-edp', '-o2', type=click.Path())
def main(midi_path, out_file_ed, out_file_edp):

    midi_path = Path(midi_path) if midi_path else params.midi_musicnet
    out_file_ed = Path(out_file_ed) if out_file_ed else params.element_data
    out_file_edp = Path(out_file_edp) if out_file_edp else params.element_data_processed

    element_data = data_utils.get_midi_elements(midi_path)
    element_data_processed = data_utils.process_element_data(element_data)

    click.secho('Writing dataframe..', fg='bright_green')

    data_dumped_ed = data_utils.dump_data(element_data, out_file_ed)
    data_dumped_edp = data_utils.dump_data(element_data_processed, out_file_edp)

    if data_dumped_ed & data_dumped_edp:
        click.secho('All done.', fg='bright_green')
    else:
        click.secho('Problem Writing DataFrame.', fg='bright_red')

if __name__ == '__main__':
    main()  #pylint: disable=E1120
