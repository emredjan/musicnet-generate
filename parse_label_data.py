from pathlib import Path

import click
import joblib
import pandas as pd

from musicnet import data_utils, mn_utils, params


@click.command()
@click.option('--musicnet-data', '-m', type=click.Path(exists=True))
@click.option('--out-file', '-o', type=click.Path())
def main(musicnet_data, out_file):

    musicnet_data = Path(
        musicnet_data) if musicnet_data else params.musicnet_data
    out_file = Path(out_file) if out_file else params.label_data

    data = mn_utils.load_musicnet_data(musicnet_data)
    label_data = mn_utils.prepare_label_data(data)
    comp_df = mn_utils.export_label_data(label_data)

    click.secho('Writing dataframe..', fg='bright_green')

    data_dumped = data_utils.dump_data(comp_df, out_file)

    if data_dumped:
        click.secho('All done.', fg='bright_green')
    else:
        click.secho('Problem Writing DataFrame.', fg='bright_red')


if __name__ == '__main__':
    main()  #pylint: disable=E1120
