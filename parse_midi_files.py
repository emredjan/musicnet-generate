from pathlib import Path

import click
import joblib
import pandas as pd

from musicnet.data_utils import get_notes


@click.command()
@click.argument('midi-path', type=click.Path(exists=True))
@click.argument('out-file', type=click.Path())
def main(midi_path, out_file):

    midi_path = Path(midi_path)
    out_file = Path(out_file)


    songs, notes = get_notes(midi_path)

    click.secho('Generating dataframe..', fg='bright_green')
    df = pd.DataFrame(
        {
            'songs': songs,
            'notes': notes
        }
    )

    click.secho('Writing dataframe to file..', fg='bright_green')
    if out_file.exists():
        out_file.unlink()

    joblib.dump(df, out_file)

    click.secho('All done.', fg='bright_green')


if __name__ == '__main__':
    main()  #pylint: disable=E1120
