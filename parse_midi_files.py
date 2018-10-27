from pathlib import Path

import click
import joblib

from musicnet.data_utils import get_notes


@click.command()
@click.argument('midi-path', type=click.Path(exists=True))
@click.argument('out-file', type=click.Path())
def main(midi_path, out_file):

    midi_path = Path(midi_path)
    out_file = Path(out_file)

    notes = get_notes(midi_path)

    if out_file.exists:
        out_file.unlink()

    joblib.dump(notes, out_file)


if __name__ == '__main__':
    main()  #pylint: disable=E1120
