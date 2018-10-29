from pathlib import Path
import yaml

with open('parameters.yml', 'r') as stream:
    try:
        p = yaml.load(stream)
    except yaml.YAMLError as exc:
        print('Error parsing YAML parameters file.')

# file locations
fl = p['file_locations']
metadata_file = Path(fl['metadata_file'])
midi_gm_mapping_file = Path(fl['midi_gm_mapping_file'])
musicnet_data = Path(fl['musicnet_data'])
midi_samples = Path(fl['midi_samples'])
midi_musicnet = Path(fl['midi_musicnet'])

label_data = Path(fl['interim_data']['label_data'])
element_data = Path(fl['interim_data']['element_data'])

# musicnet
mn = p['musicnet']
sample_rate = mn['sample_rate']
resolution = mn['resolution']
