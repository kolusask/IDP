import os
import sys

from json import load


with open('config.json') as config:
    CONFIG = load(config)

OUTPUTS_FOLDER = 'outputs/' + CONFIG['CONFIG_NAME']

DATE_FORMAT = '%Y-%m-%d'

if not os.path.exists(OUTPUTS_FOLDER):
    os.makedirs(OUTPUTS_FOLDER)

def out_path(p: str):
    return os.path.join(OUTPUTS_FOLDER, p)


if 'src' not in sys.path:
    sys.path.append('src')
