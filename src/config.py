from json import load

with open('config.json') as config:
    CONFIG = load(config)

SPECTRAL_THRESHOLD = CONFIG['SPECTRAL_THRESHOLD']
HIDDEN_LAYER_NUMBER = CONFIG['HIDDEN_LAYER_NUMBER']
