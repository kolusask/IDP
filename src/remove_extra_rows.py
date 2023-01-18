from os import path, listdir
from sys import argv

dirpath = path.join(argv[1], 'Ampeldaten_20220525_SP')
for det in listdir(dirpath):
    filepath = path.join(dirpath, det, 'Detektorzaehlwerte/DetCount_20211031.csv')
    with open(filepath, 'r') as file:
        lines = file.readlines()
    if len(lines) > 98:
        with open(filepath, 'w') as file:
            file.write(''.join(lines[:10] + lines[14:]))

