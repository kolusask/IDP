from sys import argv
from os import path, listdir, makedirs
from pandas import read_csv, DataFrame
from pandas.errors import ParserError
from pprint import pprint


in_dir_path = path.join(argv[1], 'Ampeldaten_20220525_SP')
out_dir_path = path.join(argv[2], 'Ampeldaten_20220525_SP')
detectors = listdir(in_dir_path)
for i, det in enumerate(detectors):
    if det not in ['3050']:
        print(f'{det} - {i + 1}/{len(detectors)}')
        days_dir_path = path.join(det, 'Detektorzaehlwerte')
        in_det_path = path.join(in_dir_path, days_dir_path)
        out_det_path = path.join(out_dir_path, days_dir_path)
        if not path.exists(out_det_path):
            makedirs(out_det_path)
        header = None
        files_missing_header = []
        files_with_wrong_header = []
        for day in listdir(in_det_path):
            in_day_path = path.join(in_det_path, day)
            out_day_path = path.join(out_det_path, day)
            df = read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python')
            if 'Unnamed' in df.columns[0]:
                df.drop(columns=df.columns[0], axis=1, inplace=True)
            if df.columns[0] != 'DATUM':
                files_missing_header.append(day)
            elif 'Unnamed' in df.columns[1]:
                files_with_wrong_header.append(day)
            else:
                header = df.columns
                df.set_index('DATUM').to_csv(out_day_path, sep=';')
        assert header is not None
        out_det_path = path.join(out_dir_path, det, 'Detektorzaehlwerte')
        if not path.exists(out_det_path):
            makedirs(out_det_path)
        for day in files_missing_header:
            in_day_path = path.join(in_det_path, day)
            out_day_path = path.join(out_det_path, day)
            read_csv(in_day_path, sep=None, names=header, on_bad_lines='skip', engine='python')\
                .set_index('DATUM').to_csv(out_day_path, sep=';')
        for day in files_with_wrong_header:
            in_day_path = path.join(in_det_path, day)
            out_day_path = path.join(out_det_path, day)
            read_csv(in_day_path, sep=None, names=header, on_bad_lines='skip', engine='python')\
                .iloc[1:].set_index('DATUM').to_csv(in_day_path, sep=';')

