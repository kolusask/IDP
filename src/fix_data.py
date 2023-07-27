from sys import argv
from os import path, listdir, makedirs
from pandas import read_csv, DataFrame
from pandas.errors import ParserError
from pprint import pprint
from shutil import copyfile


in_dir_path = path.join(argv[1], 'Ampeldaten_20220525_SP')
out_dir_path = path.join(argv[2], 'Ampeldaten_20220525_SP')
detectors = listdir(in_dir_path)
for i, det in enumerate(detectors):
    if det not in ['2040', '4170', '8007']:
        print(f'{det} - {i + 1}/{len(detectors)}')
        days_dir_path = path.join(det, 'Detektorzaehlwerte')
        in_det_path = path.join(in_dir_path, days_dir_path)
        out_det_path = path.join(out_dir_path, days_dir_path)
        if not path.exists(out_det_path):
            makedirs(out_det_path)
        headers = {}
        files_missing_header = []
        files_with_wrong_header = []
        for day in listdir(in_det_path):
            in_day_path = path.join(in_det_path, day)
            out_day_path = path.join(out_det_path, day)
            if det == '3050':
                df = read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python',
                    names=['DATUM', '1(DA1)', '2(DA2)', '4(DAL)', '5(DB1)', '6(DB2)', '7(DB3)', '8(DC1)', '9(DC2)', '10(DD1)'])
            else:
                df = read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python')
            if 'Unnamed' in df.columns[0]:
                df.drop(columns=df.columns[0], axis=1, inplace=True)
            if df.columns[0] != 'DATUM':
                files_missing_header.append(day)
            elif 'Unnamed' in df.columns[1]:
                files_with_wrong_header.append(day)
            else:
                headers[len(df.columns)] = df.columns
                df.set_index('DATUM').to_csv(out_day_path, sep=';')
        assert headers
        out_det_path = path.join(out_dir_path, det, 'Detektorzaehlwerte')
        if not path.exists(out_det_path):
            makedirs(out_det_path)
        for day in files_missing_header:
            in_day_path = path.join(in_det_path, day)
            out_day_path = path.join(out_det_path, day)
            n_columns = len(read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python').columns)
            try:
                header = headers[n_columns]
            except KeyError:
                header = next(h for h in headers.values())
            read_csv(in_day_path, sep=None, names=header, on_bad_lines='skip', engine='python')\
                .set_index('DATUM').to_csv(out_day_path, sep=';')
        for day in files_with_wrong_header:
            in_day_path = path.join(in_det_path, day)
            out_day_path = path.join(out_det_path, day)
            n_columns = len(read_csv(in_day_path, sep=None, names=header, on_bad_lines='skip', engine='python').columns)
            read_csv(in_day_path, sep=None, names=header[n_columns], on_bad_lines='skip', engine='python')\
                .iloc[1:].set_index('DATUM').to_csv(in_day_path, sep=';')
        d20211031 = path.join(out_det_path, 'DetCount_20211031.csv')
        read_csv(d20211031, sep=';').drop(index=list(range(8, 12))).to_csv(d20211031, sep=';')

lut_name = 'Look Up Table (Detectors Mapping on links).xlsx'
copyfile(path.join(argv[1], '..', lut_name), path.join(argv[1], lut_name))
