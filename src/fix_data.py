from io import StringIO
from sys import argv
from os import path, listdir, makedirs
from pandas import read_csv, DataFrame
from pandas.errors import ParserError
from pprint import pprint
from shutil import copyfile, move, rmtree
from typing import List, Dict


in_dir_path = path.join(argv[1], 'Ampeldaten_20220525_SP')
out_dir_path = path.join(argv[2], 'Ampeldaten_20220525_SP')
sec_subdir_name = 'Detektorzaehlwerte'


def prepare_det_paths(det: str):
    days_dir_path = path.join(det, sec_subdir_name)
    in_det_path = path.join(in_dir_path, days_dir_path)
    out_det_path = path.join(out_dir_path, days_dir_path)
    if not path.exists(out_det_path):
        makedirs(out_det_path)
    
    return in_det_path, out_det_path

def process_headers(in_det_path: str, out_det_path: str):
    headers = {}
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
            headers[len(df.columns)] = df.columns
            df.set_index('DATUM').to_csv(out_day_path, sep=';')
    assert headers

    return headers, files_missing_header, files_with_wrong_header

def fix_missing_headers(in_det_path: str, out_det_path: str, files_missing_header: List[str], headers: Dict[int, str]):
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

def fix_wrong_headers(in_det_path: str, files_with_wrong_header: List[str], headers: Dict[int, str]):
    for day in files_with_wrong_header:
        in_day_path = path.join(in_det_path, day)
        with open(in_day_path, 'r') as csv:
            text = csv.read()
        n_columns = len(text.split('\n')[1].split(';'))
        read_csv(StringIO(text), sep=None, names=headers[n_columns], on_bad_lines='skip', engine='python')\
            .iloc[1:].set_index('DATUM').to_csv(in_day_path, sep=';')

def process_detector(det: str):
    in_det_path, out_det_path = prepare_det_paths(det)
    headers, files_missing_header, files_with_wrong_header = process_headers(in_det_path, out_det_path)

    fix_missing_headers(in_det_path, out_det_path, files_missing_header, headers)
    fix_wrong_headers(in_det_path, files_with_wrong_header, headers)

    d20211031 = path.join(out_det_path, 'DetCount_20211031.csv')
    read_csv(d20211031, sep=';').drop(index=list(range(8, 12))).to_csv(d20211031, sep=';')


det_3050_20211017_path = path.join(in_dir_path, '3050', sec_subdir_name, 'DetCount_20211017.csv')
with open(det_3050_20211017_path, 'r') as det_3050_20211017_file:
    det_3050_20211017_lines = det_3050_20211017_file.readlines()
det_3050_20211017_lines[0] = 'DATUM;1(DA1);2(DA2);4(DAL);5(DB1);6(DB2);7(DB3);8(DC1);9(DC2);10(DD1)\n'
# det_3050_20211017_lines[:5] = [
#     '2021-10-17 00:00:00;0;0;0;0;0;0;0;0;0',
#     '2021-10-17 00:15:00;0;0;0;0;0;0;0;0;0',
#     '2021-10-17 00:30:00;0;0;0;0;0;0;0;0;0',
#     '2021-10-17 00:45:00;0;0;0;0;0;0;0;0;0',
# ]
with open(det_3050_20211017_path, 'w') as det_3050_20211017_file:
    det_3050_20211017_file.write(''.join(det_3050_20211017_lines))

def split_section(in_sec: str, split_groups: Dict[str, List[str]], remove_original=True):
    temp_dir_path = path.join(out_dir_path, in_sec + '_unsplit')
    in_sec_dir_path = path.join(temp_dir_path, sec_subdir_name)
    if not path.exists(in_sec_dir_path):
        in_sec_dir_old_path = path.join(out_dir_path, in_sec, sec_subdir_name)
        move(in_sec_dir_old_path, in_sec_dir_path)
    for day in listdir(in_sec_dir_path):
        in_day_path = path.join(in_sec_dir_path, day)
        df = read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python')
        for out_sec, detectors in split_groups.items():
            out_sec_dir_path = path.join(out_dir_path, out_sec, sec_subdir_name)
            if not path.exists(out_sec_dir_path):
                makedirs(out_sec_dir_path)
            df[['DATUM'] + detectors].set_index('DATUM').to_csv(path.join(out_sec_dir_path, day))
    if remove_original:
        rmtree(temp_dir_path)

def fill_first_rows_with_zeros(sec: str, day: str, n_zeros: int):
    file_path = path.join(in_dir_path, sec, sec_subdir_name, f'DetCount_{day}.csv')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines[1:], start=1):
        values = line.split(';')
        if len(values) == 2:
            lines[i] = values[0] + ';' + ';'.join('0' * n_zeros) + '\n'
    with open(file_path, 'w') as file:
        file.write(''.join(lines))

fill_first_rows_with_zeros('3050', '20211017', 9)
fill_first_rows_with_zeros('3060', '20210621', 10)

detectors = listdir(in_dir_path)
ignore = []
ignore += ['2040', '4170', '8007']  # empty
# for i, det in enumerate(detectors):
for i, det in enumerate(['3050', '3060']):
    if det not in ignore:
        print(f'{det} - {i + 1}/{len(detectors)}')
        process_detector(det)
split_section('3060', {
    '3060': ['1(DA1)', '2(DA2)', '3(DB1)'],
    '3060_G': ['4(DD1)', '5(DD2)', '6(DE1)', '9(DF2)', '10(DF1)'],
    '3060_L': ['7(DC1)', '8(DC2)'],
    })

lut_name = 'Look Up Table (Detectors Mapping on links).xlsx'
copyfile(path.join(argv[1], '..', lut_name), path.join(argv[1], lut_name))
