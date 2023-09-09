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


def prepare_sec_paths(sec: str):
    in_sec_path = path.join(in_dir_path, sec, sec_subdir_name)
    out_sec_path = path.join(out_dir_path, sec, sec_subdir_name)
    if not path.exists(out_sec_path):
        makedirs(out_sec_path)
    
    return in_sec_path, out_sec_path

def process_headers(in_sec_path: str, out_sec_path: str):
    headers = {}
    files_missing_header = []
    files_with_wrong_header = []
    for day in listdir(in_sec_path):
        in_day_path = path.join(in_sec_path, day)
        out_day_path = path.join(out_sec_path, day)
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

def fix_missing_headers(in_sec_path: str, out_sec_path: str, files_missing_header: List[str], headers: Dict[int, str]):
    for day in files_missing_header:
        in_day_path = path.join(in_sec_path, day)
        out_day_path = path.join(out_sec_path, day)
        n_columns = len(read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python').columns)
        try:
            header = headers[n_columns]
        except KeyError:
            header = next(h for h in headers.values())
        read_csv(in_day_path, sep=None, names=header, on_bad_lines='skip', engine='python')\
            .set_index('DATUM').to_csv(out_day_path, sep=';')

def fill_first_rows_with_zeros(in_day_path: str, n_zeros: int):
    with open(in_day_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines[1:], start=1):
        values = line.split(';')
        if len(values) == 2:
            lines[i] = values[0] + ';' + ';'.join('0' * n_zeros) + '\n'
    with open(in_day_path, 'w') as file:
        file.write(''.join(lines))

def fix_wrong_headers(in_sec_path: str, out_sec_path: str, files_with_wrong_header: List[str], headers: Dict[int, str]):
    assert len(headers) == 1
    header = headers[list(headers.keys())[0]]
    for day in files_with_wrong_header:
        in_day_path = path.join(in_sec_path, day)
        out_day_path = path.join(out_sec_path, day)
        fill_first_rows_with_zeros(in_day_path, list(headers.keys())[0] - 1)
        with open(in_day_path, 'r') as csv:
            text = csv.read()
        n_columns = len(text.split('\n')[1].split(';'))
        read_csv(StringIO(text), sep=None, names=headers[n_columns], on_bad_lines='skip', engine='python')\
            .iloc[1:].set_index('DATUM').to_csv(out_day_path, sep=';')

def process_section(sec: str):
    in_sec_path, out_sec_path = prepare_sec_paths(sec)
    headers, files_missing_header, files_with_wrong_header = process_headers(in_sec_path, out_sec_path)

    fix_missing_headers(in_sec_path, out_sec_path, files_missing_header, headers)
    fix_wrong_headers(in_sec_path, out_sec_path, files_with_wrong_header, headers)

    d20211031 = path.join(out_sec_path, 'DetCount_20211031.csv')
    read_csv(d20211031, sep=';').drop(index=list(range(8, 12))).to_csv(d20211031, sep=';')


sec_3050_20211017_path = path.join(in_dir_path, '3050', sec_subdir_name, 'DetCount_20211017.csv')
with open(sec_3050_20211017_path, 'r') as sec_3050_20211017_file:
    sec_3050_20211017_lines = sec_3050_20211017_file.readlines()
sec_3050_20211017_lines[0] = 'DATUM;1(DA1);2(DA2);4(DAL);5(DB1);6(DB2);7(DB3);8(DC1);9(DC2);10(DD1)\n'
with open(sec_3050_20211017_path, 'w') as sec_3050_20211017_file:
    sec_3050_20211017_file.write(''.join(sec_3050_20211017_lines))

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

# fill_first_rows_with_zeros('3050', '20211017', 9)
# fill_first_rows_with_zeros('3060', '20210621', 10)
# fill_first_rows_with_zeros('0036', '20211025', 6)

sections = listdir(in_dir_path)
ignore = []
ignore += ['2040', '4170', '8007']  # empty
ignore += ['6021']                  # conflicting headers
for i, sec in enumerate(sections, start=1):
    if sec in ignore:
        print(f'{sec} - {i}/{len(sections)} - ignored')
    else:
        print(f'{sec} - {i}/{len(sections)}')
        process_section(sec)
split_section('3060', {
    '3060': ['1(DA1)', '2(DA2)', '3(DB1)'],
    '3060_G': ['4(DD1)', '5(DD2)', '6(DE1)', '9(DF2)', '10(DF1)'],
    '3060_L': ['7(DC1)', '8(DC2)'],
    })

lut_name = 'Look Up Table (Detectors Mapping on links).xlsx'
copyfile(path.join(argv[1], '..', lut_name), path.join(argv[1], lut_name))
