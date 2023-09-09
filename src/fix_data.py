from io import StringIO
from sys import argv
from os import path, listdir, makedirs

from pandas import read_csv

from shutil import copyfile, move, rmtree
from typing import List, Dict


in_dir_path = path.join(argv[1], 'Ampeldaten_20220525_SP')
out_dir_path = path.join(argv[2], 'Ampeldaten_20220525_SP')
int_subdir_name = 'Detektorzaehlwerte'


def prepare_int_paths(int: str):
    in_int_path = path.join(in_dir_path, int, int_subdir_name)
    out_int_path = path.join(out_dir_path, int, int_subdir_name)
    if not path.exists(out_int_path):
        makedirs(out_int_path)
    
    return in_int_path, out_int_path

def process_headers(in_int_path: str, out_int_path: str):
    files_missing_header = []
    files_with_wrong_header = []
    header = None
    for day in listdir(in_int_path):
        in_day_path = path.join(in_int_path, day)
        out_day_path = path.join(out_int_path, day)
        df = read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python')
        if 'Unnamed' in df.columns[0]:
            df.drop(columns=df.columns[0], axis=1, inplace=True)
        if df.columns[0] != 'DATUM':
            files_missing_header.append(day)
        elif 'Unnamed' in df.columns[1]:
            files_with_wrong_header.append(day)
        else:
            if header is None:
                header = list(df.columns)
            else:
                assert header == list(df.columns)
            df.set_index('DATUM').to_csv(out_day_path, sep=';')
    assert header

    return header, files_missing_header, files_with_wrong_header

def fix_missing_headers(in_int_path: str, out_int_path: str, files_missing_header: List[str], header: List[str]):
    for day in files_missing_header:
        in_day_path = path.join(in_int_path, day)
        out_day_path = path.join(out_int_path, day)
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

def fix_wrong_headers(in_int_path: str, out_int_path: str, files_with_wrong_header: List[str], header: List[str]):
    for day in files_with_wrong_header:
        in_day_path = path.join(in_int_path, day)
        out_day_path = path.join(out_int_path, day)
        fill_first_rows_with_zeros(in_day_path, len(header) - 1)
        with open(in_day_path, 'r') as csv:
            text = csv.read()
        n_columns = len(text.split('\n')[1].split(';'))
        read_csv(StringIO(text), sep=None, names=header, on_bad_lines='skip', engine='python')\
            .iloc[1:].set_index('DATUM').to_csv(out_day_path, sep=';')

def process_intersection(int: str):
    in_int_path, out_int_path = prepare_int_paths(int)
    header, files_missing_header, files_with_wrong_header = process_headers(in_int_path, out_int_path)

    fix_missing_headers(in_int_path, out_int_path, files_missing_header, header)
    fix_wrong_headers(in_int_path, out_int_path, files_with_wrong_header, header)

    d20211031 = path.join(out_int_path, 'DetCount_20211031.csv')
    read_csv(d20211031, sep=';').drop(index=list(range(8, 12))).to_csv(d20211031, sep=';')


int_3050_20211017_path = path.join(in_dir_path, '3050', int_subdir_name, 'DetCount_20211017.csv')
with open(int_3050_20211017_path, 'r') as int_3050_20211017_file:
    int_3050_20211017_lines = int_3050_20211017_file.readlines()
int_3050_20211017_lines[0] = 'DATUM;1(DA1);2(DA2);4(DAL);5(DB1);6(DB2);7(DB3);8(DC1);9(DC2);10(DD1)\n'
with open(int_3050_20211017_path, 'w') as int_3050_20211017_file:
    int_3050_20211017_file.write(''.join(int_3050_20211017_lines))

def split_intersection(in_int: str, split_groups: Dict[str, List[str]], remove_original=True):
    print(f'Splitting {in_int}...', end='')
    temp_dir_path = path.join(out_dir_path, in_int + '_unsplit')
    in_int_dir_path = path.join(temp_dir_path, int_subdir_name)
    if not path.exists(in_int_dir_path):
        in_int_dir_old_path = path.join(out_dir_path, in_int, int_subdir_name)
        move(in_int_dir_old_path, in_int_dir_path)
    for day in listdir(in_int_dir_path):
        in_day_path = path.join(in_int_dir_path, day)
        df = read_csv(in_day_path, sep=None, on_bad_lines='skip', engine='python')
        for out_int, detectors in split_groups.items():
            out_int_dir_path = path.join(out_dir_path, out_int, int_subdir_name)
            if not path.exists(out_int_dir_path):
                makedirs(out_int_dir_path)
            df[['DATUM'] + detectors].set_index('DATUM').to_csv(path.join(out_int_dir_path, day))
    if remove_original:
        rmtree(temp_dir_path)
    print('Done')

intersections = listdir(in_dir_path)
ignore = []
ignore += ['2040', '4170', '8007']  # empty
ignore += ['6021']                  # conflicting headers
for i, int in enumerate(intersections, start=1):
    if int in ignore:
        print(f'{int} - {i}/{len(intersections)} - ignored')
    else:
        print(f'{int} - {i}/{len(intersections)}')
        process_intersection(int)
split_intersection('3060', {
    '3060': ['1(DA1)', '2(DA2)', '3(DB1)'],
    '3060_G': ['4(DD1)', '5(DD2)', '6(DE1)', '9(DF2)', '10(DF1)'],
    '3060_L': ['7(DC1)', '8(DC2)'],
    })
split_intersection('5090', {
    '5090': ['2(DA2)', '3(DB1)', '4(DB2)', '5(DC1)', '6(DC2)'],
    '5090_Teilknoten 1': ['7(DD1)', '8(DE1)', '9(DE2)'],
    '5090_Teilknoten 2': ['1(DA1)'],
})
split_intersection('6010', {
    '6010_F1a': ['1(RES1)', '2(DE1)', '3(DE2)', '4(DH1)', '5(DH2)', '6(DDL1)', '7(DDL2)'],
    '6010_F1b': ['8(DAL1)', '9(RES2)', '10(DBR1)', '11(DBR2)'],
})


lut_name = 'Look Up Table (Detectors Mapping on links).xlsx'
copyfile(path.join(argv[1], '..', lut_name), path.join(argv[1], lut_name))
