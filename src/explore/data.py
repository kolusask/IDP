import pandas as pd
import numpy as np

from os import listdir
from os.path import join
from datetime import datetime, timedelta


def norm_int_id(id):
    try:
        return f'{int(id):04d}'
    except ValueError:
        return id


class DetectorDataProvider:
    def __init__(self, data_path: str):
        self.data_path = join(data_path, 'Ampeldaten_20220525_SP')
        self.DELETEME = set()

    def list_intersections(self):
        return listdir(self.data_path)
    
    def get_data_for_day(self, int_id: int, day: int, month: int):
        int_id = str(int_id)
        date = datetime(year=2021, month=month, day=day)
        file_name = date.strftime('DetCount_%Y%m%d.csv')
        file_path = join(self.data_path, int_id, 'Detektorzaehlwerte',
            file_name)
        # TODO Preprocess it in fix_headers.py
        timestamps = [datetime(year=2021, month=month, day=day) + timedelta(minutes=m) for m in range(0, 24 * 60, 15)]
        timestamps = pd.DataFrame(timestamps, columns=['DATUM'])

        try:
            csv = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')
        except FileNotFoundError:
            return timestamps

        csv['DATUM'] = pd.to_datetime(csv['DATUM'], errors='coerce')
        csv = csv.merge(timestamps, how='outer', on='DATUM')

        return csv
    
    def get_data_for_period(self, int_id, start: datetime, end: datetime):
        int_id = norm_int_id(int_id)
        data = []
        for date in (start + timedelta(days=n) for n in range((end - start).days)):
            data.append(self.get_data_for_day(int_id, date.day, date.month))
        return pd.concat(data).set_index('DATUM')


class LookUpTable:
    def __init__(self, data_path):
        ignore = (
            '2040',                 # missing folder
            '26',                   # conflicting detector names - 2(DC1) vs 2(DB1)
            '3050',                 # missing header
            '3060_G',               # confusing - discuss
            '3060_L',               # confusing - discuss
            '4170',                 # missing folder
            '42',                   # conflicting detector names
            '5090_Teilknoten 1',    # confusing - discuss
            '5090_Teilknoten 2',    # confusing - discuss
            '5090_Teilknoten 3',    # confusing - discuss
            '5090_Teilknoten 4',    # confusing - discuss
            '6010_F1a',             # -//-
            '6010_F1b',             # -//-
            '6060',                 # conflicting detector names - 13(DA4) vs 13(DD4)
            '7',                    # no traffic detector
            '8007',                 # missing folder
            'nan',                  # ???
        )
        data_path = join(data_path,
            'Look Up Table (Detectors Mapping on links).xlsx')
        self.lookup_table = pd.read_excel(data_path).astype(str)
        self.lookup_table = self.lookup_table[
            ~self.lookup_table['Ending Intersection'].isin(ignore)
        ][
            ~self.lookup_table['Starting Intersection'].isin(ignore)
        ]
    
    def list_intersections(self):
        return np.unique(np.concatenate([
            self.lookup_table['Starting Intersection'].unique(),
            self.lookup_table['Ending Intersection'].unique()
        ]))

    def get_detectors_on(self, intersection):
        return pd.concat([
            self.get_detectors_from(intersection),
            self.get_detectors_to(intersection)
        ])
    
    def get_detectors_between(self, int_1_id, int_2_id):
        detectors_from_1 = self.get_detectors_from(int_1_id)
        detectors_from_2 = self.get_detectors_from(int_2_id)
        detectors_from_1 = detectors_from_1[
            detectors_from_1['Ending Intersection'] == int_2_id]
        detectors_from_2 = detectors_from_2[
            detectors_from_2['Ending Intersection'] == int_1_id]
        return (
            detectors_from_1['Detector'].tolist(),
            detectors_from_2['Detector'].tolist()
        )

    def get_detectors_from(self, int_id):
        int_id = str(int_id)
        return self.lookup_table.loc[
            self.lookup_table['Starting Intersection'] == int_id]
    
    def get_detectors_to(self, int_id):
        int_id = str(int_id)
        return self.lookup_table.loc[
            self.lookup_table['Ending Intersection'] == int_id]

    def get_sources(self):
        return iter(self.lookup_table['Starting Intersection'].unique())
    
    def get_destinations(self):
        return iter(self.lookup_table['Ending Intersection'].unique())
