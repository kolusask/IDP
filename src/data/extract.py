from globals import Period

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from os.path import join
from os import listdir


def norm_int_id(id):
    try:
        return f'{int(id):04d}'
    except ValueError:
        return id


class DetectorDataProvider:
    def __init__(self, data_path: str):
        self.data_path = join(data_path, 'Ampeldaten_20220525_SP')

    def list_intersections(self):
        return listdir(self.data_path)
    
    def get_data_for_day(self, int_id: int, day: int, month: int):
        int_id = str(int_id)
        date = datetime(year=2021, month=month, day=day)
        file_name = date.strftime('DetCount_%Y%m%d.csv')
        file_path = join(self.data_path, int_id, 'Detektorzaehlwerte',
            file_name)
        # TODO Preprocess it in fix_headers.py
        timestamps = [datetime(year=2021, month=month, day=day) + timedelta(minutes=m) \
                      for m in range(0, 24 * 60, 15)]
        timestamps = pd.DataFrame(timestamps, columns=['DATUM'])

        try:
            csv = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')
        except FileNotFoundError:
            return timestamps

        csv['DATUM'] = pd.to_datetime(csv['DATUM'], errors='coerce')
        csv = csv.merge(timestamps, how='outer', on='DATUM')

        return csv
    
    def get_data_for_period(self, int_id, period: Period):
        start, end = period
        int_id = norm_int_id(int_id)
        data = []
        for date in (start + timedelta(days=n) for n in range((end - start).days)):
            data.append(self.get_data_for_day(int_id, date.day, date.month))
            
        return pd.concat(data).set_index('DATUM')
    
    def get_counts_entering_section(self, section_end, detectors, period: Period):
        section_data = self.get_data_for_period(section_end, period)
        for col in section_data.columns:
            section_data[col] = pd.to_numeric(section_data[col], errors='coerce')

        return list(section_data[detectors].fillna(0).sum(axis=1, numeric_only=True))


class LookUpTable:
    def __init__(self, data_path):
        ignore = (
            '2040',                 # missing folder - empty in the cloud
            '4170',                 # missing folder - empty in the cloud
            '7',                    # no traffic detector
            '8007',                 # missing folder - empty in the cloud
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
    
    def get_sections(self):
        sections = set()
        for inter in self.list_intersections():
            detectors = self.get_detectors_on(inter)
            for sec in detectors[['Starting Intersection', 'Ending Intersection']].values:
                sections.add(tuple(sorted(sec)))
        
        return sections
    
    def get_detectors_per_section(self):
        int_det = []
        for int_1, int_2 in self.get_sections():
            det_1_2, det_2_1 = self.get_detectors_between(int_1, int_2)
            int_det.append((int_1, int_2, det_1_2))
            int_det.append((int_2, int_1, det_2_1))
        int_det = pd.DataFrame(int_det, columns=['Start', 'End', 'Detectors'])

        return int_det
