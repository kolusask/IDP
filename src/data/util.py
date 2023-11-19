from globals import *

import torch

from datetime import datetime, timedelta
from typing import Tuple


POINTS_PER_HOUR = 60 // 15
POINTS_PER_DAY = POINTS_PER_HOUR * 24


def count_points_in_period(period: Period):
    start, end = period
    return int((end - start).total_seconds()
               // 3600 * POINTS_PER_HOUR)


def crop_q_between(mat_q, old_period: Period, new_period: Period):
    old_start, old_end = old_period
    new_start, new_end = new_period
    assert old_start <= new_start
    assert new_end <= old_end
    beg_offset = count_points_in_period((old_start, new_start))
    end_offset = count_points_in_period((new_end, old_end))

    return mat_q[beg_offset:len(mat_q) if end_offset == 0 else -end_offset]


def iter_week_days(mat, start_date: datetime):
    assert len(mat) % POINTS_PER_DAY == 0
    days = len(mat) // POINTS_PER_DAY
    for d in range(days):
        yield (start_date.weekday() + d) % 7, mat[d * POINTS_PER_DAY:(d + 1) * POINTS_PER_DAY]
    yield (start_date.weekday() + d + 1) % 7, None


def split_weekdays_and_weekends(mat, start_date: datetime):
    wd_data = []
    we_data = []
    weekdays = []
    weekends = []

    if start_date.weekday() % 7 < 5:
        weekdays.append(start_date)
    else:
        weekends.append(start_date)

    current_period = []

    for i, (d, m) in enumerate(iter_week_days(mat, start_date)):
        if current_period:
            if d == 0:
                if m is not None:
                    weekdays.append(start_date + timedelta(days=i))
                we_data.append(torch.column_stack(current_period))
                current_period.clear()
            elif d == 5:
                if m is not None:
                    weekends.append(start_date + timedelta(days=i))
                wd_data.append(torch.column_stack(current_period))
                current_period.clear()
        if m is not None:
            current_period.append(m.T)

    return (wd_data, weekdays), (we_data, weekends)
