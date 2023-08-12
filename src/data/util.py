from globals import *

import torch

from datetime import datetime
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


def extract_week_day(mat, start_date: datetime, weekday: int):
    assert len(mat) % POINTS_PER_DAY == 0
    return mat[
        (torch.arange(len(mat) // POINTS_PER_DAY) + start_date.weekday())
        .repeat_interleave(POINTS_PER_DAY) % 7 == weekday
    ]


def split_weekdays_and_weekends(mat, start_date: datetime):
    weekdays = torch.row_stack([extract_week_day(mat, start_date, d)
                           for d in range(5)]).flatten(0, 0)
    weekends = torch.row_stack([extract_week_day(mat, start_date, d)
                           for d in range(5, 7)]).flatten(0, 0)

    return weekdays, weekends

