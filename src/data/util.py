import torch

from datetime import datetime, timedelta

POINTS_PER_HOUR = 60 // 15


def count_points_between(start_date: datetime, end_date: datetime):
    return int((end_date - start_date).total_seconds() // 3600 * POINTS_PER_HOUR)


def crop_q_between(mat_q, old_start, old_end, new_start, new_end):
    assert old_start <= new_start < new_end <= old_end
    beg_offset = count_points_between(old_start, new_start)
    end_offset = count_points_between(new_end, old_end)

    return mat_q[beg_offset:-end_offset]


def split_weekdays_and_weekends(mat_c, start_date: datetime, end_date: datetime):
    assert start_date.hour == 0 and start_date.minute == 0 and start_date.second == 0 and start_date.microsecond == 0
    assert end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0

    weekend_mask = torch.tensor([(start_date + timedelta(days=d)).weekday() in [5, 6] for d in range((end_date - start_date).days)])\
        .repeat_interleave(POINTS_PER_HOUR * 24)
    weekdays = mat_c[~weekend_mask]
    weekends = mat_c[weekend_mask]

    return weekdays, weekends
