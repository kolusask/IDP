from datetime import datetime

def count_points_between(start_date: datetime, end_date: datetime):
    return int((end_date - start_date).total_seconds() // (60 * 15))

def crop_q_between(mat_q, old_start, old_end, new_start, new_end):
    assert old_start <= new_start < new_end <= old_end
    beg_offset = count_points_between(old_start, new_start)
    end_offset = count_points_between(new_end, old_end)

    return mat_q[beg_offset:-end_offset]
