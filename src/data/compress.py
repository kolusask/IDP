import torch


def build_correlation_matrix(mat_q, remove_empty=False):
    usable = torch.ones(mat_q.shape[1], dtype=bool)
    if remove_empty:
        usable[mat_q.sum(dim=0) == 0] = False

    return torch.corrcoef(mat_q.T[usable]).nan_to_num(0), usable


    if remove_empty:
        nonempty = mat.diag() > 0
        mat = mat[nonempty, :][:, nonempty]
        nonempty = torch.where(nonempty)[0]
        return mat, nonempty
    else:
        return mat

def split_sections_into_groups(mat_r, alpha):
    mat_r = torch.abs(mat_r)
    ungrouped = torch.ones(len(mat_r), dtype=bool, device=mat_r.device)
    groups = []

    def _new_group():
        nonlocal ungrouped, group_queue, current_group
        s = ungrouped.to(dtype=int).argmax().item()
        ungrouped[s] = False
        group_queue = [s,]
        current_group = [s,]
    
    _new_group()

    while ungrouped.sum() > 0:
        if not group_queue:
            groups.append(current_group)
            _new_group()
        
        section = group_queue.pop()
        correlated = torch.bitwise_and(mat_r[section] > alpha, ungrouped)
        correlated[section] = False
        correlated = torch.where(correlated)[0].tolist()
        current_group += correlated
        group_queue += correlated
        ungrouped[correlated] = False
    
    groups.append(current_group)
    
    return groups


def get_compression_matrix(mat_q, groups):
    representatives = torch.tensor([g[0] for g in groups])
    mat_c = mat_q[:, representatives]
    assert mat_c.shape == (mat_q.shape[0], len(groups))

    return mat_c, representatives


def get_compressed_matrix(mat_c, mat_q):
    return torch.linalg.pinv(mat_c) @ mat_q
