import torch


def build_correlation_matrix(mat_q):
    return torch.corrcoef(mat_q.T).nan_to_num(0)


def split_sections_into_groups(mat_r, alpha):
    mat_r = torch.abs(mat_r)
    ungrouped = torch.ones(len(mat_r), dtype=bool)
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
    representatives = torch.stack([g[0] for g, _, _ in groups])
    mat_c = mat_q[:, representatives]
    assert mat_c.shape == (mat_q.shape[0], len(groups))

    return mat_c


def get_compressed_matrix(mat_c, mat_q):
    return torch.linalg.pinv(mat_c) @ mat_q
