import torch

def build_correlation_matrix(mat_q):
    d, p = mat_q.shape
    mat_q_normalized = mat_q - mat_q.mean(dim=0)
    mat_r = torch.zeros(p, p)
    for i in range(p):
        for j in range(i, p):
            i_col = mat_q_normalized[:, i]
            j_col = mat_q_normalized[:, j]
            i_norm = max(torch.norm(i_col), 1e-12)
            j_norm = max(torch.norm(j_col), 1e-12)
            if i_norm == 0 or j_norm == 0:
                if i_norm == j_norm:
                    mat_r[i][j] = 1
                else:
                    mat_r[i][j] = 0
            else:
                mat_r[i][j] = (i_col @ j_col) / i_norm / j_norm
            mat_r[j][i] = mat_r[i][j]
    
    return mat_r / mat_r.max()


def split_sections_into_groups(mat_r, alpha):
    n_sections, _ = mat_r.shape
    n_ungrouped = n_sections
    groups = []

    mat_r_copy = mat_r - torch.diag(mat_r.diag())
    while n_ungrouped > 0:
        new_group_idx = torch.nonzero(mat_r_copy > alpha)
        if len(new_group_idx) > 0:
            corr = mat_r[new_group_idx[:, 0], new_group_idx[:, 1]]
            new_group_idx = new_group_idx[:, 0].unique()

            n_ungrouped -= len(new_group_idx)
            mat_r_copy[new_group_idx, :] = 0
            mat_r_copy[:, new_group_idx] = 0
            groups.append((new_group_idx, corr.min(), corr.max()))
            if mat_r_copy.max() == 0:
                break
            else:
                mat_r_copy /= mat_r_copy.max()
    
    return groups, n_ungrouped


def get_compression_matrix(mat_q, groups):
    representatives = torch.stack([g[0] for g, _, _ in groups])
    mat_c = mat_q[:, representatives]
    assert mat_c.shape == (mat_q.shape[0], len(groups))

    return mat_c


def get_compressed_matrix(mat_c, mat_q):
    return torch.linalg.pinv(mat_c) @ mat_q
