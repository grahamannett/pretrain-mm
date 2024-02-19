import torch


def symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """

    u_, s_, vh_ = torch.linalg.svd(mat)

    # sqrt is unstable around 0, just use 0 in such case
    si = torch.where(s_ < eps, s_, torch.sqrt(s_))

    # use diag_embed when mat is 3d as in batch mode
    return (u_ @ torch.diag_embed(si)) @ vh_.transpose(-2, -1)


def trace_sqrt_product(sigma, sigma_v):
    """PyTorch version of trace_sqrt_product."""
    # Compute the square root of sigma
    sqrt_sigma = symmetric_matrix_square_root(sigma)

    # Compute sqrt(A sigma_v A) = sqrt_sigma * sigma_v * sqrt_sigma
    sqrt_a_sigmav_a = sqrt_sigma @ (sigma_v @ sqrt_sigma)

    # Compute the square root of sqrt(A sigma_v A)
    sqrt_root = symmetric_matrix_square_root(sqrt_a_sigmav_a)

    # Return the trace of the square root
    return torch.diagonal(sqrt_root, dim1=-2, dim2=-1).sum(dim=-1)


def sample_covariance(x: torch.Tensor, y: torch.Tensor, invert: bool = False, f_dim: int = -1) -> torch.Tensor:

    div_val = x.shape[f_dim] # somewhat of a normalization 

    cov = (x.transpose(-2, -1) @ y) / div_val

    if invert:
        cov = torch.linalg.pinv(cov)

    return cov


@torch.no_grad
def fid(x: torch.Tensor, y: torch.Tensor, estimator: callable = sample_covariance, mean_dim=-1, f_dim=-2):
    """notes:

    - mean_dim is -1 and f_dim is -2 which is different than below (which I verified sort of official implementation values) but,
    the values here made no sense

    you want the values to be like:
        `batch_size x feature(e.g. vocab/hidden dim) x seq_len`
    for both x and y
    """

    m_x = torch.mean(x, dim=mean_dim)
    m_y = torch.mean(y, dim=mean_dim)
    m_dist = torch.norm(x.mean(dim=mean_dim) - y.mean(dim=mean_dim), dim=-1) ** 2

    if (x.ndim == 3) and (m_x.ndim == 2):
        m_x, m_y = map(lambda t: t.unsqueeze(mean_dim), (m_x, m_y))

    c_x = estimator(x - m_x, y - m_y, f_dim=f_dim)
    c_y = estimator(y - m_y, x - m_x, f_dim=f_dim)

    # should be idential to cfid but just without conditional
    c_dist = trace_sqrt_product(c_x, c_y)
    c_dist = torch.diagonal(c_x + c_y, dim1=-2, dim2=-1).sum(dim=-1) - 2 * c_dist

    return m_dist + c_dist


# @torch.fx
@torch.no_grad
def cfid(
    y_true: torch.Tensor,
    y_predict: torch.Tensor,
    x_true: torch.Tensor,
    estimator: callable = sample_covariance,
    mean_dim: int = -2,
    f_dim: int = -1,
    **kwargs,
):
    """_summary_

    Args:
        y_true (torch.Tensor): _description_
        y_pred (torch.Tensor): _description_
        x_true (torch.Tensor): _description_
        estimator (sample_covariance): _description_
    """

    # take mean along FEATURE dim (e.g. vocab as we pass in transposed to follow official)
    # so cov is over seq. I think you have to do it over seq since if it is cov wrt tokens then
    # the cov matrix ends up being way too big to actually compute due to vocab size being (segfault)

    m_y_true = torch.mean(y_true, dim=mean_dim)
    m_y_predict = torch.mean(y_predict, dim=mean_dim)
    m_x_true = torch.mean(x_true, dim=mean_dim)

    if (x_true.ndim == 3) and (m_x_true.ndim == 2):
        # m_x_true, m_y_true, m_y_predict = apply(m_x_true, m_y_true, m_y_predict, fn=lambda t: t.unsqueeze(1))
        m_x_true, m_y_true, m_y_predict = map(lambda t: t.unsqueeze(mean_dim), (m_x_true, m_y_true, m_y_predict))

    c_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true, f_dim=f_dim)
    c_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true, f_dim=f_dim)

    c_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true, f_dim=f_dim)
    c_x_true_y_predict = estimator(x_true - m_x_true, y_predict - m_y_predict, f_dim=f_dim)

    c_y_predict_y_predict = estimator(y_predict - m_y_predict, y_predict - m_y_predict, f_dim=f_dim)
    c_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true, f_dim=f_dim)
    inv_c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True, f_dim=f_dim)

    # conditional mean and covariance estimations
    # THESE ARENT USED SO COMMENTED OUT?
    # cov_est = x_true - m_x_true
    # A = inv_c_x_true_x_true @ cov_est.transpose(-2, -1)
    # m_y_true_given_x_true = m_y_true + c_y_true_x_true @ A
    # m_y_predict_given_x_true = m_y_predict + c_y_predict_x_true @ A

    c_y_true_given_x_true = c_y_true_y_true - (c_y_true_x_true @ (inv_c_x_true_x_true @ c_x_true_y_true))
    c_y_predict_given_x_true = c_y_predict_y_predict - (c_y_predict_x_true @ (inv_c_x_true_x_true @ c_x_true_y_predict))

    c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
    c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

    m_dist = m_y_true - m_y_predict
    m_dist = torch.norm(m_dist, dim=-1) ** 2  # same as torch.einsum("...k,...k->...", m_dist, m_dist)

    c_dist1 = c_y_true_x_true_minus_c_y_predict_x_true @ inv_c_x_true_x_true
    c_dist1 = c_dist1 @ c_x_true_y_true_minus_c_x_true_y_predict
    c_dist1 = c_dist1.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    c_dist2 = trace_sqrt_product(c_y_predict_given_x_true, c_y_true_given_x_true)
    c_dist2 = (
        torch.diagonal(c_y_true_given_x_true + c_y_predict_given_x_true, dim1=-2, dim2=-1).sum(dim=-1) - 2 * c_dist2
    )

    return m_dist.transpose(0, -1) + c_dist1 + c_dist2
