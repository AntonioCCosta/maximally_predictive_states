import numpy as np
import numpy.ma as ma
from numba import jit,prange
import numpy as np

def segment_maskedArray(tseries,min_size=50):
    '''
    Segments  time series in case it has missing data
    '''
    if ~np.ma.isMaskedArray(tseries):
        tseries = ma.masked_invalid(tseries)
    if len(tseries.shape)>1:
        mask = ~np.any(tseries.mask,axis=1)
    else:
        mask = ~tseries.mask
    segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
    segs_ = []
    for t0,tf in segments:
        if tf-t0>min_size:
            segs_.append([t0,tf])
    segments = np.vstack(segs_)
    return segments

@jit(nopython=True)
def tm_seg(X,K):
    '''
    Build a trajectory matrix
    X: N x dim data
    K: the number of delays
    out: (N-K)x(dim*K) dimensional
    '''
    tm=np.zeros(((len(X)-K-1),X.shape[1]*(K+1)))
    for t in range(len(X)-K-1):
        x = X[t:t+K+1,:][::-1]
        x_flat = x.flatten()
        tm[t] = x_flat
    return tm

def trajectory_matrix(X,K):
    min_seg=K+1
    segments = segment_maskedArray(X,min_seg)
    traj_matrix = ma.zeros((len(X),X.shape[1]*(K+1)))
    for t0,tf in segments:
        traj_matrix[t0+int(np.floor(K/2)):tf-int(np.ceil(K/2)+1)] = ma.masked_invalid(tm_seg(ma.filled(X[t0:tf],np.nan),K))
    traj_matrix[traj_matrix==0]=ma.masked
    return traj_matrix

def whitening(X):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_mean = np.mean(X,axis=0)
    X_centered = X - X_mean
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    U, Lambda, V = np.linalg.svd(Sigma)
    W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
    return U,X_mean,W,np.dot(X_centered, W.T)

def embed_tseries(X,K,m=0,whiten=False,return_modes=False):
    '''
    Get delay embedding with K delays and m SVD dimensions
    K: number of delays
    m: number of SVD modes (default = 0 return the full trajectory matrix)
    out: if m>0 modes and projections, otherwise the full trajectory matrix
    '''
    traj_matrix = trajectory_matrix(X,K)
    if m>0:
        if whiten:
            if ma.count_masked(traj_matrix)>0:
                modes,mean_X,transformation,whitened_X = whitening(ma.compress_rows(traj_matrix))
                whitened = ma.zeros(traj_matrix.shape)
                whitened[~np.any(traj_matrix.mask,axis=1)] = whitened_X
                whitened[np.any(traj_matrix.mask,axis=1)] = ma.masked
                return modes,mean_X,transformation,whitened[:,:m]
            else:
                modes,mean_X,transformation,whitened_X = whitening(traj_matrix)
                return modes,mean_X,transformation,whitened_X[:,:m]
        else:
            if ma.count_masked(traj_matrix)>0:
                u,s,v=np.linalg.svd(ma.compress_rows(traj_matrix),full_matrices=False)
                phspace = ma.zeros((traj_matrix.shape[0],m))
                phspace[~np.any(traj_matrix.mask,axis=1),:]=u[:,:m]
                phspace[phspace==0]=ma.masked
            else:
                u,s,v=np.linalg.svd(traj_matrix,full_matrices=False)
                phspace = u[:,:m]
        if return_modes:
            return v,s,phspace
        else:
            return phspace
    else:
        phspace = traj_matrix
        return phspace


def embed_tseries_pca(X,K,return_modes=True,part_ratio=False):
    '''
    Get delay embedding with K delays and m PCA dimensions
    X: numpy masked array
    K: number of delays
    m: number of SVD modes (default = 0 return the full trajectory matrix)
    out: if m>0 modes and projections, otherwise the full trajectory matrix
    '''
    traj_matrix = trajectory_matrix(X,K)
    X_ = ma.compress_rows(traj_matrix)
    cov = np.cov(X_.T)
    eigvals,eigvecs = np.linalg.eig(cov)
    if part_ratio:
        dim = 2*int(np.ceil(np.sum(eigvals)**2/np.sum(eigvals**2)))
    else:
        var_exp = np.cumsum(eigvals)/np.sum(eigvals)
        dim = np.arange(len(var_exp))[var_exp>0.99][0]
    phspace = ma.zeros((traj_matrix.shape[0],dim))
    phspace[~np.any(traj_matrix.mask,axis=1),:]= X_.dot(eigvecs[:,:dim])
    phspace[phspace==0]=ma.masked
    if return_modes:
        return eigvals,eigvecs,phspace
    else:
        return phspace


from numpy import *
from numpy.linalg import *


def lyapunov_exponent_map(f_df, single_initial_condition=None, multiple_initial_conditions=None,
                      tol=0.01, max_it=50000, min_it_percentage=0.1):
    """Numerical approximation of all Lyapunov Exponents of a map.
    Ref: Alligood, Kathleen T., Tim D. Sauer, and James A. Yorke. Chaos. Springer New York, 1996.
    Parameters
    ----------
    f_df : callable
        The map to be considered for computation of Lyapunov Exponents
        f_df (x, w) -> (f(x), df(x, w)), where
        f(x) is the map evaluated at the point x and
        df(x, w) is the differential of this map evaluated at x in the direction of w.
    single_initial_condition : ndarray of shape (n)
        Single initial condition to computed the Lyapunov Exponent.
    multiple_initial_conditions : ndarray of shape (m,n)
        m initial conditions to computed the average Lyapunov Exponent.
    tol : float
        Tolerance to stop the approximation.
    max_it : int
        Max numbers of iterations.
    min_it_percentage : float, optional
        Min number of iterations as a percentage of the max_it.
    Returns
    -------
    : ndarray of shape (n)
        The Lyapunov exponents computed associated to a single initial condition or
        the average value considering several initial conditions
    """

    if multiple_initial_conditions is not None:
        (m, n) = shape(multiple_initial_conditions)
        ls = zeros((m, n))
        for i in range(m):
            ls[i] = lyapunov_exponent_map(f_df=f_df, single_initial_condition=multiple_initial_conditions[i],
                                      tol=tol, max_it=max_it, min_it_percentage=min_it_percentage)
        return apply_along_axis(lambda v: mean(v), 0, ls)

    elif single_initial_condition is None:
        raise Exception('Either single_initial_condition or multiple_initial_conditions must be provided.')

    n = len(single_initial_condition)
    x = single_initial_condition
    w = eye(n)
    h = zeros(n)
    trans_it = int(max_it * min_it_percentage)
    l = -1

    for i in range(0, max_it):
        x_next, w_next = f_df(x, w)
        w_next = orthogonalize_columns(w_next)

        h_next = h + log_of_the_norm_of_the_columns(w_next)
        l_next = h_next / (i + 1)

        if i > trans_it and norm(l_next - l) < tol:
            return sort(l_next)

        h = h_next
        x = x_next
        w = normalize_columns(w_next)
        l = l_next

    raise Exception('Lyapunov Exponents computation did no convergence' +
                    ' at ' + str(single_initial_condition) +
                    ' with tol=' + str(tol) +
                    ' max_it=' + str(max_it) +
                    ' min_it_percentage=' + str(min_it_percentage))


def orthogonalize_columns(a):
    q, r = qr(a)
    return q @ diag(r.diagonal())


def normalize_columns(a):
    return apply_along_axis(lambda v: v / norm(v), 0, a)


def log_of_the_norm_of_the_columns(a):
    return apply_along_axis(lambda v: log(norm(v)), 0, a)


class HenonMap:
    """ Class to hold the parameters of the Henon Map and evaluate it along its directional derivative.
        The default parameters are chosen as the canonical ones in the initialization.
        It instantiates a callable object f_df, in such a way that f_df(xy, w)
        returns two values f(xy) and df(xy, w), where
        f(xy) is the Henon map evaluated at the point xy and
        df(xy, w) is the differential of the Henon evaluated at xy in the direction of w.
    """
    def __init__(_, a=1.4, b=0.3):
        _.a, _.b = a, b

    def f(_, xy):
        x, y = xy
        return array([_.a - x ** 2 + _.b * y, x])

    def df(_, xy, w):
        x, y = xy
        j = array([[-2 * x, _.b],
                   [1, 0]])
        return j @ w

    def __call__(_, xy, w):
        return _.f(xy), _.df(xy, w)


# HENON_MAP = HenonMap()
# x_points, y_points = mgrid[-1:1:0.25, -1:1:0.25]
# HENON_MAP_INITIAL_CONDITIONS = column_stack((x_points.ravel(), y_points.ravel()))
# l = lyapunov_exponent(HENON_MAP, multiple_initial_conditions=HENON_MAP_INITIAL_CONDITIONS)
