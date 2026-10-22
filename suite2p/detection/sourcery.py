"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import trange
import numpy as np
import scipy.sparse
from scipy.ndimage import filters, gaussian_filter
import logging
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

from .utils import circleMask

# number of image rows smoothed at once on the GPU in getVmap
# (128 rows x 608 px x 2472 components x 4 bytes ~ 0.8 GB per block)
GPU_VMAP_ROWS = 128


def masks_to_sparse(ypix, xpix, lam, Ly, Lx):
    """
    Build the ROI mask matrix as a sparse matrix of shape (Ly * Lx, nrois).

    Equivalent to the dense ``L = np.zeros((Ly, Lx, nrois)); L[ypix[n], xpix[n], n] = lam[n]``
    reshaped to (Ly * Lx, nrois), but without allocating Ly * Lx * nrois floats
    (~19 GB for 4000 ROIs on a 2000 x 600 field of view).
    """
    rows, cols, vals = [], [], []
    for n in range(len(ypix)):
        idx = np.asarray(ypix[n], dtype=np.int64) * Lx + np.asarray(xpix[n], dtype=np.int64)
        la = np.asarray(lam[n], dtype=np.float32)
        # like fancy assignment, keep the last value for repeated pixels
        _, last = np.unique(idx[::-1], return_index=True)
        keep = len(idx) - 1 - last
        rows.append(idx[keep])
        cols.append(np.full(len(keep), n, dtype=np.int64))
        vals.append(la[keep])
    if rows:
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    else:
        rows, cols, vals = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    return scipy.sparse.csc_matrix((vals, (rows, cols)), shape=(Ly * Lx, len(ypix)),
                                   dtype=np.float32)


def _gaussian_kernel1d(sigma, truncate=4.0):
    """Same kernel as scipy.ndimage.gaussian_filter1d (order 0)."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / float(sigma) ** 2 * x ** 2)
    return phi / phi.sum(), radius


def _use_gpu(device):
    return (torch is not None and device is not None and torch.device(device).type == "cuda"
            and torch.cuda.is_available())


def getSVDdata(mov: np.ndarray, diameter):
    """
    Compute SVD spatial components from temporally binned movie data.

    Smooths each frame with a 2D Gaussian filter, computes the temporal covariance
    matrix, and returns the top SVD spatial components reshaped as images.

    Parameters
    ----------
    mov : numpy.ndarray
        Temporally binned movie of shape (nbins, Ly, Lx).
    diameter : float or list of float
        Cell diameter used to set the Gaussian smoothing sigma (sigma = diameter / 10).

    Returns
    -------
    U : numpy.ndarray
        Spatial SVD components of shape (Ly, Lx, nsvd).
    u : numpy.ndarray
        Temporal SVD components of shape (nbins, nsvd).
    """
    nbins, Lyc, Lxc = np.shape(mov)

    sig = diameter / 10.  # PICK UP
    _smooth_frames(mov, sig)  # mov[j] = gaussian_filter(mov[j], sig) for every frame

    # compute noise variance across frames
    mov = np.reshape(mov, (-1, Lyc * Lxc))
    # compute covariance of binned frames
    cov = mov @ mov.transpose() / mov.shape[1]
    cov = cov.astype("float32")

    nsvd_for_roi = min(nbins, int(cov.shape[0] / 2))
    u, s, v = np.linalg.svd(cov)

    u = u[:, :nsvd_for_roi]
    U = u.transpose() @ mov
    U = np.reshape(U, (-1, Lyc, Lxc))
    U = np.transpose(U, (1, 2, 0)).copy()
    return U, u


def getSVDproj(mov: np.ndarray, u, diameter, smooth_masks=False):
    """
    Project temporally binned movie data onto existing SVD temporal components.

    Optionally smooths frames before projection. Used during refinement to
    recompute spatial components from updated movie data.

    Parameters
    ----------
    mov : numpy.ndarray
        Temporally binned movie of shape (nbins, Ly, Lx).
    u : numpy.ndarray
        Temporal SVD components of shape (nbins, nsvd).
    diameter : float or list of float
        Cell diameter used to set the Gaussian smoothing sigma.
    smooth_masks : bool, optional (default False)
        If True, smooth each frame before projection.

    Returns
    -------
    U : numpy.ndarray
        Spatial SVD projections of shape (Ly, Lx, nsvd).
    """
    nbins, Lyc, Lxc = np.shape(mov)
    if smooth_masks:
        sig = np.maximum([.5, .5], diameter / 20.)
        for j in range(nbins):
            mov[j, :, :] = gaussian_filter(mov[j, :, :], sig)
    
    mov = np.reshape(mov, (-1, Lyc * Lxc))

    U = u.transpose() @ mov
    U = U.transpose().copy().reshape((Lyc, Lxc, -1))
    
    return U


def getStU(diameter, U):
    """
    Compute neuropil basis functions and their covariances with the SVD spatial components.

    Parameters
    ----------
    diameter : float or list of float
        Cell diameter used for neuropil basis construction.
    U : numpy.ndarray
        Spatial SVD components of shape (Ly, Lx, nsvd).

    Returns
    -------
    S : numpy.ndarray
        Neuropil basis functions of shape (Ly, Lx, nbasis).
    StU : numpy.ndarray
        Cross-covariance of neuropil basis with SVD components, shape (nbasis, nsvd).
    StS : numpy.ndarray
        Auto-covariance of neuropil basis, shape (nbasis, nbasis).
    """
    Lyc, Lxc, nbins = np.shape(U)
    S = create_neuropil_basis(diameter, Lyc, Lxc)
    # compute covariance of neuropil masks with spatial masks
    StU = S.reshape((Lyc * Lxc, -1)).transpose() @ U.reshape((Lyc * Lxc, -1))
    StS = S.reshape((Lyc * Lxc, -1)).transpose() @ S.reshape((Lyc * Lxc, -1))
    #U = np.reshape(U, (-1,Lyc,Lxc))
    return S, StU, StS


class NeuropilBasis:
    """
    The neuropil basis of create_neuropil_basis without building it.

    Each basis function is an outer product of a function of y and a function of x,
    S[:, ky * nX + kx] = outer(Ky[:, ky], Kx[:, kx]) / ||outer(...)||, so products with
    S can be computed with the small (Ly, nY) and (Lx, nX) factors. For a 1978 x 608
    field of view with diameter 3 there are 221 x 69 = 15249 basis functions, and the
    dense S would be ~73 GB; S @ X costs ~300 times fewer operations this way.
    """

    def __init__(self, diameter, Ly, Lx, rows=64):
        ratio_neuropil = 6.
        tile_factor = 1.
        ntilesY = 1 + 2 * int(
            np.ceil(tile_factor * Ly / (ratio_neuropil * diameter[0] / 2)) / 2)
        ntilesX = 1 + 2 * int(
            np.ceil(tile_factor * Lx / (ratio_neuropil * diameter[1] / 2)) / 2)
        ntilesY = int(np.maximum(2, ntilesY))
        ntilesX = int(np.maximum(2, ntilesX))
        ys = np.arange(0, Ly)
        xs = np.arange(0, Lx)
        Kx = np.ones((Lx, ntilesX), "float32")
        Ky = np.ones((Ly, ntilesY), "float32")
        for k in range(int((ntilesX - 1) / 2)):
            Kx[:, 2 * k + 1] = np.sin(2 * math.pi * (xs + 0.5) * (1 + k) / Lx)
            Kx[:, 2 * k + 2] = np.cos(2 * math.pi * (xs + 0.5) * (1 + k) / Lx)
        for k in range(int((ntilesY - 1) / 2)):
            Ky[:, 2 * k + 1] = np.sin(2 * math.pi * (ys + 0.5) * (1 + k) / Ly)
            Ky[:, 2 * k + 2] = np.cos(2 * math.pi * (ys + 0.5) * (1 + k) / Ly)
        # the norm of an outer product is the product of the norms
        self.Ky = (Ky / np.sqrt((Ky**2).sum(axis=0))).astype("float32")
        self.Kx = (Kx / np.sqrt((Kx**2).sum(axis=0))).astype("float32")
        self.Ly, self.Lx, self.nY, self.nX = Ly, Lx, ntilesY, ntilesX
        self.nbasis = ntilesY * ntilesX
        self.rows = rows

    def dense(self):
        """Dense S of shape (Ly, Lx, nbasis) (for tests on small images)."""
        S = self.Ky[:, None, :, None] * self.Kx[None, :, None, :]
        return S.reshape(self.Ly, self.Lx, self.nbasis)

    def pixel_rows(self, ypix, xpix):
        """S[ypix, xpix, :] of shape (npix, nbasis)."""
        return (self.Ky[ypix][:, :, None] * self.Kx[xpix][:, None, :]).reshape(len(ypix), -1)

    def gram(self):
        """S.T @ S of shape (nbasis, nbasis)."""
        return np.kron(self.Ky.T @ self.Ky, self.Kx.T @ self.Kx).astype("float32")

    def rmatmul(self, X):
        """S.T @ X for X of shape (Ly, Lx, k); returns (nbasis, k)."""
        k = X.shape[-1]
        out = np.zeros((self.nY, self.nX, k), "float32")
        for y0 in range(0, self.Ly, self.rows):
            y1 = min(self.Ly, y0 + self.rows)
            P = np.matmul(self.Kx.T[None], X[y0:y1])  # (h, nX, k)
            out += np.tensordot(self.Ky[y0:y1], P, axes=([0], [0]))
        return out.reshape(self.nbasis, k)

    def add_matmul(self, X, coef, sign=1.):
        """X += sign * (S @ coef) in place, for X of shape (Ly, Lx, k) and coef (nbasis, k)."""
        k = X.shape[-1]
        N = np.ascontiguousarray(coef.reshape(self.nY, self.nX, k), dtype="float32")
        for y0 in range(0, self.Ly, self.rows):
            y1 = min(self.Ly, y0 + self.rows)
            M = np.tensordot(self.Ky[y0:y1], N, axes=([1], [0]))  # (h, nX, k)
            R = np.matmul(self.Kx[None], M)  # (h, Lx, k)
            if sign > 0:
                X[y0:y1] += R
            else:
                X[y0:y1] -= R


def residual_maps(U, Sb, neu, L, codes, out=None):
    """
    Compute U - (S @ neu + L @ codes) reshaped like U, without temporaries of the size of U.

    Writes into `out` if given (and not U itself). L is the sparse (Ly * Lx, ncells) mask
    matrix, so L @ codes is only computed for pixels that belong to an ROI.
    """
    if out is None or out is U or out.shape != U.shape:
        out = np.empty_like(U)
    np.copyto(out, U)
    Sb.add_matmul(out, neu, sign=-1.)
    if L.shape[1] > 0 and L.nnz > 0:
        Lr = L.tocsr()
        rows = np.flatnonzero(np.diff(Lr.indptr))
        flat = out.reshape(-1, out.shape[-1])
        flat[rows] -= np.asarray(Lr[rows] @ codes, dtype="float32")
    return out


def mean_square(X, rows=8):
    """(X**2).mean() computed in blocks of rows, summed in float64 (only used for logging)."""
    total = 0.
    for y0 in range(0, X.shape[0], rows):
        b = X[y0:y0 + rows].astype(np.float64).ravel()
        total += float(np.dot(b, b))
    return total / X.size


def _smooth_frames(mov, sig, nworkers=8):
    """mov[j] = gaussian_filter(mov[j], sig) for every frame, in threads (same result)."""
    def smooth(j):
        mov[j] = gaussian_filter(mov[j], sig)
    if nworkers <= 1:
        for j in range(mov.shape[0]):
            smooth(j)
        return
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        list(pool.map(smooth, range(mov.shape[0])))


def create_neuropil_basis(diameter, Ly, Lx):
    """
    Compute Fourier-based neuropil basis functions for the image.

    Creates a set of 2D Fourier basis functions tiled across the image,
    used to model spatially varying neuropil contamination.

    Parameters
    ----------
    diameter : float or list of float
        Cell diameter used to determine the tiling density.
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.

    Returns
    -------
    S : numpy.ndarray
        Normalized neuropil basis functions of shape (Ly, Lx, nbasis).
    """

    ratio_neuropil = 6.
    tile_factor = 1.
    
    ntilesY = 1 + 2 * int(
        np.ceil(tile_factor * Ly / (ratio_neuropil * diameter[0] / 2)) / 2)
    ntilesX = 1 + 2 * int(
        np.ceil(tile_factor * Lx / (ratio_neuropil * diameter[1] / 2)) / 2)
    ntilesY = np.maximum(2, ntilesY)
    ntilesX = np.maximum(2, ntilesX)
    yc = np.linspace(1, Ly, ntilesY)
    xc = np.linspace(1, Lx, ntilesX)
    ys = np.arange(0, Ly)
    xs = np.arange(0, Lx)

    Kx = np.ones((Lx, ntilesX), "float32")
    Ky = np.ones((Ly, ntilesY), "float32")
    if 1:
        # basis functions are fourier modes
        for k in range(int((ntilesX - 1) / 2)):
            Kx[:, 2 * k + 1] = np.sin(2 * math.pi * (xs + 0.5) * (1 + k) / Lx)
            Kx[:, 2 * k + 2] = np.cos(2 * math.pi * (xs + 0.5) * (1 + k) / Lx)
        for k in range(int((ntilesY - 1) / 2)):
            Ky[:, 2 * k + 1] = np.sin(2 * math.pi * (ys + 0.5) * (1 + k) / Ly)
            Ky[:, 2 * k + 2] = np.cos(2 * math.pi * (ys + 0.5) * (1 + k) / Ly)
    else:
        for k in range(ntilesX):
            Kx[:, k] = np.cos(math.pi * (xs + 0.5) * k / Lx)
        for k in range(ntilesY):
            Ky[:, k] = np.cos(math.pi * (ys + 0.5) * k / Ly)

    S = np.zeros((ntilesY, ntilesX, Ly, Lx), np.float32)
    for kx in range(ntilesX):
        for ky in range(ntilesY):
            S[ky, kx, :, :] = np.outer(Ky[:, ky], Kx[:, kx])

    S = np.reshape(S, (ntilesY * ntilesX, Ly * Lx))
    S = S / np.reshape(np.sum(S**2, axis=-1)**0.5, (-1, 1))
    S = np.transpose(S, (1, 0)).copy()
    S = np.reshape(S, (Ly, Lx, -1))
    return S


def morphOpen(V, footprint):
    """
    Compute the morphological opening of a correlation map.

    Applies a minimum filter followed by a maximum filter (negated minimum
    of negation) using the given footprint to remove small bright features.

    Parameters
    ----------
    V : numpy.ndarray
        2D correlation map of shape (Ly, Lx).
    footprint : numpy.ndarray
        2D boolean footprint for the morphological operation.

    Returns
    -------
    vrem : numpy.ndarray
        Morphologically opened image of shape (Ly, Lx).
    """
    vrem = filters.minimum_filter(V, footprint=footprint)
    vrem = -filters.minimum_filter(-vrem, footprint=footprint)
    return vrem


def localMax(V, footprint, thres):
    """
    Find local maxima of a correlation map above a threshold.

    Uses a maximum filter with the given footprint to identify pixels that
    are local maxima and exceed the threshold.

    Parameters
    ----------
    V : numpy.ndarray
        2D correlation map of shape (Ly, Lx).
    footprint : numpy.ndarray
        2D boolean footprint for the maximum filter (usually circular).
    thres : float
        Minimum value for a local maximum to be included.

    Returns
    -------
    i : numpy.ndarray
        Y-indices of local maxima, dtype int32.
    j : numpy.ndarray
        X-indices of local maxima, dtype int32.
    """
    maxV = filters.maximum_filter(V, footprint=footprint, mode="reflect")
    imax = V > np.maximum(thres, maxV - 1e-10)
    i, j = imax.nonzero()
    i = i.astype(np.int32)
    j = j.astype(np.int32)
    return i, j




def getVmap_gpu(Ucell, sig, device, rows=GPU_VMAP_ROWS, keep_on_gpu=True):
    """
    GPU version of getVmap: same smoothing (Gaussian, truncate 4.0, mode "wrap")
    as scipy's gaussian_filter(Ucell, [sig[0], sig[1], 0.], mode="wrap").

    Ucell is sent to the GPU in blocks of rows (with wrapped halo rows), which are
    contiguous in memory, so the (Ly, Lx, nsvd) array does not have to fit on the
    GPU and no transposed copies are needed.

    If keep_on_gpu and there is enough GPU memory, the smoothed components `us` are
    returned as a torch tensor on the GPU (only a few hundred pixels of it are read
    afterwards), which avoids copying ~12 GB back to the CPU. Otherwise `us` is a
    numpy array.
    """
    Ly, Lx, nsvd = Ucell.shape
    device = torch.device(device)
    if keep_on_gpu:
        rows = min(rows, 64)  # smaller blocks, since us itself takes most of the GPU memory
        need = Ucell.nbytes + 4 * (rows + 16) * Lx * nsvd * 4
        free, _ = torch.cuda.mem_get_info(device)
        keep_on_gpu = free > 1.05 * need
    if keep_on_gpu:
        try:
            return _getVmap_gpu(Ucell, sig, device, rows, keep_on_gpu=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.info("getVmap: not enough GPU memory to keep smoothed maps on the GPU")
    return _getVmap_gpu(Ucell, sig, device, rows, keep_on_gpu=False)


def _getVmap_gpu(Ucell, sig, device, rows, keep_on_gpu):
    Ly, Lx, nsvd = Ucell.shape
    wy, ry = _gaussian_kernel1d(sig[0])
    wx, rx = _gaussian_kernel1d(sig[1])
    wy = [float(w) for w in wy]
    wx = [float(w) for w in wx]
    if keep_on_gpu:
        us = torch.empty((Ly, Lx, nsvd), dtype=torch.float32, device=device)
    else:
        us = np.empty_like(Ucell)
    mean_us2 = np.empty((Ly, Lx), np.float32)
    mean_U2 = np.empty((Ly, Lx), np.float32)
    for y0 in range(0, Ly, rows):
        y1 = min(Ly, y0 + rows)
        h = y1 - y0
        if y0 - ry >= 0 and y1 + ry <= Ly:
            block = Ucell[y0 - ry:y1 + ry]
        else:
            block = np.take(Ucell, np.arange(y0 - ry, y1 + ry) % Ly, axis=0)
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)  # (h + 2 ry, Lx, nsvd)
        mean_U2[y0:y1] = (x[ry:ry + h] ** 2).mean(dim=-1).cpu().numpy()
        # scipy filters axis 0 (y) first, then axis 1 (x)
        z = wy[0] * x[0:h]
        for k in range(1, 2 * ry + 1):
            z += wy[k] * x[k:k + h]
        del x
        if rx > 0:
            zp = torch.cat((z[:, -rx:], z, z[:, :rx]), dim=1)
            z = wx[0] * zp[:, 0:Lx]
            for k in range(1, 2 * rx + 1):
                z += wx[k] * zp[:, k:k + Lx]
            del zp
        mean_us2[y0:y1] = (z ** 2).mean(dim=-1).cpu().numpy()
        if keep_on_gpu:
            us[y0:y1] = z
        else:
            us[y0:y1] = z.cpu().numpy()
        del z
    log_variances = mean_us2 / gaussian_filter(mean_U2, sig, mode="wrap")
    return log_variances.astype("float64"), us


def us_at(us, i, j):
    """us[i, j, :] as a numpy array, for us from getVmap (numpy array or GPU tensor)."""
    if torch is not None and isinstance(us, torch.Tensor):
        return us[i, j].cpu().numpy()
    return us[i, j, :]


def getVmap(Ucell, sig, device=None):
    """
    Compute the variance ratio map from SVD spatial components.

    Smooths the spatial components and computes the ratio of smoothed
    variance to total variance, producing a map that highlights regions
    with locally correlated activity.

    Parameters
    ----------
    Ucell : numpy.ndarray
        Residual SVD spatial components of shape (Ly, Lx, nsvd).
    sig : numpy.ndarray or list of float
        Gaussian smoothing sigma [sy, sx] in pixels.

    Returns
    -------
    log_variances : numpy.ndarray
        Variance ratio map of shape (Ly, Lx), dtype float64.
    us : numpy.ndarray or torch.Tensor
        Smoothed spatial components of shape (Ly, Lx, nsvd). A torch tensor on the
        GPU when computed there and it fits; read single pixels with `us_at`.
    """
    if _use_gpu(device):
        return getVmap_gpu(Ucell, sig, device)
    us = gaussian_filter(Ucell, [sig[0], sig[1], 0.], mode="wrap")
    # compute log variance at each location
    log_variances = (us**2).mean(axis=-1) / gaussian_filter(
        (Ucell**2).mean(axis=-1), sig, mode="wrap")
    return log_variances.astype("float64"), us





def get_connected(Ly, Lx, stat):
    """
    Extract the connected component of an ROI starting from its brightest pixel.

    Grows outward from the pixel with maximum weight, keeping only pixels
    that are contiguous and have non-zero weight.

    Parameters
    ----------
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    stat : dict
        ROI statistics dictionary containing "ypix", "xpix", and "lam".
        Modified in-place.

    Returns
    -------
    stat : dict
        Updated ROI dictionary with connected pixels only.
    """
    ypix, xpix, lam = stat["ypix"], stat["xpix"], stat["lam"]
    i0 = lam.argmax()
    mask = np.zeros((Ly, Lx))
    mask[ypix, xpix] = lam
    ypix, xpix = ypix[i0], xpix[i0]
    nsel = 1
    while 1:
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx)
        ix = mask[ypix, xpix] > 1e-10
        ypix, xpix = ypix[ix], xpix[ix]
        if len(ypix) <= nsel:
            break
        nsel = len(ypix)
    lam = mask[ypix, xpix]
    stat["ypix"], stat["xpix"], stat["lam"] = ypix, xpix, lam
    return stat


def connected_region(stat, connected, Ly, Lx):
    """
    Optionally restrict each ROI to its largest connected component.

    Parameters
    ----------
    stat : list of dict
        List of ROI statistics dictionaries, each containing "ypix", "xpix",
        and "lam".
    connected : bool
        If True, apply connected component extraction to each ROI.
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.

    Returns
    -------
    stat : list of dict
        Updated list of ROI dictionaries.
    """
    if connected:
        for j in range(len(stat)):
            stat[j] = get_connected(Ly, Lx, stat[j])
    return stat


def extendROI(ypix, xpix, Ly, Lx, niter=1):
    """
    Expand an ROI by one pixel in each cardinal direction.

    Adds the 4-connected neighbors of the current pixel set and removes
    any that fall outside the image boundaries. Repeated for `niter`
    iterations.

    Parameters
    ----------
    ypix : numpy.ndarray
        Y-coordinates of the current ROI pixels.
    xpix : numpy.ndarray
        X-coordinates of the current ROI pixels.
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    niter : int, optional (default 1)
        Number of expansion iterations.

    Returns
    -------
    ypix : numpy.ndarray
        Expanded Y-coordinates.
    xpix : numpy.ndarray
        Expanded X-coordinates.
    """
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix - 1, ypix + 1), (xpix, xpix + 1, xpix - 1, xpix,
                                                       xpix))
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def iter_extend(ypix, xpix, Ucell, code, refine=-1, change_codes=False):
    """
    Iteratively extend an ROI by projecting onto the SVD code vector.

    Starting from seed pixels, repeatedly expands the region and keeps
    pixels whose projection onto the code vector exceeds a threshold.
    Stops when no new pixels are added or the region exceeds 10000 pixels.

    Parameters
    ----------
    ypix : numpy.ndarray or int
        Initial Y-coordinates of the ROI seed.
    xpix : numpy.ndarray or int
        Initial X-coordinates of the ROI seed.
    Ucell : numpy.ndarray
        Residual SVD spatial components of shape (Ly, Lx, nsvd).
    code : numpy.ndarray
        SVD code vector for this ROI, shape (nsvd,).
    refine : int, optional (default -1)
        Refinement stage indicator. Negative means initial detection.
    change_codes : bool, optional (default False)
        If True and refine < 0, update the code vector during extension.

    Returns
    -------
    ypix : numpy.ndarray
        Y-coordinates of the extended ROI.
    xpix : numpy.ndarray
        X-coordinates of the extended ROI.
    lam : numpy.ndarray
        Normalized pixel weights (projections onto code vector).
    ix : numpy.ndarray
        Boolean mask of pixels kept in the final iteration.
    code : numpy.ndarray
        Updated code vector, shape (nsvd,).
    """
    Lyc, Lxc, nsvd = Ucell.shape
    npix = 0
    iter = 0
    while npix < 10000:
        npix = ypix.size
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)
        usub = Ucell[ypix, xpix, :]
        lam = usub @ np.expand_dims(code, axis=1)
        lam = np.squeeze(lam, axis=1)
        # ix = lam>max(0, np.mean(lam)/3)
        ix = lam > max(0, lam.max() / 5.0)
        if ix.sum() == 0:
            break
        ypix, xpix, lam = ypix[ix], xpix[ix], lam[ix]
        lam = lam / np.sum(lam**2 + 1e-10)**.5
        if refine < 0 and change_codes:
            code = lam @ usub[ix, :]
        if iter == 0:
            sgn = 1.
            #sgn = np.sign(ix.sum()-npix)
        if np.sign(sgn * (ix.sum() - npix)) <= 0:
            break
        else:
            npix = ypix.size
        iter += 1
    return ypix, xpix, lam, ix, code


def sourcery(mov: np.ndarray, sdmov, diameter, threshold_scaling=1.0,
             connected=True, max_iterations=20, smooth_masks=False, device=None):
    """
    Detect ROIs using the Sourcery algorithm (SVD-based iterative detection).

    Computes SVD components of the movie, builds neuropil basis functions 
    and subtracts them, and iteratively detects ROIs by finding local 
    maxima in the variance ratio map and extending them via similarity 
    in SVD projection.

    Parameters
    ----------
    mov : numpy.ndarray
        Temporally binned movie of shape (nbins, Ly, Lx). Divided by
        `sdmov` in-place before processing.
    sdmov : numpy.ndarray
        Standard deviation of the movie, shape (Ly * Lx,) or (Ly, Lx),
        used for normalization.
    diameter : float or list of float
        Expected cell diameter in pixels.
    threshold_scaling : float, optional (default 1.0)
        Scaling factor for the peak detection threshold.
    connected : bool, optional (default True)
        If True, restrict each detected ROI to its largest connected component.
    max_iterations : int, optional (default 20)
        Maximum number of detection and refinement iterations.
    smooth_masks : bool, optional (default False)
        If True, spatially smooth frames before SVD projection during refinement.
    device : torch.device or str, optional (default None)
        If a CUDA device is given and available, the smoothing in getVmap runs
        on the GPU. Otherwise it runs on the CPU with scipy.

    Returns
    -------
    new_settings : dict
        Dictionary with detection metadata including "diameter", "Vcorr",
        and placeholder keys "Vmax", "ihop", "Vsplit", "Vmap",
        "spatscale_pix".
    stat : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix",
        "xpix", and "lam".
    """
    mov /= sdmov

    change_codes = True
    t0 = time.time()

    U, u = getSVDdata(mov=mov, diameter=diameter)  # get SVD components
    Ly, Lx, nsvd = U.shape
    # neuropil basis S (Ly, Lx, nbasis), kept as the product of y and x factors
    Sb = NeuropilBasis(diameter, Ly, Lx)
    StU = Sb.rmatmul(U)
    StS = Sb.gram()
    d0 = diameter
    sig = np.ceil(d0 / 4)  # smoothing constant
    # make array of radii values of size (2*d0+1,2*d0+1)
    rs, dy, dx = circleMask(d0)
    nsvd = U.shape[-1]
    nbasis = Sb.nbasis
    codes = np.zeros((0, nsvd), np.float32)
    LtU = np.zeros((0, nsvd), np.float32)
    LtS = np.zeros((0, nbasis), np.float32)
    # ROI masks as a sparse (Ly * Lx, ncells) matrix (each ROI has only ~tens of pixels)
    L = masks_to_sparse([], [], [], Ly, Lx)
    # regress maps onto basis functions and subtract neuropil contribution
    neu = np.linalg.solve(StS, StU).astype("float32")
    Ucell = residual_maps(U, Sb, neu, L, codes)

    it = 0
    ncells = 0
    refine = -1

    # initialize
    ypix, xpix, lam = [], [], []
    logger.info(f"max_iterations = {max_iterations}; will stop when no more peaks above threshold, or max_iterations reached")
    for it in range(max_iterations):
        if refine < 0:
            V, us = getVmap(Ucell, sig, device=device)
            if it == 0:
                vrem = morphOpen(V, rs <= 1.)
            V = V - vrem  # make V more uniform
            if it == 0:
                V = V.astype("float64")
                # find indices of all maxima in +/- 1 range
                maxV = filters.maximum_filter(V, footprint=np.ones((3, 3)),
                                              mode="reflect")
                imax = V > (maxV - 1e-10)
                peaks = V[imax]
                # use the median of these peaks to decide if ROI is accepted
                thres = threshold_scaling * np.median(peaks[peaks > 1e-4])
                Vcorr = V.copy()
            V = np.minimum(V, Vcorr)

            # add extra ROIs here
            n = ncells
            while n < ncells + 200:
                ind = np.argmax(V)
                i, j = np.unravel_index(ind, V.shape)
                if V[i, j] < thres:
                    break
                yp, xp, la, ix, code = iter_extend(i, j, Ucell, us_at(us, i, j),
                                                   change_codes=change_codes)
                ypix.append(yp)
                xpix.append(xp)
                lam.append(la)
                Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], code)

                yp, xp = extendROI(yp, xp, Ly, Lx, int(np.mean(d0)))
                V[yp, xp] = 0
                n += 1
            del us
            newcells = len(ypix) - ncells
            if it == 0:
                Nfirst = newcells
            LtU = np.append(LtU, np.zeros((newcells, nsvd), "float32"), axis=0)
            LtS = np.append(LtS, np.zeros((newcells, nbasis), "float32"), axis=0)
            for n in range(ncells, len(ypix)):
                LtU[n, :] = lam[n] @ U[ypix[n], xpix[n], :]
                LtS[n, :] = lam[n] @ Sb.pixel_rows(ypix[n], xpix[n])
            ncells += newcells
            L = masks_to_sparse(ypix, xpix, lam, Ly, Lx)

            # regression with neuropil
            LtL = (L.transpose() @ L).toarray().astype("float32")
            cellcode = np.concatenate((LtL, LtS), axis=1)
            neucode = np.concatenate((LtS.transpose(), StS), axis=1)
            codes = np.concatenate((cellcode, neucode), axis=0)
            Ucode = np.concatenate((LtU, StU), axis=0)
            codes = np.linalg.solve(codes + 1e-3 * np.eye((codes.shape[0])),
                                    Ucode).astype("float32")
            neu = codes[ncells:, :]
            codes = codes[:ncells, :]

        # Ucell = U - (S @ neu + L @ codes), reusing the Ucell buffer (unless it is U)
        Ucell = residual_maps(U, Sb, neu, L, codes, out=Ucell)
        # reestimate masks
        n, k = 0, 0
        while n < len(ypix):
            Ucell[ypix[n], xpix[n], :] += np.outer(lam[n], codes[k, :])
            ypix[n], xpix[n], lam[n], ix, codes[n, :] = iter_extend(
                ypix[n], xpix[n], Ucell, codes[k, :], refine, change_codes=change_codes)
            k += 1
            if ix.sum() == 0:
                logger.info("dropped ROI with no pixels")
                del ypix[n], xpix[n], lam[n]
                continue
            Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], codes[n, :])
            n += 1
        codes = codes[:n, :]
        ncells = len(ypix)
        L = masks_to_sparse(ypix, xpix, lam, Ly, Lx)
        LtU = np.zeros((ncells, nsvd), "float32")
        LtS = np.zeros((ncells, nbasis), "float32")
        for n in range(ncells):
            if refine < 0:
                LtU[n, :] = lam[n] @ U[ypix[n], xpix[n], :]
                LtS[n, :] = lam[n] @ Sb.pixel_rows(ypix[n], xpix[n])
        err = mean_square(Ucell)
        t1 = time.time() - t0
        logger.info(f"iter {it},\tROIs: {ncells},\terr: {err:0.4f}, \ttime: {t1:0.2f} sec")

        if refine == 0:
            break
        if refine == 2:
            # good place to get connected regions
            stat = [{
                "ypix": ypix[n],
                "lam": lam[n],
                "xpix": xpix[n]
            } for n in range(ncells)]
            stat = connected_region(stat, connected, Ly, Lx)
            ypix = [stat[n]["ypix"] for n in range(len(stat))]
            xpix = [stat[n]["xpix"] for n in range(len(stat))]
            lam = [stat[n]["lam"] for n in range(len(stat))]
            ncells = len(ypix)
        if refine > 0:
            # Ucell = Ucell + S @ neu
            if Ucell is U:
                Ucell = U.copy()
            Sb.add_matmul(Ucell, neu, sign=1.)
        if refine < 0 and (newcells < Nfirst / 10 or it == max_iterations - 1):
            refine = 3
            U = getSVDproj(mov, u, diameter, smooth_masks)
            Ucell = U
        if refine >= 0:
            StU = Sb.rmatmul(Ucell)
            neu = np.linalg.solve(StS, StU).astype("float32")
        refine -= 1
    # (the original recomputed Ucell = U - S @ neu here, but it is not used afterwards)
    if _use_gpu(device):
        torch.cuda.empty_cache()

    sdmov = np.reshape(sdmov, (Ly, Lx))
    stat = [{
        "ypix": ypix[n],
        "lam": lam[n] * sdmov[ypix[n], xpix[n]],
        "xpix": xpix[n]
    } for n in range(ncells)]

    stat = connected_region(stat, connected, Ly, Lx)
    # Remove empty cells
    stat = [s for s in stat if len(s["ypix"]) != 0]
    stat = np.array(stat)
    
    new_settings = {
        "diameter": diameter,
        "Vmax": 0,
        "ihop": 0,
        "Vsplit": 0,
        "Vcorr": Vcorr,
        "Vmap": 0,
        "spatscale_pix": 0
    }
    return new_settings, stat
