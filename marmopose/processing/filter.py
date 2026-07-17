import numpy as np


def interpolate_data(values: np.ndarray, max_gap: int = None) -> np.ndarray:
    """
    Interpolates data to fill NaN values.

    Args:
        values: The data to be interpolated.
        max_gap: If set, contiguous NaN runs longer than this many frames are
            left as NaN instead of being bridged. Long runs usually mean the
            subject was out of frame rather than briefly occluded, so a
            straight-line interpolation across them is not meaningful.

    Returns:
        The interpolated data. Runs longer than max_gap remain NaN.
    """
    nans = np.isnan(values)
    if not nans.any():
        return np.copy(values)
    if nans.all():
        return np.zeros_like(values)

    idx = lambda z: np.nonzero(z)[0]
    out = np.copy(values)
    out[nans] = np.interp(idx(nans), idx(~nans), values[~nans])

    if max_gap is not None:
        out[find_long_gaps(nans, max_gap)] = np.nan

    return out


def find_long_gaps(nans: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Finds contiguous runs of True in `nans` longer than max_gap frames.

    Args:
        nans: Boolean array marking missing values.
        max_gap: Maximum run length (in frames) that is not considered "long".

    Returns:
        Boolean mask, same shape as `nans`, marking the long runs.
    """
    padded = np.concatenate(([False], nans, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    mask = np.zeros_like(nans)
    for start, end in zip(starts, ends):
        if end - start > max_gap:
            mask[start:end] = True
    return mask


def fill_hold(values: np.ndarray) -> np.ndarray:
    """
    Fills NaNs by holding the nearest preceding valid value. Leading NaNs
    (no valid value yet seen) hold the first valid value instead.

    Args:
        values: The data to be filled.

    Returns:
        The filled data, used as a solver seed - not a claim about the
        true (unknown) position during long gaps.
    """
    valid_idx = np.flatnonzero(~np.isnan(values))
    if valid_idx.size == 0:
        return np.zeros_like(values)

    positions = np.arange(len(values))
    nearest_prior = np.searchsorted(valid_idx, positions, side='right') - 1
    nearest_prior[nearest_prior < 0] = 0

    return values[valid_idx[nearest_prior]]
