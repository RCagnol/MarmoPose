import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import optimize
from scipy.sparse import dok_matrix
from tqdm import trange

from marmopose.calibration.cameras import CameraGroup
from marmopose.processing.filter import interpolate_data, fill_hold

logger = logging.getLogger(__name__)

# Points closer to (or behind) a camera's image plane than this (mm) get a constant reprojection
# residual for that camera, instead of a projection that divides by ~0 or mirrors through the lens.
MIN_DEPTH = 10.0


def optimize_coordinates(
    config,
    camera_group: CameraGroup,
    points_3d: np.ndarray,
    points_with_score_2d: np.ndarray,
    start_frame: int = 0,
    batch_size: int = 7500,
) -> np.ndarray:
    """
    Optimize the 3D points by minimizing the reprojection error, smoothness error, limb length error.

    Args:
        config: Configuration dictionary.
        camera_group: The camera group corresponding to the points.
        points_3d: 3D points with shape (n_frames, n_bodyparts, 3), final channel (x, y, z).
        points_with_score_2d: 2D points with shape (n_cams, n_frames, n_bodyparts, 3), final channel (x, y, score).
        start_frame: Index of the first frame to optimize.
        batch_size: Number of frames to process in each batch.

    Returns:
        Optimized 3D points with shape (n_frames, n_bodyparts, 3).
    """
    n_deriv_smooth = config.optimization['n_deriv_smooth']
    scale_smooth = config.optimization['scale_smooth']
    scale_length = config.optimization['scale_length']
    scale_length_weak = config.optimization['scale_length_weak']
    max_interp_gap = config.optimization['max_interp_gap']
    max_nfev = config.optimization['max_nfev']
    ftol = config.optimization['ftol']
    saturation_px = config.optimization['reproj_saturation_px']

    bodypart_dist = parse_constraints(config, 'bodypart_distance')
    bodypart_dist_weak = parse_constraints(config, 'bodypart_distance_weak')

    points_3d_prior = points_3d
    points_3d_interp = np.apply_along_axis(interpolate_data, 0, points_3d_prior, max_gap=max_interp_gap)

    # Gaps longer than max_interp_gap are left NaN by interpolate_data (the subject was
    # likely out of frame, not just briefly occluded). The optimizer still needs finite
    # values to work with, so seed those frames by holding the nearest known position,
    # then restore the NaNs in the final result so they aren't reported as real data.
    missing_mask = np.isnan(points_3d_interp)
    points_3d_seed = np.apply_along_axis(fill_hold, 0, points_3d_interp) if missing_mask.any() else points_3d_interp

    # Reprojection is computed against undistorted (normalized) observations with a pinhole model.
    # Projecting through the lens distortion instead makes points far outside a camera's view
    # (e.g. a filled-in point seeded badly) produce residuals up to ~1e15 px, which stall the solver.
    n_cams = points_with_score_2d.shape[0]
    points_2d_norm = np.array([cam.undistort_points(np.copy(pts).reshape(-1, 2)).reshape(pts.shape)
                               for pts, cam in zip(points_with_score_2d[..., :2], camera_group.cameras)])
    scores_2d = np.nan_to_num(points_with_score_2d[..., 2], nan=0.0)
    cam_mats = np.array([cam.get_extrinsic_matrix()[:3] for cam in camera_group.cameras])  # (n_cams, 3, 4)
    focal_lengths = np.array([[cam.matrix[0, 0], cam.matrix[1, 1]] for cam in camera_group.cameras])  # (n_cams, 2)

    points_3d_original = points_3d_seed[:start_frame]
    points_3d_unprocessed = points_3d_seed[start_frame:]
    points_2d_norm_unprocessed = points_2d_norm[:, start_frame:]
    scores_2d_unprocessed = scores_2d[:, start_frame:]

    n_frames_unprocessed = points_3d_unprocessed.shape[0]

    num_batches = int(np.ceil(n_frames_unprocessed / batch_size))

    optimized_frames_list = []

    with trange(n_frames_unprocessed, ncols=100, desc="Optimizing coordinates", unit="frames") as progress_bar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_frames_unprocessed)

            points_3d_batch = points_3d_unprocessed[batch_start:batch_end]
            points_2d_norm_batch = points_2d_norm_unprocessed[:, batch_start:batch_end]
            scores_2d_batch = scores_2d_unprocessed[:, batch_start:batch_end]

            initial_params_batch = points_3d_batch.ravel()

            jac_sparsity_batch = get_jac_sparsity(
                points_2d_norm_batch,
                n_deriv_smooth,
                bodypart_dist,
                bodypart_dist_weak,
            )

            result = optimize.least_squares(
                fun=compute_residuals,
                x0=initial_params_batch,
                method='trf',
                loss='linear',  # Robustness is applied to the reprojection term only, see reprojection_residual
                ftol=ftol,
                max_nfev=max_nfev,
                jac_sparsity=jac_sparsity_batch,
                verbose=0,
                args=(
                    cam_mats,
                    focal_lengths,
                    saturation_px,
                    points_2d_norm_batch,
                    scores_2d_batch,
                    n_deriv_smooth,
                    scale_smooth,
                    scale_length,
                    scale_length_weak,
                    bodypart_dist,
                    bodypart_dist_weak,
                ),
            )

            points_3d_optimized_batch = result.x.reshape(points_3d_batch.shape)

            optimized_frames_list.append(points_3d_optimized_batch)

            frames_processed = batch_end - batch_start
            progress_bar.update(frames_processed)

    points_3d_optimized = np.vstack(optimized_frames_list)

    points_3d_result = np.vstack((points_3d_original, points_3d_optimized))
    points_3d_result[missing_mask] = np.nan

    return points_3d_result
    

def get_jac_sparsity(points_2d: np.ndarray, n_deriv_smooth: int,
                     bodypart_dist: List[Tuple[Tuple[int, int], float]], 
                     bodypart_dist_weak: List[Tuple[Tuple[int, int], float]]) -> dok_matrix:
    """
    Calculate Jacobian Sparsity Pattern.

    Args:
        points_2d: 2D points as input.
        n_deriv_smooth: Number of derivatives to smooth.
        bodypart_dist: Strong constraints on body parts.
        bodypart_dist_weak: Weak constraints on body parts.

    Returns:
        Sparse Jacobian Matrix.
    """
    n_cams, n_frames, n_bodyparts, _ = points_2d.shape
    n_constraints, n_constraints_weak = len(bodypart_dist), len(bodypart_dist_weak)
    points_2d_flat = points_2d[..., 0].ravel() # (n_cams * n_frames * n_bodyparts,)
    mask_valid = ~np.isnan(points_2d_flat)

    n_errors_reproj = np.sum(mask_valid)
    n_errors_smooth = (n_frames-n_deriv_smooth) * n_bodyparts
    n_errors_lengths = n_constraints * n_frames
    n_errors_lengths_weak = n_constraints_weak * n_frames
    n_errors = n_errors_reproj + n_errors_smooth + n_errors_lengths + n_errors_lengths_weak
    logger.debug(f'Optimizing {n_errors_reproj} reprojection errors, {n_errors_smooth} smoothness errors, {n_errors_lengths} limb length errors, {n_errors_lengths_weak} weak limb length errors')

    sparse_jac = dok_matrix((n_errors, n_frames*n_bodyparts*3), dtype='int16')

    # Setting the sparsity pattern for reprojection errors
    indices_params = np.tile(np.arange(n_frames*n_bodyparts), n_cams)
    indices_params_valid = indices_params[mask_valid]
    indices_reproj = np.arange(n_errors_reproj)
    for k in range(3):
        sparse_jac[indices_reproj, indices_params_valid*3 + k] = 1

    # Setting the sparsity pattern for smoothness constraint
    frames = np.arange(n_frames-n_deriv_smooth)
    for j in range(n_bodyparts):
        for n in range(n_deriv_smooth+1):
            pa = frames*n_bodyparts + j
            pb = (frames+n)*n_bodyparts + j
            for k in range(3):
                sparse_jac[n_errors_reproj + pa, pb*3 + k] = 1
    
    # Setting the sparsity pattern for strong constraints
    start = n_errors_reproj + n_errors_smooth
    point_indices_3d = np.arange(n_frames*n_bodyparts).reshape((n_frames, n_bodyparts))
    frames = np.arange(n_frames)
    all_constraints = bodypart_dist + bodypart_dist_weak
    for cix, ((bp1, bp2), length) in enumerate(all_constraints):
        pa = point_indices_3d[frames, bp1]
        pb = point_indices_3d[frames, bp2]
        for k in range(3):
            sparse_jac[start + cix*n_frames + frames, pa*3 + k] = 1
            sparse_jac[start + cix*n_frames + frames, pb*3 + k] = 1

    return sparse_jac


def compute_residuals(points_3d_flat: np.ndarray, *args: Tuple) -> np.ndarray:
    """
    Compute Residuals for Optimization.

    Args:
        points_3d_flat: Flattened 3D points.
        *args: Additional arguments including camera settings, 2D points, etc.

    Returns:
        Residuals.
    """
    cam_mats, focal_lengths, saturation_px, points_2d_norm, scores_2d, n_deriv_smooth, scale_smooth, \
        scale_length, scale_length_weak, bodypart_dist, bodypart_dist_weak = args

    n_cams, n_frames, n_joints, _ = points_2d_norm.shape
    points_3d = points_3d_flat.reshape((n_frames, n_joints, 3))
    errors_reproj = reprojection_residual(points_3d, points_2d_norm, scores_2d, cam_mats, focal_lengths, saturation_px)
    errors_smooth = smoothness_residual(points_3d, n_deriv_smooth, scale_smooth)
    errors_lengths = bodypart_length_residual(points_3d, bodypart_dist, bodypart_dist_weak, scale_length, scale_length_weak)
    
    residuals = np.hstack((errors_reproj, errors_smooth, errors_lengths))
    return residuals


def reprojection_residual(points_3d: np.ndarray, points_2d_norm: np.ndarray, scores_2d: np.ndarray,
                          cam_mats: np.ndarray, focal_lengths: np.ndarray, saturation_px: float) -> np.ndarray:
    """
    Calculate Reprojection Residuals, Cauchy-saturated so outliers cannot dominate the cost.

    Each residual r (pixels) becomes delta * sqrt(log(1 + (r/delta)^2)) with delta = saturation_px:
    unchanged for r << delta, growing only logarithmically beyond. This robustifies the reprojection
    term alone; scipy's `loss=` would also weaken the bone-length and smoothness terms.

    Args:
        points_3d: 3D coordinates of points, shape (n_frames, n_bodyparts, 3).
        points_2d_norm: Undistorted normalized 2D observations, shape (n_cams, n_frames, n_bodyparts, 2).
        scores_2d: Detection scores, NaN replaced by 0, shape (n_cams, n_frames, n_bodyparts).
        cam_mats: Extrinsic matrices, shape (n_cams, 3, 4).
        focal_lengths: (fx, fy) per camera, shape (n_cams, 2), to express errors in pixels.
        saturation_px: Cauchy scale delta, in pixels.

    Returns:
        Reprojection residuals for valid observations, shape (n_valid,).
    """
    n_cams = points_2d_norm.shape[0]
    points_3d_flat = points_3d.reshape(-1, 3)
    points_2d_flat = points_2d_norm.reshape((n_cams, -1, 2))

    points_cam = np.einsum('cij,nj->cni', cam_mats[:, :, :3], points_3d_flat) + cam_mats[:, np.newaxis, :, 3]  # (n_cams, N, 3)
    depth = points_cam[..., 2]
    in_front = depth > MIN_DEPTH
    projected = points_cam[..., :2] / np.where(in_front, depth, 1.0)[..., np.newaxis]

    errors = np.linalg.norm((projected - points_2d_flat) * focal_lengths[:, np.newaxis, :], axis=2)  # (n_cams, N), NaN where unobserved
    errors = saturation_px * np.sqrt(np.log1p((errors / saturation_px) ** 2))

    # Constant (zero-gradient) cost for observed points behind the camera: the pinhole projection is meaningless there
    errors[~in_front & ~np.isnan(errors)] = saturation_px * np.sqrt(np.log1p(1e8))

    errors = errors * scores_2d.reshape((n_cams, -1))

    errors_valid = errors[~np.isnan(errors)] # (n_cams * valid n_frames*n_bodyparts,)
    return errors_valid


def smoothness_residual(points_3d: np.ndarray, n_deriv_smooth: int, scale_smooth: float) -> np.ndarray:
    """
    Calculate Smoothness Residuals.

    Args:
        points_3d: 3D coordinates of points.
        n_deriv_smooth: Number of derivatives for smoothing.
        scale_smooth: Scaling factor for smoothness.

    Returns:
        Smoothness residuals.
    """
    diff = np.diff(points_3d, n=n_deriv_smooth, axis=0)
    # TODO: Maybe not L2 norm, squared L2 norm?
    errors = np.linalg.norm(diff, axis=2).ravel() * scale_smooth # (n_frames-n_deriv_smooth * n_bodyparts,)

    return errors


def bodypart_length_residual(points_3d: np.ndarray,
                             bodypart_dist: List[Tuple[Tuple[int, int], float]],
                             bodypart_dist_weak: List[Tuple[Tuple[int, int], float]],
                             scale_length: float, scale_length_weak: float) -> np.ndarray:
    """
    Calculate Body Part Length Residuals.

    Args:
        points_3d : 3D coordinates of points.
        bodypart_dist: Strong constraints for body parts.
        bodypart_dist_weak: Weak constraints for body parts.
        scale_length: Scaling factor for strong constraints.
        scale_length_weak: Scaling factor for weak constraints.

    Returns:
         Length residuals.
    """
    n_frames = points_3d.shape[0]

    # Lengths within `tolerance` of the expected length cost nothing; beyond it the penalty grows linearly
    errors = np.empty((len(bodypart_dist), n_frames), dtype='float64')
    for cix, ((bp1, bp2), (expected_length, tolerance)) in enumerate(bodypart_dist):
        actual_lengths = np.linalg.norm(points_3d[:, bp1] - points_3d[:, bp2], axis=1)
        # TODO: Maybe not L2 norm, squared L2 norm?
        errors[cix] = np.maximum(np.abs(actual_lengths - expected_length) - tolerance, 0)
    errors = errors.ravel() * scale_length # (n_constraints * n_frames,)

    errors_weak = np.empty((len(bodypart_dist_weak), n_frames), dtype='float64')
    for cix, ((bp1, bp2), (expected_length, tolerance)) in enumerate(bodypart_dist_weak):
        actual_lengths = np.linalg.norm(points_3d[:, bp1] - points_3d[:, bp2], axis=1)
        errors_weak[cix] = np.maximum(np.abs(actual_lengths - expected_length) - tolerance, 0)
    errors_weak = errors_weak.ravel() * scale_length_weak

    errors = np.hstack((errors, errors_weak))
    return errors


def parse_constraints(config: Dict[str, Any], key: str) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
    """
    Parse Body Part Constraints from Configuration.

    Each entry is either `'a - b': length` (exact target) or `'a - b': [length, tolerance]`
    (any length within length ± tolerance is unpenalized).

    Args:
        config: Configuration dictionary.
        key: The key to look for in the dictionary.

    Returns:
        Parsed constraints as ((bp1, bp2), (length, tolerance)).
    """
    bodyparts = config.animal['bodyparts']
    bodypart_indices = {bp_name: idx for idx, bp_name in enumerate(bodyparts)}

    constraint_dict = config.optimization[key]
    constraint_list = []
    for key, value in constraint_dict.items():
        bp = tuple([bodypart_indices[bp.strip()] for bp in key.split('-')])
        length, tolerance = (value, 0.0) if np.isscalar(value) else value
        constraint_list.append((bp, (float(length), float(tolerance))))

    return constraint_list