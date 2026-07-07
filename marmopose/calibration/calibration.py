# Portions of this file are derived from the Anipose project
# (BSD 2-Clause License, Copyright (c) 2019-2023, Lili Karashchuk)
# The code has been modified to suit local requirements

import pickle
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import json

import cv2
import numpy as np

from marmopose.calibration.cameras import CameraGroup
from marmopose.calibration.boards import Checkerboard, CharucoBoard
from marmopose.utils.data_io import load_axes
from marmopose.utils.helpers import orthogonalize_vector
from marmopose.config import Config

logger = logging.getLogger(__name__)


class Calibrator:
    def __init__(self, config: Config):
        self.config = config
        self.init_dir(config)

    def init_dir(self, config):
        self.calibration_path = Path(config.sub_directory['calibration'])
        self.calib_video_paths = sorted(self.calibration_path.glob(f"*.mp4"))
        self.output_path = self.calibration_path / 'camera_params.json'
    
    
    def calibrate(self):
        cam_names, video_list = self.get_video_list(self.calib_video_paths)
        board = self.get_calibration_board(self.config)
        print(board)

        if not self.output_path.exists():
            detected_file = self.calibration_path / 'detected_boards.pickle'
            if detected_file.exists():
                logger.info(f'Loading detected boards from: {detected_file}')
                with open(detected_file, 'rb') as f:
                    all_rows = pickle.load(f)
            else:
                logger.info('Detecting boards in videos...')
                all_rows = self.get_rows_videos(video_list, board)
                with open(detected_file, 'wb') as f:
                    pickle.dump(all_rows, f)

            cgroup = CameraGroup.from_names(cam_names, self.config.calibration['fisheye'])
            cgroup.set_camera_sizes_videos(video_list)

            cgroup.calibrate_rows(all_rows, board,
                                 init_intrinsics=True, init_extrinsics=True,
                                 initial_focal_length=self.config.calibration['initial_focal_length'],
                                 n_iters=10, start_mu=15, end_mu=1,
                                 max_nfev=200, ftol=1e-5,
                                 n_samp_iter=500, n_samp_full=1000,
                                 error_threshold=2.5, verbose=True)
        else:
            logger.info(f'Calibration result already exists in: {self.output_path}')
            cgroup = CameraGroup.load_from_json(str(self.output_path))
        
        if self.config.triangulation['user_define_axes']:
            self.update_extrinsics_by_user_define_axes(cgroup)
        
        cgroup.save_to_json(self.output_path)
        logger.info(f'Calibration done! Result stored in: {self.output_path}')
    
    def set_coordinates(self, video_inds: List, offset: Tuple[float, float, float], obj_name: str = 'axes', frame_idx: int = 0, order: Tuple[int, int, int] = (0, 1, 2), with_offset_point: bool = False) -> None:
        """
        Set coordinates for each camera by capturing from video frames.

        Args:
            video_inds: The index of videos for setting coordinates.
            offset: 3D Offset values (x, y, z).
            obj_name: Name of the object for which coordinates are being set.
            frame_idx: Frame index from which to capture the coordinates. Defaults to 0.
        """
        video_paths = sorted(Path(self.config.sub_directory['videos_raw']).glob(f"*.mp4"))

        coordinates_dict = {'offset': offset, 'order': order}
        for i, video_path in enumerate(video_paths):
            if i+1 not in video_inds:
                continue
            cam_name = video_path.stem
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            coordinates_dict[cam_name] = capture_coordinates(cap, cam_name, with_offset_point)
            cap.release()

        output_path = self.calibration_path / f'{obj_name}.json'
        with open(output_path, 'w') as f:
            json.dump(coordinates_dict, f, indent=4)

        logger.info(f'Coordinates of {obj_name} saved in: {output_path}')

    @staticmethod
    def get_video_list(video_paths):
        cam_videos = defaultdict(list)
        cam_names = set()
        for video_path in video_paths:
            name = video_path.stem
            cam_videos[name].append(str(video_path))
            cam_names.add(name)

        cam_names = sorted(cam_names)
        video_list = [sorted(cam_videos[cname]) for cname in cam_names]
        
        return cam_names, video_list
    
    @staticmethod
    def get_calibration_board(config: Config) -> Checkerboard:
        print(config.calibration)
        board_type = config.calibration['board_type']
        board_size = config.calibration['board_size']
        square_length = config.calibration['board_square_side_length']
        marker_length = config.calibration['marker_length']
        if board_type == 'checkerboard':
            return Checkerboard(squaresX=board_size[0], 
                                squaresY=board_size[1], 
                                square_length=square_length)
        elif board_type == 'charuco':
            return CharucoBoard(squaresX=board_size[0],
                                squaresY=board_size[1],
                                square_length=square_length,
                                marker_length=marker_length)
        else:
            raise ValueError("board_type has to be either checkerboard or charuco") 
    
    @staticmethod
    def get_rows_videos(video_list, board):
        all_rows = []

        for videos in video_list:
            rows_cam = []
            for vnum, vidname in enumerate(videos):
                logger.info(vidname)
                rows = board.detect_video(vidname, prefix=vnum, progress=True)
                logger.info(f"{len(rows)} boards detected")
                rows_cam.extend(rows)
            all_rows.append(rows_cam)

        return all_rows
    
    def update_extrinsics_by_user_define_axes(self, camera_group):
        axes_path = Path(self.calibration_path) / 'axes.json'
        if not axes_path.exists():
            logger.info(f'Axes file not found in: {axes_path}')
        else:
            logger.info('Updating extrinsics by user-defined axes')
            axes = load_axes(axes_path)
            T = construct_transformation_matrix(camera_group, axes)
            for camera in camera_group.cameras:
                update_camera_parameters(camera, T)
            test_updated_axes(camera_group, axes)

def test_updated_axes(camera_group, axes):
    offset = np.array(axes['offset'])
    cam_names = [key for key in axes.keys() if not key in ['offset','order']]
    axes_2d = np.array([axes[cam_name][:3] for cam_name in cam_names], dtype=np.float32)
    
    sub_camera_group = camera_group.subset_cameras_names(cam_names)

    if len(axes[cam_names[0]]) == 4:
        axes_3d = sub_camera_group.triangulate_ransac(axes_2d, undistort=True)
        offset_point_coords = np.array([axes[cam_name][3] for cam_name in cam_names], dtype=np.float32)[:, None, :]
        offset_point_3d = sub_camera_group.triangulate_ransac(offset_point_coords, undistort=True)

    else:
        axes_3d = sub_camera_group.triangulate(axes_2d, undistort=True) - offset
    
def update_camera_parameters(camera, T):
    # Update the rotation vector
    old_R, _ = cv2.Rodrigues(camera.get_rotation())
    new_R = old_R @ T[:3, :3].T 
    new_rvec, _ = cv2.Rodrigues(new_R)
    
    # Update the translation vector
    old_t = camera.get_translation()
    new_t = old_t - old_R @ T[:3, :3].T @ T[:3, 3]

    camera.set_rotation(new_rvec.flatten())
    camera.set_translation(new_t.flatten())


def construct_transformation_matrix(camera_group, axes):
    offset = np.array(axes['offset'])
    cam_names = [key for key in axes.keys() if not key in ['offset','order']]
    axes_2d = np.array([axes[cam_name][:3] for cam_name in cam_names], dtype=np.float32)
    
    sub_camera_group = camera_group.subset_cameras_names(cam_names)
    axes_3d = sub_camera_group.triangulate(axes_2d, undistort=True)

    if len(axes[cam_names[0]]) == 4:
        offset_point_coords = np.array([axes[cam_name][3] for cam_name in cam_names], dtype=np.float32)[:, None, :]
        origin = sub_camera_group.triangulate(offset_point_coords, undistort=True)[0]
    else:
        origin = axes_3d[0]

    new_x_axis = axes_3d[1] - axes_3d[0]
    new_y_axis = orthogonalize_vector(axes_3d[2] - axes_3d[0], new_x_axis)
    new_z_axis = np.cross(new_x_axis, new_y_axis)

    if 'order' in axes.keys():
        R = np.vstack([new_x_axis, new_y_axis, new_z_axis])[axes['order'],:]
    else:
        R = np.vstack([new_x_axis, new_y_axis, new_z_axis])
    R /= np.linalg.norm(R, axis=1)[:, None]
    
    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ (origin - offset.T @ R) 

    return T


def capture_event(event: int, x: int, y: int, flags: int, params: Tuple[List[Tuple[int, int]], int]) -> None:
    """
    Mouse callback function to add coordinates to the list when the left mouse button is clicked.

    Args:
        event: Type of mouse event.
        x: X-coordinate of mouse event.
        y: Y-coordinate of mouse event.
        flags: Any relevant flags related to the mouse event.
        params: Tuple containing list to which coordinates are appended and current point index.
    """
    cam_coordinates, current_point_idx = params
    if event == cv2.EVENT_LBUTTONDOWN:
        point_types = ['original point', 'x-axis point', 'y-axis point','offset point']
        print(f'{point_types[current_point_idx[0]]}: ({x}, {y})')
        cam_coordinates.append((x, y))
        current_point_idx[0] += 1


def capture_coordinates(cap: cv2.VideoCapture, cam_name: str, with_offset_point: bool = False) -> List[Tuple[int, int]]:
    """
    Display video frame and capture coordinates of mouse clicks on it.

    Args:
        cap: VideoCapture object from which frames are read.
        cam_name: Name of the camera for which coordinates are being captured.

    Returns:
        List of coordinates captured from the video frame.
    """
    n_points = 4 if with_offset_point else 3
    print(f'\nSetting axes for {cam_name}...')
    ret, img = cap.read()
    if not ret:
        return []

    cv2.imshow(cam_name, img)
    cam_coords = []
    current_point_idx = [0]
    cv2.setMouseCallback(cam_name, capture_event, (cam_coords, current_point_idx))

    while len(cam_coords) < n_points:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return cam_coords
