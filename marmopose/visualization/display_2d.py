import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import av
import cv2
import skvideo.io
import numpy as np
from tqdm import trange

from marmopose.config import Config
from marmopose.calibration.cameras import CameraGroup
from marmopose.utils.data_io import load_points_bboxes_2d_h5, load_points_3d_h5
from marmopose.utils.helpers import get_color_list

logger = logging.getLogger(__name__)


class Visualizer2D:
    def __init__(self, config: Config, with_gaze = False):
        self.with_gaze = with_gaze
        self.init_dir(config)
        self.init_visual_cfg(config)
        if with_gaze:
            self.camera_group = CameraGroup.load_from_json(Path(config.sub_directory['calibration']) / 'camera_params.json')
        else:
            self.camera_group = None

    def init_dir(self, config):
        self.config = config
        self.videos_raw_dir = Path(config.sub_directory['videos_raw'])
        self.points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
        if self.with_gaze:
            self.points_3d_path = Path(config.sub_directory['points_3d']) / 'optimized.h5'
        self.videos_labeled_2d_dir = Path(config.sub_directory['videos_labeled_2d'])

    def init_visual_cfg(self, config):
        bodyparts = config.animal['bodyparts']
        skeleton = config.visualization['skeleton']
        self.skeleton_indices = [[bodyparts.index(bp) for bp in line] for line in skeleton]
        self.skeleton_color_list = get_color_list(config.visualization['skeleton_cmap'], number=len(skeleton))
        if self.with_gaze:
            self.skeleton_indices += [[bodyparts.index('head'),len(bodyparts)]]
            self.skeleton_color_list += [self.skeleton_color_list[0]]

        # TODO: The order is specified to be consistent with the marmoset dataset
        colors = get_color_list(config.visualization['track_cmap'])
        new_order = [1, 0, 4, 3, 2]
        self.track_color_list = [colors[i] if i < len(new_order) else colors[i] for i in new_order] + colors[len(new_order):]

    def generate_videos_2d(self, file_names: list = None):
        # TODO: Visualize specific video, visualize a certain range of frames
        if file_names is None:
            video_paths = sorted(self.videos_raw_dir.glob(f"*.mp4"))
        else:
            video_paths = sorted([self.videos_raw_dir / (file + '.mp4') for file in file_names])
        all_points_with_score_2d, all_bboxes = load_points_bboxes_2d_h5(self.points_2d_path, file_names)
        if self.with_gaze:
            gaze_vectors = self.generate_gaze_vectors()
            norm_gaze_vectors = np.sqrt(np.einsum('ijkl,ijkl -> ijk', gaze_vectors, gaze_vectors))[..., np.newaxis]
            gaze_vectors = np.concatenate((gaze_vectors,np.zeros((*gaze_vectors.shape[:3],1))),axis=3)
            idx_head = self.config.animal['bodyparts'].index('head')
            gaze_points = (gaze_vectors* 400/norm_gaze_vectors)[..., np.newaxis,:] + all_points_with_score_2d[... ,idx_head:idx_head+1,:]
            all_points_with_score_2d = np.concatenate((all_points_with_score_2d, gaze_points), axis = 3)
        with ThreadPoolExecutor() as executor:
            futures = []
            for video_path, points_with_score_2d, bboxes in zip(video_paths, all_points_with_score_2d, all_bboxes):
                output_path = self.videos_labeled_2d_dir / video_path.name
                future = executor.submit(self.render_video_with_pose, video_path, points_with_score_2d, bboxes, output_path)
                futures.append(future)
            for future in as_completed(futures):
                future.result()
    
    def render_video_with_pose(self, video_path: Path, points_with_score_2d: np.ndarray, bboxes: np.ndarray, output_path: Path):
        input_container = av.open(video_path)
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = 'AUTO'

        writer = skvideo.io.FFmpegWriter(output_path, inputdict={'-framerate': str(input_stream.average_rate)},
                                         outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

        n_frames = points_with_score_2d.shape[1]
        for frame_idx, frame in zip(trange(n_frames, ncols=100, desc=f'2D Visualizing {output_path.stem}', unit='frames'), input_container.decode(video=0)):
            # TODO: Try to add score to the visualization
            points_2d = points_with_score_2d[:, frame_idx, :, :2]  # (n_tracks, n_bodyparts, 2)
            bbox = bboxes[:, frame_idx]  # (n_tracks, 4)

            img = frame.to_ndarray(format='rgb24')
            img = self.draw_pose_on_image(img, points_2d, bbox)

            writer.writeFrame(img)
        
        input_container.close()
        writer.close()

    def draw_pose_on_image(self, img: np.ndarray, points_2d: np.ndarray, bboxes: np.ndarray) -> None:
        for track_idx, (points, bbox) in enumerate(zip(points_2d, bboxes)):
            # Draw bounding boxes
            if not np.any(np.isnan(bbox)):
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), self.track_color_list[track_idx], 2)
            # Draw points
            valid_points = points[~np.isnan(points).any(axis=-1)].astype(int)
            for x, y in valid_points:
                cv2.circle(img, (x, y), 6, self.track_color_list[track_idx], -1)
            # Draw lines
            self.draw_lines(img, points)

        return img

    def draw_lines(self, img: np.ndarray, points: np.ndarray) -> None:
        for idx, bodypart_indices in enumerate(self.skeleton_indices):
            for a, b in zip(bodypart_indices, bodypart_indices[1:]):
                if np.any(np.isnan(points[[a,b]])):
                    continue
                pa, pb = tuple(map(int, points[a])), tuple(map(int, points[b]))
                cv2.line(img, pa, pb, self.skeleton_color_list[idx], 2)

    def generate_gaze_vectors(self):
        all_points_3d = load_points_3d_h5(self.points_3d_path)
        point_head = np.nonzero(np.array(self.config.animal["bodyparts"]) == "head")[0]
        point_leftear = np.nonzero(np.array(self.config.animal["bodyparts"]) == "leftear")[0]
        point_rightear = np.nonzero(np.array(self.config.animal["bodyparts"]) == "rightear")[0]
        head_coordinates = all_points_3d[:,:,point_head,:]
        leftear_coordinates = all_points_3d[:,:,point_leftear,:]
        rightear_coordinates = all_points_3d[:,:,point_rightear,:]
        middleear_coordinates = (leftear_coordinates + rightear_coordinates)/2
        vectors = np.array([cam.project(head_coordinates)-cam.project(middleear_coordinates) for cam in self.camera_group.cameras])
        vectors = vectors[:,np.newaxis, ...]
        return vectors
