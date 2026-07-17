import logging
from importlib import reload
import shutil
import os
import sys
import tempfile
import argparse
import csv
import numpy as np
from pathlib import Path

from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.processing.prediction import Predictor
from marmopose.visualization.display_2d import Visualizer2D
from marmopose.visualization import display_3d
from marmopose.visualization.display_3d import Visualizer3D, Visualizer3DCombined
from marmopose.processing.triangulation import Reconstructor3D
from marmopose.utils.data_io import get_offset_from_point, load_points_3d_h5
from marmopose.utils.analysis import remove_absurd_3d_data, export_session_data, read_cage_sessions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f'MarmoPose version: {marmopose_version}')

config_path = '../configs/default.yaml'
# Kept on /scratch (local disk) rather than the OS default temp dir, since this
# holds full-resolution intermediate videos/frames.
SCRATCH_ROOT = Path('../demos/.analysis_scratch')

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directories", nargs="+", required=True, help="Recording directories")
parser.add_argument("-c", "--cage", choices=["etho", "home", "both", "combined", "full"], required=True, help="Cage, must be etho, home or both")
parser.add_argument("-m", "--model", choices=["base", "finetune"], required=True, help="Model, must be base or finetune")
parser.add_argument("-n", "--novideos", action="store_true", help="Disable video creation")
parser.add_argument("--skip-2d", action="store_true", default=False, help="Skip 2D pose estimation if already present")
parser.add_argument("--skip-visualization", action="store_true", default=False, help="Skip generation of labeled 2D and 3D videos")

args = parser.parse_args()
cage1 = args.cage
model = args.model
directories = args.directories
combined = cage1 == 'combined' or cage1 == 'full'
both = cage1 == 'both' or cage1 == 'full'

VIDEO_DIR = Path('/srv/MarmOT/VideoTracking/Videos/')
SCRATCH_OUTPUT_DIR = Path('/scratch/VideoTracking/Videos/')
# Accumulates one row per session per animal, across all recording directories/dates.
CSV_DIR = Path('/srv/MarmOT/ISCMJ/Video_data')
home_dimensions = [1200, 815, 900, 30]
etho_dimensions = [660, 560, 800, 30]

if cage1 == 'home':
    room_dimensions = home_dimensions
    cage2 = 'etho'

elif cage1 == 'etho':
    room_dimensions = etho_dimensions
    cage2 = 'home'

if model == 'finetune':
    if both or combined:
        det_model_etho = f'../data/detection_model_finetune_etho_with_home'
        pose_model_etho = f'../data/pose_model_finetune_etho_with_home'
        det_model_home = f'../data/detection_model_finetune_home_with_etho'
        pose_model_home = f'../data/pose_model_finetune_home_with_etho'
    else:
        det_model = f'../data/detection_model_finetune_{cage1}_with_{cage2}'
        pose_model = f'../data/pose_model_finetune_{cage1}_with_{cage2}'
        model_output_directory = 'Output'
elif model == 'base':
    if both or combined:
        det_model_etho = '../models/detection_model'
        pose_model_etho = '../models/pose_model'
        det_model_home = '../models/detection_model'
        pose_model_home = '../models/pose_model'
    else:
        det_model = '../models/detection_model'
        pose_model = '../models/pose_model'
        model_output_directory = 'Output_basemodel'


def run_pipeline_on_dir(src_dir, dist_dir, det_model, pose_model, config, shift_frames, skip_2d_if_present = False, skip_visualization = False, suffix_3d = None):
    dir_config_videos = Path(config.sub_directory['videos_raw'])
    logger.info(src_dir)
    output_dir = dist_dir / 'Output'
    dist_calib_dir = dist_dir / 'Calib'


    calib_dir = src_dir / 'Calib_preprocessed'

    input_dir = src_dir / 'Input_preprocessed'
    assert input_dir.is_dir(), f'Missing input directory: {input_dir}'
    videos_input = [video for video in os.listdir(input_dir) if video[-4:] == '.mp4']
    assert videos_input, f'No videos found in {input_dir}'

    videos_present = [] if not os.path.isdir(output_dir / 'videos_labeled_2d') else [video for video in os.listdir(output_dir / 'videos_labeled_2d') if video[-4:] == '.mp4']
    skip_2d = True if skip_2d_if_present and len(videos_input) == len(videos_present) else False

    if True and not skip_2d:
        for video in videos_input:
            if video[-4:] == '.mp4':
                shutil.copy(input_dir / video, dir_config_videos)

    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dist_calib_dir, exist_ok=True)

    calib_camera_params = calib_dir / f'camera_params{suffix_3d}.json'
    calib_axes = calib_dir / f'axes{suffix_3d}.json'
    calib_boards = calib_dir / 'detected_boards.pickle'
    for calib_file in (calib_camera_params, calib_axes, calib_boards):
        assert calib_file.is_file(), f'Missing calibration file: {calib_file}'

    shutil.copy(calib_camera_params, Path(config.sub_directory['calibration']) / 'camera_params.json')
    shutil.copy(calib_camera_params, dist_calib_dir)
    shutil.copy(calib_axes, dist_calib_dir)
    shutil.copy(calib_boards, dist_calib_dir)

    if not skip_2d:
        predictor = Predictor(config, batch_size=4)
        predictor.predict(frames_indices = shift_frames)

        shutil.copytree(config.sub_directory['points_2d'], output_dir / 'points_2d', dirs_exist_ok=True)
        file_names = predictor.file_names

    else:
        shutil.copytree(output_dir / 'points_2d', config.sub_directory['points_2d'], dirs_exist_ok=True)
        file_names = [video[:-4] for video in videos_input]

    reconstructor_3d = Reconstructor3D(config)
    reconstructor_3d.triangulate(file_names = file_names)
    shutil.copytree(config.sub_directory['points_3d'], output_dir / f'points_3d{suffix_3d}', dirs_exist_ok=True)

    if not skip_visualization:
        if skip_2d:
            shutil.copytree(output_dir / 'videos_labeled_2d', config.sub_directory['videos_labeled_2d'], dirs_exist_ok=True)
        else:
            visualizer_2d = Visualizer2D(config)
            visualizer_2d.generate_videos_2d(file_names = file_names, frames_indices = shift_frames)
            shutil.copytree(config.sub_directory['videos_labeled_2d'], output_dir / 'videos_labeled_2d', dirs_exist_ok=True)

        visualizer_3d = Visualizer3D(config)
        # visualizer_3d.generate_video_3d(source_3d='optimized', video_type='composite', file_names_2d = [f'output{i}' for i in [1,3]])
        visualizer_3d.generate_video_3d(source_3d='optimized', video_type='composite', file_names_2d = file_names)
        shutil.copytree(config.sub_directory['videos_labeled_3d'], output_dir / 'videos_labeled_3d', dirs_exist_ok=True)

    for video in os.listdir(dir_config_videos):
        try:
            os.remove(dir_config_videos / video)
        except OSError:
            pass


def run_pipeline_on_dir_combined(etho_dir, home_dir, output_dir, session, names_etho_home, skip_visualization = False):
    name_etho, name_home = names_etho_home

    etho_points_3d_path = output_dir / 'Etho' / 'Output'/ 'points_3d_both' / 'optimized.h5'
    home_points_3d_path = output_dir / 'Home' / 'Output'/ 'points_3d_both' / 'optimized.h5'

    shutil.copy(etho_points_3d_path, Path(config_etho.sub_directory['points_3d']) / 'optimized.h5')
    shutil.copy(home_points_3d_path, Path(config_etho.sub_directory['points_3d']) / 'optimized_home.h5')

    offset_etho = get_offset_from_point(etho_dir / 'Calib_preprocessed')
    offset_home = get_offset_from_point(home_dir / 'Calib_preprocessed')

    output_combined_dir = output_dir / 'Output_combined'
    os.makedirs(output_combined_dir, exist_ok=True)

    etho_points_3d = load_points_3d_h5(etho_points_3d_path)[0]
    home_points_3d = load_points_3d_h5(home_points_3d_path)[0]

    duration = np.min((etho_points_3d.shape[0],home_points_3d.shape[0]))
    etho_points_3d = etho_points_3d[:duration,:,:]
    etho_points_3d = remove_absurd_3d_data(etho_points_3d, offset_etho, etho_dimensions)
    home_points_3d = home_points_3d[:duration,:,:]
    home_points_3d = remove_absurd_3d_data(home_points_3d, offset_home, home_dimensions)

    idx_head = config_etho.animal['bodyparts'].index('head')
    idx_leftear = config_etho.animal['bodyparts'].index('leftear')
    idx_rightear = config_etho.animal['bodyparts'].index('rightear')

    export_session_data(
        etho_points_3d, home_points_3d,
        name_etho, name_home,
        offset_etho, offset_home,
        etho_dimensions, home_dimensions,
        idx_head, idx_leftear, idx_rightear,
        indices_position = (idx_head, idx_leftear, idx_rightear),
        session = session,
        session_output_dir = output_combined_dir,
        csv_dir = CSV_DIR,
        skip_visualization = skip_visualization,
        positions_bin_size = 30,
    )

    if not skip_visualization:
        etho_sees_home = np.load(output_combined_dir / f'{name_etho}_sees_{name_home}.npy')
        home_sees_etho = np.load(output_combined_dir / f'{name_home}_sees_{name_etho}.npy')
        joint_gaze = np.load(output_combined_dir / 'joint_gaze.npy')

        visualizer_3d_combined = Visualizer3DCombined(config_etho, room_dimensions = (etho_dimensions, home_dimensions), offsets = (offset_etho, offset_home), is_seen = (home_sees_etho, etho_sees_home), joint_gaze = joint_gaze, with_gaze=True)
        visualizer_3d_combined.generate_video_3d(source_3d='optimized')
        shutil.copy(Path(config_etho.sub_directory['videos_labeled_3d']) / 'optimized_combined.mp4', output_combined_dir)

    try:
        os.remove(Path(config_etho.sub_directory['points_3d']) / 'optimized_home.h5')
    except OSError:
        pass


def get_shift_from_led_dict(led_dict, etho_input_dir, home_input_dir):
    if not etho_input_dir is None:
        led_frames_etho = list(led_dict['etho'].values())
        videos_input_etho = sorted([video[:-4] for video in os.listdir(etho_input_dir) if video[-4:] == '.mp4'])

    else:
        led_frames_etho = []

    if not home_input_dir is None:
        led_frames_home = list(led_dict['home'].values())
        videos_input_home = sorted([video[:-4] for video in os.listdir(home_input_dir) if video[-4:] == '.mp4'])
    else:
        led_frames_home = []
    led_frames = led_frames_etho + led_frames_home
    max_led_frame = max(led_frames)
    min_led_frame = min(led_frames)

    if not etho_input_dir is None:
        shift_frames_etho = [(led_dict['etho'][video], led_dict['etho'][video] - max_led_frame) for video in videos_input_etho]
    else:
        shift_frames_etho = None
    if not home_input_dir is None:
        shift_frames_home = [(led_dict['home'][video], led_dict['home'][video] - max_led_frame) for video in videos_input_home]
    else:
        shift_frames_home = None
    return shift_frames_etho, shift_frames_home


# Each session (one iteration of the loop below) gets its own scratch project
# directory, instead of every invocation and every session sharing
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

for directory in args.directories:
    directory_path = VIDEO_DIR / directory
    assert os.path.exists(directory_path), f'{directory} not present in {VIDEO_DIR}'

    etho_path = directory_path / 'Etho'
    home_path = directory_path / 'Home'

    path_led_frames = directory_path / 'led_frames.csv'
    if os.path.isfile(path_led_frames):
        led_frames_dict = {}
        with open(path_led_frames, 'r') as led_file:
            led_frames_reader = csv.reader(led_file)
            first_line = next(led_frames_reader)
            if first_line[0] == '#':
                comment, *keys = first_line
            else:
                keys = first_line
            values = next(led_frames_reader)


        led_frames_dict['home'] = dict([('output' + key[-1], int(value)) for key, value in zip(keys, values) if key[:-1] == 'home'])
        led_frames_dict['etho'] = dict([('output' + key[-1], int(value)) for key, value in zip(keys, values) if key[:-1] == 'etho'])
        if both or combined:
            shift_frames_etho, shift_frames_home = get_shift_from_led_dict(led_frames_dict, etho_path / 'Input_preprocessed', home_path / 'Input_preprocessed')
        elif cage1 == 'home':
            _, shift_frames = get_shift_from_led_dict(led_frames_dict, None, home_path / 'Input_preprocessed')
        elif cage1 == 'etho':
            shift_frames, _ = get_shift_from_led_dict(led_frames_dict, etho_path / 'Input_preprocessed', None)

    else:
        shift_frames_etho = shift_frames_home = shift_frames = None

    if both or combined:
        skip_visualization = args.skip_visualization
        suffix = '_both' if combined else ''

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as project_dir_etho, \
             tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as project_dir_home:
            config_etho = Config(
                config_path=config_path,
                n_tracks=1,
                project=project_dir_etho,
                det_model = det_model_etho,
                pose_model = pose_model_etho,
                room_dimensions = etho_dimensions,
                # do_optimize=False,
            )
            Path(config_etho.sub_directory['calibration']).mkdir(exist_ok=True)
            Path(config_etho.sub_directory['videos_raw']).mkdir(exist_ok=True)

            if both:
                config_home = Config(
                    config_path=config_path,
                    n_tracks=1,
                    project=project_dir_home,
                    det_model = det_model_home,
                    pose_model = pose_model_home,
                    room_dimensions = home_dimensions,
                    # do_optimize=False,
                )
                Path(config_home.sub_directory['calibration']).mkdir(exist_ok=True)
                Path(config_home.sub_directory['videos_raw']).mkdir(exist_ok=True)
                scratch_etho = SCRATCH_OUTPUT_DIR / directory / 'Etho'
                scratch_home = SCRATCH_OUTPUT_DIR / directory / 'Home'
                run_pipeline_on_dir(etho_path, scratch_etho, det_model_etho, pose_model_etho, config_etho, shift_frames_etho, skip_2d_if_present = args.skip_2d, suffix_3d = suffix, skip_visualization = skip_visualization)
                run_pipeline_on_dir(home_path, scratch_home, det_model_home, pose_model_home, config_home, shift_frames_home, skip_2d_if_present = args.skip_2d, suffix_3d = suffix, skip_visualization = skip_visualization)
                output_etho = etho_path / 'Output'
                output_home = home_path / 'Output'
                output_scratch_etho = scratch_etho / 'Output'
                output_scratch_home = scratch_home / 'Output'
                output_etho.mkdir(exist_ok=True)
                output_home.mkdir(exist_ok=True)
                shutil.copytree(output_scratch_etho / 'points_2d', output_etho / 'points_2d', dirs_exist_ok=True)
                shutil.copytree(output_scratch_home / 'points_2d', output_home / 'points_2d', dirs_exist_ok=True)
                shutil.copytree(output_scratch_etho / f'points_3d{suffix}', output_etho / 'points_3d', dirs_exist_ok=True)
                shutil.copytree(output_scratch_home / f'points_3d{suffix}', output_home / 'points_3d', dirs_exist_ok=True)

            if combined:
                run_pipeline_on_dir_combined(etho_path, home_path, SCRATCH_OUTPUT_DIR / directory, directory, read_cage_sessions(directory), skip_visualization=skip_visualization)
                output_combined = directory_path / 'Output_combined'
                output_scratch_combined = SCRATCH_OUTPUT_DIR / directory / 'Output_combined'
                output_combined.mkdir(exist_ok=True)
                for file in os.listdir(output_scratch_combined):
                    if file[-4:] == '.npy':
                        shutil.copy(output_scratch_combined / file, output_combined)


    else:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as project_dir:
            config = Config(
                config_path=config_path,
                n_tracks=1,
                project=project_dir,
                det_model = det_model,
                pose_model = pose_model,
                room_dimensions = room_dimensions,
                # do_optimize=False,
            )
            Path(config.sub_directory['calibration']).mkdir(exist_ok=True)
            Path(config.sub_directory['videos_raw']).mkdir(exist_ok=True)

            idx_head = config.animal['bodyparts'].index('head')
            idx_leftear = config.animal['bodyparts'].index('leftear')
            idx_rightear = config.animal['bodyparts'].index('rightear')
            name_etho, name_home = read_cage_sessions(directory)
            name = name_etho if cage1 == 'etho' else name_home

            Cage = cage1[0].upper() + cage1[1:]
            directory_path_cage = directory_path / Cage
            assert os.path.exists(directory_path_cage), f'{Cage} not present in {directory_path}'
            scratch_cage = SCRATCH_OUTPUT_DIR / directory / Cage
            run_pipeline_on_dir(directory_path_cage, scratch_cage, det_model, pose_model, config, shift_frames, skip_2d_if_present = args.skip_2d, skip_visualization = args.skip_visualization, suffix_3d = '')
            output_cage = directory_path_cage / 'Output'
            shutil.copytree(scratch_cage / 'Output' / 'points_2d', output_cage / 'points_2d', dirs_exist_ok=True)
            shutil.copytree(scratch_cage / 'Output' / 'points_3d', output_cage / 'points_3d', dirs_exist_ok=True)

            offset = get_offset_from_point(directory_path_cage / 'Calib_preprocessed')
            points_3d = load_points_3d_h5(output_cage / 'points_3d' / 'optimized.h5')[0]
            points_3d = remove_absurd_3d_data(points_3d, offset, room_dimensions)
            session_output_dir = output_cage / 'Output_analysis'
            os.makedirs(session_output_dir, exist_ok=True)

            export_session_data(
                points_3d if cage1 == 'etho' else None, points_3d if cage1 == 'home' else None,
                name if cage1 == 'etho' else None, name if cage1 == 'home' else None,
                offset if cage1 == 'etho' else None, offset if cage1 == 'home' else None,
                room_dimensions if cage1 == 'etho' else None, room_dimensions if cage1 == 'home' else None,
                idx_head, idx_leftear, idx_rightear,
                indices_position = (idx_head, idx_leftear, idx_rightear),
                session = directory,
                session_output_dir = session_output_dir,
                csv_dir = CSV_DIR,
                skip_visualization = args.skip_visualization,
                positions_bin_size = 30,
            )
