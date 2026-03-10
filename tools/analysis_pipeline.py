import logging
from importlib import reload
import shutil
import os

from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.processing.prediction import Predictor
from marmopose.visualization.display_2d import Visualizer2D
from marmopose.visualization.display_3d import Visualizer3D
from marmopose.processing.triangulation import Reconstructor3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f'MarmoPose version: {marmopose_version}')

config_path = '../configs/default.yaml'

config = Config(
    config_path=config_path,
    
    n_tracks=1,
    project='../demos/single',
    det_model= '../data/detection_model_family_finetune',
    pose_model= '../data/pose_model_finetune',
)
print(config.sub_directory)

test_dirs = [f'Test3.{i}' for i in range(26)]

for test_dir in test_dirs:
    DIR = "/srv/MarmOT/VideoTracking/Videos/" + test_dir
    DIST_DIR = "/scratch/VideoTracking/Videos"
    print(test_dir)
    if not os.path.exists(DIR) or not os.path.isdir(DIR):
        continue

    for i in range(4):
        shutil.copy(DIR +f'/Input/output{i+1}.mp4', config.sub_directory['videos_raw'])

    os.makedirs(DIST_DIR + '/Output', exist_ok=True)
    shutil.copytree(config.sub_directory['calibration'], DIST_DIR + '/Calib', dirs_exist_ok=True)

    predictor = Predictor(config, batch_size=4)
    predictor.predict()
    shutil.copytree(config.sub_directory['points_2d'], DIST_DIR + '/Output/points_2d', dirs_exist_ok=True)

    reconstructor_3d = Reconstructor3D(config)
    reconstructor_3d.triangulate(file_names = predictor.file_names)
    shutil.copytree(config.sub_directory['points_3d'], DIST_DIR + '/Output/points_3d', dirs_exist_ok=True)

    visualizer_2d = Visualizer2D(config)
    visualizer_2d.generate_videos_2d(file_names = predictor.file_names)
    shutil.copytree(config.sub_directory['videos_labeled_2d'], DIST_DIR + '/Output/videos_labeled_2d', dirs_exist_ok=True)

    visualizer_3d = Visualizer3D(config)
    visualizer_3d.generate_video_3d(source_3d='optimized', video_type='composite', file_names_2d = predictor.file_names)
    shutil.copytree(config.sub_directory['videos_labeled_3d'], DIST_DIR + '/Output/videos_labeled_3d', dirs_exist_ok=True)
