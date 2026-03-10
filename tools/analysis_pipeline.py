import logging
from importlib import reload
import shutil
import os
import sys
from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.processing.prediction import Predictor
from marmopose.visualization.display_2d import Visualizer2D
from marmopose.visualization import display_3d
from marmopose.visualization.display_3d import Visualizer3D
from marmopose.processing.triangulation import Reconstructor3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f'MarmoPose version: {marmopose_version}')

config_path = '../configs/default.yaml'

if len(sys.argv) > 1 and sys.argv[1] == 'home':
    room_dimensions = [1200,730, 900, 30]
else:
    room_dimensions = [660, 560, 800, 30]
config = Config(
    config_path=config_path,
    
    n_tracks=1,
    project='../demos/single',
    # det_model= '../data/detection_model_finetune',
    # pose_model= '../data/pose_model_finetune',
    det_model= '../models/detection_model',
    pose_model= '../models/pose_model',
    room_dimensions = room_dimensions
)
print(config.sub_directory)

test_dirs = [f'TestHome7.{i}' for i in range(5)]

for test_dir in test_dirs:
    DIR = os.path.join("/srv/MarmOT/VideoTracking/Videos/", test_dir)
    DIST_DIR = os.path.join("/scratch/VideoTracking/Videos/", test_dir)
    print(test_dir)
    if not os.path.exists(DIR) or not os.path.isdir(DIR):
        continue
    if os.path.exists(DIR +'/Input'):
        input_path = 'Input'
    elif os.path.exists(DIR +'/input'):
        input_path = 'input'
    else:
        continue

    for i in range(4):
        shutil.copy(DIR +f'/{input_path}/output{i+1}.mp4', config.sub_directory['videos_raw'])
    print(config.sub_directory['calibration'])
    os.makedirs(DIST_DIR + '/Output_basemodel', exist_ok=True)
    shutil.copytree(config.sub_directory['calibration'], DIST_DIR + '/Calib', dirs_exist_ok=True)
    print(display_3d.__file__)

    predictor = Predictor(config, batch_size=4)
    predictor.predict()
    shutil.copytree(config.sub_directory['points_2d'], DIST_DIR + '/Output_basemodel/points_2d', dirs_exist_ok=True)
  
    reconstructor_3d = Reconstructor3D(config)
    reconstructor_3d.triangulate(file_names = predictor.file_names)
    shutil.copytree(config.sub_directory['points_3d'], DIST_DIR + '/Output_basemodel/points_3d', dirs_exist_ok=True)

    visualizer_2d = Visualizer2D(config)
    visualizer_2d.generate_videos_2d(file_names = predictor.file_names)
    shutil.copytree(config.sub_directory['videos_labeled_2d'], DIST_DIR + '/Output_basemodel/videos_labeled_2d', dirs_exist_ok=True)

    visualizer_3d = Visualizer3D(config)
    visualizer_3d.generate_video_3d(source_3d='optimized', video_type='composite', file_names_2d = predictor.file_names)
    shutil.copytree(config.sub_directory['videos_labeled_3d'], DIST_DIR + '/Output_basemodel/videos_labeled_3d', dirs_exist_ok=True)
