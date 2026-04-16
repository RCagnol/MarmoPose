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

if sys.argv[1] == 'home':
    room_dimensions = [1200,730, 900, 30]

elif sys.argv[1] == 'etho':
    room_dimensions = [660, 560, 800, 30]
else:
    print("First argument should be 'home' or 'etho'")
    raise ValueError

if sys.argv[2] == 'finetune':
    det_model = '../data/detection_model_finetune_' + sys.argv[1]
    pose_model = '../data/pose_model_finetune_' + sys.argv[1]
    model_output_directory = 'Output'
elif sys.argv[2] == 'base':
    det_model = '../models/detection_model'
    pose_model = '../models/pose_model'
    model_output_directory = 'Output_basemodel'
else:
    print("Second argument should be 'finetune' or 'base'")
    raise ValueError

config = Config(
    config_path=config_path,
    
    n_tracks=1,
    project='../demos/single',
    det_model= det_model,
    pose_model= pose_model,
    room_dimensions = room_dimensions,
    # do_optimize=False,
)
print(config.sub_directory)

for j in sys.argv[3:]:
    if sys.argv[1] == 'home':
        test_dirs = [f'TestHomeWithEtho{j}.{i}' for i in range(5)]
    
    elif sys.argv[1] == 'etho':
        test_dirs = [f'TestEthoWithHome{j}.{i}' for i in range(5)]
    # test_dirs = ['TestHome']

    for test_dir in test_dirs:
        logger.info(test_dir)
        DIR = os.path.join("/srv/MarmOT/VideoTracking/Videos/", test_dir)
        DIST_DIR = os.path.join("/scratch/VideoTracking/Videos/", test_dir)

        if sys.argv[1] == 'home':
            CALIB_DIR = os.path.join("/srv/MarmOT/VideoTracking/Videos/", test_dir, 'Calib_preprocessed')    
        elif sys.argv[1] == 'etho':
            CALIB_DIR = "/srv/MarmOT/VideoTracking/Videos/CalibEtho"


        model_output_path = os.path.join(DIST_DIR,model_output_directory)

        if not os.path.exists(DIR) or not os.path.isdir(DIR):
            continue
        if os.path.exists(os.path.join(DIR, 'Input_preprocessed')):
            input_path = os.path.join(DIR, 'Input_preprocessed')
        elif os.path.exists(os.path.join(DIR, 'input_preprocessed')):
            input_path = os.path.join(DIR, 'input_preprocessed')
        else:
            continue

        for video in os.listdir(config.sub_directory['videos_raw']):
            try:
                os.remove(os.path.join(config.sub_directory['videos_raw'],video))
            except OSError:
                pass

        for video in os.listdir(input_path):
            if video[-4:] == '.mp4':
                shutil.copy(os.path.join(input_path,video), config.sub_directory['videos_raw'])
        os.makedirs(model_output_path, exist_ok=True)

        shutil.copy(os.path.join(CALIB_DIR,'camera_params.json'), config.sub_directory['calibration'])
        shutil.copytree(CALIB_DIR, os.path.join(DIST_DIR, 'Calib'), dirs_exist_ok=True)

        predictor = Predictor(config, batch_size=4)
        predictor.predict()
        shutil.copytree(config.sub_directory['points_2d'], model_output_path + '/points_2d', dirs_exist_ok=True)
    
        reconstructor_3d = Reconstructor3D(config)
        reconstructor_3d.triangulate(file_names = predictor.file_names)
        shutil.copytree(config.sub_directory['points_3d'], model_output_path + '/points_3d', dirs_exist_ok=True)

        visualizer_2d = Visualizer2D(config)
        visualizer_2d.generate_videos_2d(file_names = predictor.file_names)
        shutil.copytree(config.sub_directory['videos_labeled_2d'], model_output_path + '/videos_labeled_2d', dirs_exist_ok=True)

        visualizer_3d = Visualizer3D(config)
        # visualizer_3d.generate_video_3d(source_3d='optimized', video_type='composite', file_names_2d = [f'output{i}' for i in [1,3]])
        visualizer_3d.generate_video_3d(source_3d='optimized', video_type='composite', file_names_2d = predictor.file_names)
        shutil.copytree(config.sub_directory['videos_labeled_3d'], model_output_path + '/videos_labeled_3d', dirs_exist_ok=True)
