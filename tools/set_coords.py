import sys
import os
import shutil
sys.path.append('../')
import logging
from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.calibration.calibration import Calibrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f'MarmoPose version: {marmopose_version}')

VIDEO_DIR = sys.argv[1]
INPUT_DIR = os.path.join(VIDEO_DIR,'Input')
CALIB_DIR = os.path.join(VIDEO_DIR,'Calib')
for file in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR,file)
    if os.path.isfile(file_path) and file[-4:] == '.mp4':
        shutil.copy(file_path, '../demos/single/videos_raw')

VIDEO_DIR
        
config_path = '../configs/default.yaml'
config = Config(
	config_path=config_path,
    # Specify the project, where videos (~2min) with checkboard exist in the 'calibration' directory
    # For each project, calibration only needs to be done once
    # The camera parameters will be saved in 'camera_params.json'
    project='../demos/single'
)
calibrator = Calibrator(config)
calibrator.set_coordinates(video_inds=[1,2,3, 4], obj_name='axes', offset=(0,0, 0),frame_idx=200)
shutil.copy('../demos/single/calibration/axes.json', CALIB_DIR)



