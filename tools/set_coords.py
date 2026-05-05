import os
import shutil
# sys.path.append('../')
import logging
import argparse
import json

from pathlib import Path

from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.calibration.calibration import Calibrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f'MarmoPose version: {marmopose_version}')

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Video directory")
parser.add_argument("-a", "--axes-order", choices=['0','1','2'],default = '0', help="Order for axes (0: (x,y,z), 1: (y,z,x), 2: (z,x,y) Default: 0)")
parser.add_argument("-o", "--offset", nargs=3, default = ('0', '0', '0'), metavar = ('offset 1st axis','offset 2nd axis','offset 3rd axis'), help="Defines the offset, needs 3 values")
parser.add_argument("-O", "--with-offset-point", action='store_true', help="Whether to add a 4th point defining the offset point (which is combined with the offset defined in the offset argument)")
parser.add_argument("-b", "--both", action='store_true', help="Whether to also create an axes file without offset for the calibration with both cages")
parser.add_argument("-v", "--video_indices", nargs='+', default=['1','2','3','4'], help="Indices of videos to use to set calibration point")
args = parser.parse_args()
VIDEO_DIR = Path(args.input)
if args.axes_order == '0':
    order = (0, 1, 2)
elif args.axes_order == '1':
    order = (1, 2, 0)
else:
    order = (2, 0, 1)

offset = tuple([float(o) for o in args.offset])
video_indices = [int(v) for v in args.video_indices]

with_both = args.both
with_offset_point = args.with_offset_point

INPUT_DIR = VIDEO_DIR / 'Input'
CALIB_DIR = VIDEO_DIR / 'Calib'
os.makedirs(CALIB_DIR, exist_ok = True)
for file in os.listdir(INPUT_DIR):
    file_path = INPUT_DIR / file
    if os.path.isfile(file_path) and file[-4:] == '.mp4':
        shutil.copy(file_path, '../demos/single/videos_raw')
        
config_path = '../configs/default.yaml'
config = Config(
	config_path=config_path,
    # Specify the project, where videos (~2min) with checkboard exist in the 'calibration' directory
    # For each project, calibration only needs to be done once
    # The camera parameters will be saved in 'camera_params.json'
    project='../demos/single'
)
calibrator = Calibrator(config)
calibrator.set_coordinates(video_inds=video_indices, obj_name='axes', offset=offset, frame_idx=200, order=order, with_offset_point = with_offset_point)
shutil.copy(f'../demos/single/calibration/axes.json', CALIB_DIR)

if with_both:
    with open(CALIB_DIR / 'axes.json') as fp:
        axes_data = json.load(fp)

    axes_data['offset'] = (0,0,0)
    if with_offset_point:
        cam_names = [key for key in axes_data.keys() if not key in ['offset','order']]
        for cam_name in cam_names:
            axes_data[cam_name] = axes_data[cam_name][:3]
            
    with open(CALIB_DIR / 'axes_both.json', 'w') as f:
        json.dump(axes_data, f, indent=4)


