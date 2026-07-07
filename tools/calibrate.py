import logging
import tempfile

from marmopose.version import __version__ as marmopose_version
from marmopose.config import Config
from marmopose.calibration.calibration import Calibrator
from marmopose.utils.data_io import load_axes

import sys
import os
import shutil
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f'MarmoPose version: {marmopose_version}')

config_path = '../configs/default.yaml'
# Kept on /scratch (local disk) rather than the OS default temp dir: calibration
# videos are large and this must sit on the same fast local filesystem as the
# rest of the project, not a possibly small/RAM-backed /tmp.
SCRATCH_ROOT = Path('../demos/.calibration_scratch')

parser = argparse.ArgumentParser()
parser.add_argument("session_dir", help="Session directory, containing 'Etho' and/or 'Home' subdirectories with their own 'Calib' folder")
parser.add_argument("-c", "--cage", choices=["etho", "home", "both"], required=True, help="Cage, must be etho, home, or both")
parser.add_argument("-u", "--update-axes", action="store_true", help="Only update axes if present")
parser.add_argument("-s", "--suffix", help="Suffix for axes and camera_params files. Pass 'all' to process every suffix found among axes*.json files in the cage's Calib directory")
parser.add_argument("-b", "--board-type", choices=["checkerboard", "charuco"], default="charuco", help="Calibration board type, must be checkerboard or charuco")
args = parser.parse_args()

SESSION_DIR = Path(args.session_dir)
cage_arg = args.cage
update_axes_only = args.update_axes
board_type = args.board_type

board_square_side_length = 20 if board_type == 'checkerboard' else 16
path_calib_etho = Path('/srv/MarmOT/VideoTracking/Videos/TestCalibCharuco3' if board_type == 'charuco' else '/srv/MarmOT/VideoTracking/Videos/CalibEtho')

SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)


def discover_suffixes(calib_dir):
    """Suffixes (e.g. '', '_both') found from axes*.json files present in calib_dir, sorted so the unsuffixed 'axes.json' comes first."""
    suffixes = [axes_file.stem[len('axes'):] for axes_file in sorted(calib_dir.glob('axes*.json'))]
    assert suffixes, f'No axes*.json files found in {calib_dir}'
    return suffixes


def calibrate_one(cage, calib_dir, suffix, update_axes_only):
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as project_dir:
        config = Config(
            config_path=config_path,
            project=project_dir,
            board_type=board_type,
            board_square_side_length=board_square_side_length
        )

        config_calib_path = Path(config.sub_directory['calibration'])
        config_calib_path.mkdir(exist_ok=True)

        videos_path = path_calib_etho if cage == 'etho' else calib_dir

        for file in os.listdir(videos_path):
            file_path = videos_path / file
            if os.path.isfile(file_path) and file[-4:] == '.mp4':
                shutil.copy(file_path, config_calib_path)

        if cage == 'etho':
            assert os.path.exists(calib_dir / f'axes{suffix}.json'), f'File axes{suffix}.json is not present in {calib_dir}'
            shutil.copy(calib_dir / f'axes{suffix}.json', config_calib_path / 'axes.json')
            assert os.path.exists(path_calib_etho / 'detected_boards.pickle'), f'File detected_boards.pickle is not present in {path_calib_etho}'
            shutil.copy(path_calib_etho / 'detected_boards.pickle', config_calib_path)
        elif update_axes_only:
            assert os.path.exists(calib_dir / f'axes{suffix}.json'), f'File axes{suffix}.json is not present in {calib_dir}'
            shutil.copy(calib_dir / f'axes{suffix}.json', config_calib_path / 'axes.json')
            assert os.path.exists(calib_dir / 'detected_boards.pickle'), f'File detected_boards.pickle is not present in {calib_dir}'
            shutil.copy(calib_dir / 'detected_boards.pickle', config_calib_path)
        else:
            assert os.path.exists(calib_dir / f'axes{suffix}.json'), f'File axes{suffix}.json is not present in {calib_dir}'
            shutil.copy(calib_dir / f'axes{suffix}.json', config_calib_path / 'axes.json')

        calibrator = Calibrator(config)
        calibrator.calibrate()

        shutil.copy(config_calib_path / 'camera_params.json', calib_dir / f'camera_params{suffix}.json')
        shutil.copy(config_calib_path / 'detected_boards.pickle', calib_dir)


for cage in (['etho', 'home'] if cage_arg == 'both' else [cage_arg]):
    calib_dir = SESSION_DIR / cage.capitalize() / 'Calib' if cage == 'etho' else SESSION_DIR / cage.capitalize() / 'Calib_preprocessed'
    assert calib_dir.is_dir(), f'Missing calibration directory: {calib_dir}'

    if args.suffix == 'all':
        suffixes = discover_suffixes(calib_dir)
    elif args.suffix is not None:
        suffixes = [f'_{args.suffix}']
    else:
        suffixes = ['']

    for i, suffix in enumerate(suffixes):
        # Board detection (the expensive step) doesn't depend on the suffix, only
        # the axes/extrinsics do. So beyond the first suffix, reuse the
        # detected_boards.pickle just produced instead of redetecting from
        # scratch for every suffix. 'etho' is unaffected: it always reuses the
        # shared rig's cached boards regardless of suffix.
        effective_update_axes_only = update_axes_only or i > 0
        calibrate_one(cage, calib_dir, suffix, effective_update_axes_only)

    if cage == 'etho':
        # Etho's Calib holds no raw videos to preprocess (those live in the shared
        # rig path above), so Calib_preprocessed should just be an alias for Calib.
        # A symlink (rather than a copy) makes it impossible for the two to drift
        # apart, and costs nothing to (re)create regardless of directory size.
        calib_preprocessed_dir = calib_dir.parent / 'Calib_preprocessed'
        if calib_preprocessed_dir.is_symlink():
            calib_preprocessed_dir.unlink()
        # If it exists as a real (non-symlink) directory, symlink_to below raises
        # FileExistsError rather than silently overwriting it: a leftover real
        # directory means Calib_preprocessed may hold data Calib doesn't (as it
        # did for one session that needed manual reconciliation), so it must be
        # resolved by hand, not deleted automatically.
        calib_preprocessed_dir.symlink_to(calib_dir.name, target_is_directory=True)
