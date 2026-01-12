import logging

from marmopose.utils.constants import IP_DICT
from marmopose.utils.helpers import MultiVideoCapture

import sys
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


if __name__ == '__main__':
    """
    ********************* NOTE: Ensure that the camera clocks are manually synchronized with the server time before running this script. *********************
    ********************* 登录每个摄像头的管理界面 - 时间设置 - 同步服务器时间 - 保存 *********************
    """

    camera_paths = [
        # Example addresses for RTSP cameras
        'rtsp://admin:M4rm0s3t@192.168.15.11:554',
        'rtsp://admin:M4rm0s3t@192.168.15.12:554',
        'rtsp://admin:M4rm0s3t@192.168.15.13:554',
        'rtsp://admin:M4rm0s3t@192.168.15.14:554',
    ]

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = 'demos/realtime_record/videos_raw'
    
    os.makedirs(output_dir, exist_ok=True)


    if len(sys.argv) > 2:
        camera_name = sys.argv[2]
    else:
        camera_name = 'output'


    camera_names = [f'{camera_name}{i+1}' for i in range(len(camera_paths))]


    print(camera_names)
    print(output_dir)
    mvc = MultiVideoCapture(camera_paths, camera_names, do_cache=False, output_dir=output_dir, duration=300, keypress='esc')
    mvc.start()
    mvc.join()

    logger.info(f"Real-time video streams saved in {output_dir}")
