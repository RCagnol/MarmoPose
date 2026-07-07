import sys
import os
import shutil
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import cv2
import av
import skvideo.io
import matplotlib.pyplot as plt
from multiprocessing import Process

SRC_DIR = sys.argv[1]

dir_name = sys.argv[2]

PREPROCESSED_DIR = os.path.join(SRC_DIR, f'{dir_name}_preprocessed')
DIR = os.path.join(SRC_DIR, dir_name)

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

txt_files = [file for file in os.listdir(DIR) if file[-4:] == '.txt']
camera_names = [file[:-4] for file in txt_files]
txt_paths = [os.path.join(DIR, txt_file) for txt_file in txt_files]
modes = ['r' for _ in camera_names]
json_files = [file for file in os.listdir(DIR) if file[-5:] == '.json']

for json_file in json_files:
    shutil.copy(Path(DIR) / json_file, PREPROCESSED_DIR)

# 2D Array storing the frame indices from the original videos, placed here according to their index in the reordered video
# Nan values indicate that the frame at this index should be a black frame
reordered_frames = np.empty((len(camera_names),0))
# Number of read timestamp for each video
cnt_read = [0 for _ in camera_names]
# Boolean list indicating whether the last timestamp was reached for each video
finished = [False for _ in camera_names]

# To ensure all .txt files are closed properly
with ExitStack() as stack:
    files = [stack.enter_context(open(fname, mode)) for fname, mode in zip(txt_paths, modes)]


    # Read the first line for each .txt file
    streams_open = np.array([file.readline().strip().split(' ')[2] for file in files]).astype(float)
    streams_start = np.array([file.readline().strip().split(' ')[2] for file in files]).astype(float)
    frames_pts = np.array([file.readline().strip().split(' ')[1] for file in files]).astype(float)
    jitter_open = (streams_open - np.min(streams_open)) * 90000 # 90000 to convert to pts
    frames_pts += jitter_open
    current_frame = 0
    # Get first PTS
    first_pts = np.min(frames_pts)
    print(first_pts)

    # While last time stamp isn't reached for each video
    while np.sum(finished) < len(camera_names):
        # New frame indices column, initialized to NaN
        reordered_frames = np.concatenate((reordered_frames,np.full((len(camera_names),1),np.nan)),axis = 1)

        # Does next timestamp correspond to either next frame or a previous one? 
        # False if it corresponds to a frame after the next expected frame
        # 3600 correpsonds to an interframe period (in PTS units)
        idx_contain_frame = np.nonzero((frames_pts - first_pts + 1800)//3600 <= current_frame)[0]

        # For each video for which next timestamp is either next expected frame or a previous one
        for idx in idx_contain_frame:
            # Skip if the last timestamp was reached and it's being re-read
            if finished[idx]:
                continue

            # Compute the true frame index corresponding to this timestamp and store its current index in the frame array
            frame_nb = int((frames_pts[idx] - first_pts + 1800)//3600)
            reordered_frames[idx,frame_nb] = cnt_read[idx]
            # One more timestamp has been read
            cnt_read[idx] += 1

            # Read newline in .txt, if it's the last line then update the finished list
            newline = files[idx].readline()
            if newline:
                frames_pts[idx] = int(newline.strip().split(' ')[1]) + jitter_open[idx]
            else:
                frames_pts[idx] = np.nan
                finished[idx] = True

        current_frame += 1

nans = np.nonzero(np.isnan(reordered_frames))
nan_sums = np.sum(np.isnan(reordered_frames),axis=0)
all_nans = np.nonzero(nan_sums == len(camera_names))[0]
if len(all_nans) > 0:
    nonconsecutive_nans = np.nonzero(all_nans[1:] - all_nans[:-1] > 1)[0]
    if len(nonconsecutive_nans) > 0:
        first_nan_endsection = all_nans[nonconsecutive_nans[-1]]
    else:
        first_nan_endsection = all_nans[0]
    reordered_frames_stripped = reordered_frames[:, :first_nan_endsection]
else:
    reordered_frames_stripped = reordered_frames


for i, camera_name in enumerate(camera_names):
    print(f'Null frames for {camera_name}: {np.nonzero(np.isnan(reordered_frames_stripped[i,:]))[0]}')
print(reordered_frames_stripped.shape)

def worker(camera_idx):
    camera_name = camera_names[camera_idx]
    input_container = av.open(os.path.join(DIR,camera_name + '.mp4'), mode='r')
    input_stream = input_container.streams.video[0]
    width = input_stream.width
    height = input_stream.height
    fps = input_stream.base_rate
    time_base = input_stream.time_base
    start_time = input_stream.start_time
    frames = input_container.decode(input_stream)
    black_frame = np.zeros((height,width,3))

    writer = skvideo.io.FFmpegWriter(os.path.join(PREPROCESSED_DIR,camera_name + '.mp4'), inputdict={'-framerate': str(fps)},
                                         outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-preset': 'superfast', '-crf': '23'})

    prev_frame = -1
    prev_lost = False
    for i,idx_frame in enumerate(reordered_frames_stripped[camera_idx,:]):
        if np.isnan(idx_frame):
            frame = black_frame
            prev_lost = True
        else:
            if idx_frame - prev_frame != 1:
                print(f'Jump {idx_frame - prev_frame} {camera_name} {idx_frame} {i}')
                target_timestamp = int(idx_frame / fps / time_base) + start_time
                input_container.seek(target_timestamp, stream=input_stream)
                frames = input_container.decode(input_stream)
            try:
                frame = next(frames)
                while frame.pts * time_base * fps < idx_frame:
                    frame = next(frames)

                if frame.key_frame or not prev_lost:
                    frame = frame.to_ndarray(format='rgb24')
                    prev_lost = False
                else:
                    frame = black_frame
                prev_frame = idx_frame

            except StopIteration:
                print(f"Couldn't read frame {idx_frame}")
                frame = black_frame
                break  # No more frames

        writer.writeFrame(frame)
    input_container.close()
    writer.close()

processes = []
for i in range(len(camera_names)):
    p = Process(target=worker, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()
