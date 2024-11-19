import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, output_folder, interval=2):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_frame = 0
    pbar = tqdm(total=total_frames, desc='Extracting Frames')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % (interval * frame_rate) == 0:
            frame_name = f"frame_{int(current_frame // (interval * frame_rate)):04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)

        current_frame += 1
        pbar.update(1)

    pbar.close()
    cap.release()

if __name__ == '__main__':
    # 视频文件路径
    video_path = 'datas/v1.mp4'
    # 存储帧的文件夹路径
    output_folder = 'dataset_v1'

    extract_frames(video_path, output_folder, interval=5)
