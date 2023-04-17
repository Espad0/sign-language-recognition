import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    video_df = []
    frame_n = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height,width,_ = image.shape
            y_coef = height/width
            result = holistic.process(image)

            data = []

            for landmark_index in range(543):
                if landmark_index < 468:
                    landmark_type = 'face'
                    if result.face_landmarks is not None:
                        points = result.face_landmarks
                    else:
                        points = np.nan
                    offset = 0
                elif landmark_index < 489:
                    landmark_type = 'left_hand'
                    if result.left_hand_landmarks is not None:
                        points = result.left_hand_landmarks
                    else:
                        points = np.nan
                    offset = 468
                elif landmark_index < 522:
                    landmark_type = 'pose'
                    if result.pose_landmarks is not None:
                        points = result.pose_landmarks
                    else:
                        np.nan
                    offset = 489
                else:
                    landmark_type = 'right_hand'
                    if result.right_hand_landmarks is not None:
                        points = result.right_hand_landmarks
                    else:
                        points = np.nan
                    offset = 522

                data.append({
                    'row_id': str(frame_n)+'-'+landmark_type+'-'+str(landmark_index-offset),
                    'type': landmark_type,
                    'landmark_index': landmark_index-offset,
                    'x': points.landmark[landmark_index-offset].x if points is not np.nan else np.nan,
                    'y': points.landmark[landmark_index-offset].y * y_coef if points is not np.nan else np.nan,
                    'z': points.landmark[landmark_index-offset].z if points is not np.nan else np.nan
                })

            frame_df = pd.DataFrame(data)
            frame_df.loc[:,'frame'] =  frame_n
            video_df.append(frame_df)

            frame_n += 1


        cap.release()
    video_df = pd.concat(video_df)
    video_df = video_df[['frame','row_id','type','landmark_index','x','y','z']]
    return video_df

def process_folder(folder_path, output_folder):
    video_names = os.listdir(folder_path)
    video_names = [name for name in video_names if not '.DS' in name]
    os.makedirs(output_folder, exist_ok=True)
    folder_len = len(video_names)
    
    for i, vname in enumerate(video_names):
        video_df = process_video(os.path.join(folder_path, vname))
        video_df = video_df.reset_index(drop=True)
        video_df['x'] = video_df['x'].astype(np.float32)
        video_df['y'] = video_df['y'].astype(np.float32)
        video_df['z'] = video_df['z'].astype(np.float32)
        video_df.to_parquet(os.path.join(output_folder, vname.replace('.mp4','.parquet')), index=False)
        print('Done',str(i+1)+'/'+str(folder_len))

    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Preparation of markup for videos")
    parser.add_argument('--video_folder', type=str, help='The path of the directory with the videos')
    parser.add_argument('--output_folder', type=str, default='./output_dataframes', help='The path to the directory for saved results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_folder(args.video_folder, args.output_folder)