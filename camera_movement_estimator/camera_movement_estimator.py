import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:

    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:20, :]      = 255
        mask_features[900:1050, :] = 255

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
            if len(camera_movement) == len(frames):
                return camera_movement
            else:
                print(f"[CameraMovement] Stub frame count ({len(camera_movement)}) != "
                      f"video frame count ({len(frames)}). Re-running estimation...")

        camera_movement = [[0, 0]] * len(frames)

        old_gray     = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance  = 0
            camera_move_x = 0
            camera_move_y = 0

            for new, old in zip(new_features, old_features):
                new_pt = new.ravel()
                old_pt = old.ravel()

                distance = measure_distance(new_pt, old_pt)
                if distance > max_distance:
                    max_distance = distance
                    camera_move_x, camera_move_y = measure_xy_distance(old_pt, new_pt)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_move_x, camera_move_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position')
                    if position is None:
                        continue
                    movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - movement[0],
                        position[1] - movement[1],
                    )
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            x_move, y_move = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera X: {x_move:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Y: {y_move:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)
        return output_frames