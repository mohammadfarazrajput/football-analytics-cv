# Step 7: Camera Movement Estimator
# File: camera_movement_estimator/camera_movement_estimator.py
#
# WHY WE NEED THIS:
#   The broadcast camera pans and zooms constantly.
#   If a player stands still but the camera pans right, their bounding box
#   moves left in pixel space. Without compensation, we'd think the player
#   moved when they didn't — this would corrupt speed/distance measurements.
#
# HOW LUCAS-KANADE OPTICAL FLOW WORKS:
#   1. Pick "feature points" in frame N — corners on static objects (grass lines,
#      ad boards). We choose features ONLY from top and bottom strips of the frame,
#      avoiding the middle where players are moving.
#   2. For each feature point in frame N, track where it ended up in frame N+1.
#   3. The movement of static features IS the camera movement.
#   4. We take the maximum displacement among all tracked features as the camera
#      movement for that frame pair.
#   5. If max displacement < 5 pixels → ignore (noise, camera didn't really move).
#
# OUTPUT: camera_movement_per_frame — list of [dx, dy] per frame

import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:

    def __init__(self, frame):
        self.minimum_distance = 5   # ignore movements smaller than 5 px (noise)

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),        # search window per feature point
            maxLevel=2,              # pyramid levels (downscale image 2x for large motion)
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,     # max iterations
                0.03,   # stop if movement < 0.03 px
            ),
        )

        # Feature detection parameters (Shi-Tomasi corner detection)
        # We only look for features in top 20 rows and rows 900-1050
        # (ad boards / pitch lines) — static regions of the frame
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:20, :]    = 255   # top strip
        mask_features[900:1050, :] = 255  # bottom strip

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Returns: camera_movement_per_frame — list of [dx, dy] per frame.
        Frame 0 is always [0, 0] (reference frame).
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize: all frames start with zero movement
        camera_movement = [[0, 0]] * len(frames)

        # Compute grayscale of first frame and detect initial features
        old_gray    = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Track old features into new frame using Lucas-Kanade
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            # Find the maximum displacement among all tracked features
            max_distance   = 0
            camera_move_x  = 0
            camera_move_y  = 0

            for new, old in zip(new_features, old_features):
                new_pt = new.ravel()
                old_pt = old.ravel()

                distance = measure_distance(new_pt, old_pt)
                if distance > max_distance:
                    max_distance = distance
                    camera_move_x, camera_move_y = measure_xy_distance(old_pt, new_pt)

            # Only record if movement is statistically significant (> 5 px)
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_move_x, camera_move_y]
                # Re-detect features on the new frame for next iteration
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        # Cache to stub
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        For every track entry that has a 'position', subtract the camera movement
        to get the 'position_adjusted' — the player's real movement, not camera drift.
        """
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
        """Overlay camera movement dx/dy values on each frame for debugging."""
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
