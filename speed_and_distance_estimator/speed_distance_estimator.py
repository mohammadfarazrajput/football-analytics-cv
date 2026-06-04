# Step 9: Speed and Distance Estimator
# File: speed_and_distance_estimator/speed_and_distance_estimator.py
#
# HOW IT WORKS:
#   Now that every player has a 'position_transformed' in METERS (from step 8)
#   and camera movement has been subtracted (step 7), we can measure real speed.
#
#   We use a SLIDING WINDOW approach:
#     - Every 5 frames ("frame_window"), compute:
#         distance = Euclidean distance between position at frame N and frame N+5
#         time     = 5 / 24 fps  =  0.208 seconds
#         speed    = distance / time  (m/s)  →  × 3.6  →  km/h
#     - Assign this speed to all 5 frames in the window (so it looks smooth)
#     - Accumulate total distance as player moves window by window
#
#   Only players inside the trapezoid (position_transformed is not None) are measured.
#   Ball and referees are skipped.

import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator:

    def __init__(self):
        self.frame_window = 5    # compute speed every N frames
        self.frame_rate   = 24   # video fps

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Adds 'speed' (km/h) and 'distance' (total meters) to each player track entry.
        Modifies tracks in-place.
        """
        total_distance = {}  # {object_type: {track_id: cumulative_meters}}

        for object_type, object_tracks in tracks.items():
            if object_type in ('ball', 'referees'):
                continue   # only measure players

            num_frames = len(object_tracks)
            total_distance[object_type] = {}

            # Step through frames in windows of frame_window
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)

                for track_id in object_tracks[frame_num]:
                    # Player must exist in both the first AND last frame of window
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = object_tracks[frame_num][track_id].get('position_transformed')
                    end_pos   = object_tracks[last_frame][track_id].get('position_transformed')

                    # Skip if outside trapezoid (None means outside view)
                    if start_pos is None or end_pos is None:
                        continue

                    distance_covered = measure_distance(start_pos, end_pos)  # meters
                    time_elapsed     = (last_frame - frame_num) / self.frame_rate  # seconds
                    speed_ms         = distance_covered / time_elapsed             # m/s
                    speed_kmh        = speed_ms * 3.6                              # km/h

                    # Accumulate total distance
                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0
                    total_distance[object_type][track_id] += distance_covered

                    # Assign speed + cumulative distance to every frame in this window
                    for fn in range(frame_num, last_frame):
                        if track_id not in object_tracks[fn]:
                            continue
                        tracks[object_type][fn][track_id]['speed']    = speed_kmh
                        tracks[object_type][fn][track_id]['distance'] = total_distance[object_type][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """
        Overlay speed (km/h) and distance (m) below each player's bounding box.
        Only drawn when the player has speed/distance data (inside trapezoid).
        Modifies frames in-place.
        """
        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type in ('ball', 'referees'):
                    continue

                for track_id, track_info in object_tracks[frame_num].items():
                    speed    = track_info.get('speed')
                    distance = track_info.get('distance')

                    if speed is None or distance is None:
                        continue

                    bbox     = track_info['bbox']
                    position = get_foot_position(bbox)

                    # Draw 40px below the foot position to avoid overlapping the ellipse
                    position = (position[0], position[1] + 40)
                    position = tuple(map(int, position))

                    cv2.putText(frame, f"{speed:.2f} km/h",
                                position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 2)

                    cv2.putText(frame, f"{distance:.2f} m",
                                (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 2)
