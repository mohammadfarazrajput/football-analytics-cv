import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator:

    def __init__(self):
        self.frame_window = 5   # compute speed every N frames
        self.frame_rate   = 24  # video fps

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Adds 'speed' (km/h) and 'distance' (total metres) to each player track entry.
        Modifies tracks in-place.
        """
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            if object_type in ('ball', 'referees'):
                continue

            num_frames = len(object_tracks)
            total_distance[object_type] = {}

            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = object_tracks[frame_num][track_id].get('position_transformed')
                    end_pos   = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_pos is None or end_pos is None:
                        continue

                    distance_covered = measure_distance(start_pos, end_pos)
                    time_elapsed     = (last_frame - frame_num) / self.frame_rate
                    speed_ms         = distance_covered / time_elapsed
                    speed_kmh        = speed_ms * 3.6

                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0
                    total_distance[object_type][track_id] += distance_covered

                    for fn in range(frame_num, last_frame):
                        if track_id not in object_tracks[fn]:
                            continue
                        tracks[object_type][fn][track_id]['speed']    = speed_kmh
                        tracks[object_type][fn][track_id]['distance'] = total_distance[object_type][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """Overlay speed (km/h) and distance (m) below each player's bounding box."""
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
