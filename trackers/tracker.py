import cv2
import numpy as np
import pickle
import os
import sys
import supervision as sv
from ultralytics import YOLO

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """Run YOLO on all frames in batches. Returns raw YOLO results."""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch, conf=0.1)
            detections += batch_detections
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            if len(tracks['players']) == len(frames):
                return tracks
            else:
                print(f"[Tracker] Stub frame count ({len(tracks['players'])}) != "
                      f"video frame count ({len(frames)}). Re-running detection...")

        detections = self.detect_frames(frames)

        tracks = {
            'players':  [],
            'referees': [],
            'ball':     [],
        }

        for frame_num, detection in enumerate(detections):
            class_names     = detection.names
            class_names_inv = {v: k for k, v in class_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            for obj_idx, class_id in enumerate(detection_sv.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_sv.class_id[obj_idx] = class_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox     = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_names[class_id] == 'player':
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                if class_names[class_id] == 'referee':
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_sv:
                bbox     = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_names[class_id] == 'ball':
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def interpolate_ball_positions(self, ball_positions):
        import pandas as pd

        ball_positions_list = [
            x.get(1, {}).get('bbox', [])
            for x in ball_positions
        ]

        df = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()

        ball_positions = [
            {1: {'bbox': x}}
            for x in df.to_numpy().tolist()
        ]
        return ball_positions

    def add_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            x2_rect = x_center + rect_w // 2
            y1_rect = y2 - rect_h // 2 + 15
            y2_rect = y2 + rect_h // 2 + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x_center,      y],
            [x_center - 10, y - 20],
            [x_center + 10, y - 20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color,     cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = np.array(team_ball_control[:frame_num + 1])
        team1_frames = (team_ball_control_till_frame == 1).sum()
        team2_frames = (team_ball_control_till_frame == 2).sum()
        total = team1_frames + team2_frames

        team1_pct = team1_frames / total if total > 0 else 0
        team2_pct = team2_frames / total if total > 0 else 0

        cv2.putText(frame, f"Team 1 Ball Control: {team1_pct * 100:.2f}%",
                    (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team2_pct * 100:.2f}%",
                    (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict  = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict    = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames