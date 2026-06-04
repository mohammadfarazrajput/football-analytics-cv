# Step 6: Player Ball Assigner
# File: player_ball_assigner/player_ball_assigner.py
#
# HOW IT WORKS:
#   For each frame:
#     1. Get the center of the ball bounding box
#     2. For every player, measure the distance from both their feet to the ball
#        (left foot = bbox bottom-left, right foot = bbox bottom-right)
#     3. Take the minimum foot-to-ball distance for each player
#     4. The player with the smallest distance gets assigned the ball
#        — BUT only if that distance < max_player_ball_distance (70 pixels)
#        — If no player is close enough → no one has the ball (-1)
#   5. In main.py: add 'has_ball': True to that player's track entry

import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:

    def __init__(self):
        # If the nearest player is farther than this, no one "has" the ball
        self.max_player_ball_distance = 70  # pixels

    def assign_ball_to_player(self, players, ball_bbox):
        """
        players:   dict of {track_id: {'bbox': ...}}   for one frame
        ball_bbox: [x1, y1, x2, y2]

        Returns: track_id of the player closest to the ball,
                 or -1 if no player is within max_player_ball_distance.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance    = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            bbox = player['bbox']

            # Left foot: (x1, y2)
            dist_left  = measure_distance((bbox[0], bbox[3]), ball_position)
            # Right foot: (x2, y2)
            dist_right = measure_distance((bbox[2], bbox[3]), ball_position)

            distance = min(dist_left, dist_right)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance    = distance
                    assigned_player = player_id

        return assigned_player
