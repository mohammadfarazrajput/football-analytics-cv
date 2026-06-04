import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:

    def __init__(self):
        # If the nearest player is farther than this, no one "has" the ball
        self.max_player_ball_distance = 70  # pixels

    def assign_ball_to_player(self, players, ball_bbox):
        """
        players:   dict of {track_id: {'bbox': ...}} for one frame
        ball_bbox: [x1, y1, x2, y2]

        Returns track_id of the closest player, or -1 if none within threshold.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance    = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            bbox = player['bbox']

            dist_left  = measure_distance((bbox[0], bbox[3]), ball_position)
            dist_right = measure_distance((bbox[2], bbox[3]), ball_position)
            distance   = min(dist_left, dist_right)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance    = distance
                    assigned_player = player_id

        return assigned_player
