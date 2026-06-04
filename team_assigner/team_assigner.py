import numpy as np
import cv2
from sklearn.cluster import KMeans


class TeamAssigner:

    def __init__(self):
        self.team_colors = {}       # {1: color_array, 2: color_array}
        self.player_team_dict = {}  # {player_id: team_id} cache

    def get_clustering_model(self, image):
        """Cluster image pixels into 2 groups using KMeans."""
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        """Extract dominant t-shirt color from one player crop."""
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[:image.shape[0] // 2, :]

        kmeans = self.get_clustering_model(top_half)
        labels = kmeans.labels_
        clustered = labels.reshape(top_half.shape[:2])

        corners = [
            clustered[0, 0], clustered[0, -1],
            clustered[-1, 0], clustered[-1, -1],
        ]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        """
        Call once at start. Clusters all player t-shirt colors into 2 teams.
        player_detections: dict of {track_id: {'bbox': ...}} from frame 0.
        """
        player_colors = []
        for _, player_info in player_detections.items():
            color = self.get_player_color(frame, player_info['bbox'])
            player_colors.append(color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1],
        }

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Returns team ID (1 or 2) for a given player. Caches result per player_id.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # convert 0/1 → 1/2

        # Hardcode goalkeeper correction if needed (adjust player_id as necessary)
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
