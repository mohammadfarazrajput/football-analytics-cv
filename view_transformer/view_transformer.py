import cv2
import numpy as np
import sys
sys.path.append('../')


class ViewTransformer:

    def __init__(self):
        # Real-world dimensions of the selected pitch region (metres)
        court_width  = 68       # metres
        court_length = 23.32    # metres

        # Pixel vertices of the trapezoid (manually calibrated for the sample video)
        # Order: top-left, top-right, bottom-right, bottom-left
        self.pixel_vertices = np.array([
            [110,  1035],
            [265,   275],
            [910,   260],
            [1640,  915],
        ], dtype=np.float32)

        # Target rectangle vertices in real-world metres
        self.target_vertices = np.array([
            [0,            court_width],
            [0,            0],
            [court_length, 0],
            [court_length, court_width],
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices,
        )

    def transform_point(self, point):
        """
        Transform a single pixel (x, y) to real-world coordinates (metres).
        Returns None if the point is outside the trapezoid.
        """
        p = (int(point[0]), int(point[1]))

        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped    = np.array([[p]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.perspective_transformer)

        return transformed.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        For every track entry with 'position_adjusted', compute real-world
        metre position and store as 'position_transformed'.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted')
                    if position is None:
                        tracks[object_type][frame_num][track_id]['position_transformed'] = None
                        continue

                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
