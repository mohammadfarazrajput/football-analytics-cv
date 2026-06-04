def get_center_of_bbox(bbox):
    """Return (cx, cy) — the center pixel of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """Return the pixel width of a bounding box."""
    return bbox[2] - bbox[0]  # x2 - x1


def get_foot_position(bbox):
    """
    Return the foot position of a player: center-x, bottom-y.
    Used as the 'ground position' for speed/distance calculations.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def measure_xy_distance(p1, p2):
    """Return (dx, dy) — signed x and y distances between two points."""
    return p1[0] - p2[0], p1[1] - p2[1]
