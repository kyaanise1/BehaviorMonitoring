
import json


def load_feeding_zones(path):
    """Load feeding-zone rectangles and their reference resolution."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ref_width, ref_height = config["ref_size"]

    if ref_width <= 0 or ref_height <= 0:
        raise ValueError("Invalid reference image dimensions.")

    zones = config["zones"]

    for zone in zones:
        if len(zone) != 4:
            raise ValueError(f"Invalid feeding zone: {zone}")

        x1, y1, x2, y2 = zone

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid rectangle coordinates: {zone}")

        if x1 < 0 or y1 < 0 or x2 > ref_width or y2 > ref_height:
            raise ValueError(f"Zone outside reference image: {zone}")

    return zones, (ref_width, ref_height)


def get_huddling_indices(centroids, zones, ref_size, frame_size):
    """
    Return indices of birds outside all feeding zones.

    centroids: [(cx, cy), ...]
    ref_size: (reference_width, reference_height)
    frame_size: (frame_width, frame_height)
    """
    if not zones:
        return list(range(len(centroids)))

    ref_width, ref_height = ref_size
    frame_width, frame_height = frame_size

    scale_x = frame_width / ref_width
    scale_y = frame_height / ref_height

    scaled_zones = [
        (
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        )
        for x1, y1, x2, y2 in zones
    ]

    keep_indices = []

    for i, (cx, cy) in enumerate(centroids):
        inside = any(
            x1 <= cx <= x2 and y1 <= cy <= y2
            for x1, y1, x2, y2 in scaled_zones
        )

        if not inside:
            keep_indices.append(i)

    return keep_indices
