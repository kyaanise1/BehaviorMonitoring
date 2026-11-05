
from pathlib import Path

import numpy as np
import motmetrics as mm
from ultralytics import YOLO
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from eval_tracking import _rows

GT_FILE = "annotated_tracking_frames/gt/gt.txt"
PRED_FILE = "pred_hybrid.txt"
WEIGHTS = "../runs/broiler_yolo11n/weights/best.pt"

gt = mm.io.loadtxt(GT_FILE, fmt="mot15-2D")
pred = mm.io.loadtxt(PRED_FILE, fmt="mot15-2D")
model = YOLO(WEIGHTS)

images = sorted(
    list(Path("tracking_frames").glob("*.jpg"))
    + list(Path("tracking_frames").glob("*.png"))
)


def match_count(gt_boxes, pred_boxes, threshold=0.5):
    """Maximum one-to-one matches with IoU >= threshold."""
    a = np.asarray(gt_boxes, dtype=float).reshape(-1, 4).copy()
    b = np.asarray(pred_boxes, dtype=float).reshape(-1, 4).copy()

    if len(a) == 0 or len(b) == 0:
        return 0

    # MOT boxes: x, y, width, height -> x1, y1, x2, y2
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)

    intersection = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

    union = area_a[:, None] + area_b[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-9)

    valid = csr_matrix(iou >= threshold)
    matches = maximum_bipartite_matching(
        valid, perm_type="column"
    )

    return int(np.sum(matches != -1))


print(
    "Frame | GT | YOLO all matched | "
    "YOLO high matched | Hybrid matched"
)
print("-" * 68)

for frame in range(4, 10):
    image = images[frame - 1]

    _, gt_boxes = _rows(gt, frame)
    _, hybrid_boxes = _rows(pred, frame)

    result = model.predict(
        str(image), conf=0.1, imgsz=640, verbose=False
    )[0]

    boxes = result.boxes.xyxy.cpu().numpy().copy()
    scores = result.boxes.conf.cpu().numpy()

    # Convert YOLO xyxy to MOT x,y,w,h
    boxes[:, 2:] -= boxes[:, :2]

    high_boxes = boxes[scores >= 0.7]

    print(
        f"{frame:5d} | "
        f"{len(gt_boxes):2d} | "
        f"{match_count(gt_boxes, boxes):16d} | "
        f"{match_count(gt_boxes, high_boxes):17d} | "
        f"{match_count(gt_boxes, hybrid_boxes):14d}"
    )
