"""Shared helpers: low-rate frame sampling for video files and live streams."""
import glob
import os
import time
import cv2

def sample_frames(source, interval_s=2.0, live=False):
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                       glob.glob(os.path.join(source, "*.png")))
        for n, p in enumerate(paths, start=1):
            yield (n - 1) * interval_s, n, cv2.imread(p)
        return
    
def sample_frames(source, interval_s=2.0, live=False):
    """Yield (t_seconds, sample_number, frame), one frame every interval_s.

    File mode : t is video time (frame_index / fps); sample_number starts at 1.
    Live mode : t is seconds since start; the stream is read continuously so
                the buffer never goes stale, and one frame is kept per interval.
    """
    # Folder of still frames: one frame per interval, in sorted filename order
    if os.path.isdir(str(source)):
        paths = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                       glob.glob(os.path.join(source, "*.png")))
        if not paths:
            raise RuntimeError(f"No .jpg/.png images found in {source}")
        for n, p in enumerate(paths, start=1):
            yield (n - 1) * interval_s, n, cv2.imread(p)
        return
    
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    n = 0
    try:
        if live:
            t0 = last = time.time()
            first = True
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(1.0)          # stream hiccup; try again
                    continue
                now = time.time()
                if first or now - last >= interval_s:
                    first, last = False, now
                    n += 1
                    yield now - t0, n, frame
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise RuntimeError("Could not read FPS from the video.")
            step = max(1, round(fps * interval_s))
            i = 0
            while True:
                if not cap.grab():
                    break
                if i % step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        n += 1
                        yield i / fps, n, frame
                i += 1
    finally:
        cap.release()