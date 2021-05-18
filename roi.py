import matplotlib.path as mpl_path
import numpy as np
import cv2

BOUNDS = [
    [80, 0],
    [347, 0],
    [347, 208],
    [80, 275],
]

class Roi:
    def __init__(self):
        self.bounds = mpl_path.Path(np.array(BOUNDS))

    def contains(self, point):
        return self.bounds.contains_point(point[:2])

    def draw(self, frame):
        points = np.array(BOUNDS, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame,
                      [points],
                      isClosed=True,
                      color=(255, 0, 255),
                      thickness=2)
