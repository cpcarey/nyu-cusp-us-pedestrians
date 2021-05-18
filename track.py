from enum import Enum
import numpy as np
import constants
import cv2
import util
from roi import Roi

roi = Roi()

class TrackClassification(Enum):
    NONE = 0
    INSIDE_ROI = 1
    OUTSIDE_ROI = 2

class Track:
    def __init__(self, person, time):
        self.person = person

        self.path = []

        self.classification = TrackClassification.NONE

        self.extend(person, time)

        self.start_point = None
        self.end_point = None

    def extend(self, person, time):
        self.path.append((person.centroid[0],
                          person.centroid[1],
                          time))

    def is_classified(self):
        return self.classification != TrackClassification.NONE

    def is_long(self):
        if self.start_point == None or self.end_point == None:
            return False

        vector = (self.end_point[0] - self.start_point[0],
                  self.end_point[1] - self.start_point[1])
        magnitude = util.get_magnitude(vector)
        if len(self.path) > 30 and magnitude > 20:
            return True
        return False

    def classify(self):
        if self.is_classified():
            return self.classification

        if len(self.path) < constants.ROI_MATCH_PATH_THRESHOLD:
            return TrackClassification.NONE

        N = 5
        path_start = self.path[0:N]
        start_centroid = (int(round(np.sum([p[0] for p in path_start]) / N)),
                          int(round(np.sum([p[1] for p in path_start]) / N)))

        start_time = self.path[0][2]
        end_centroid = None
        end_time = None
        entered = False
        splice_index = -1

        for index, point in enumerate(self.path):
            time = point[2]
            if (time - start_time >= constants.ROI_MATCH_TIME_THRESHOLD and
                index >= constants.ROI_MATCH_PATH_THRESHOLD):

                end_time = time

                path_end = self.path[index + 1 - N: index + 1]
                end_centroid = (int(round(np.sum([p[0] for p in path_end]) / N)),
                                int(round(np.sum([p[1] for p in path_end]) / N)))
                splice_index = index

                self.start_point = (start_centroid[0], start_centroid[1], start_time)
                self.end_point = (end_centroid[0], end_centroid[1], end_time)

                if roi.contains(self.start_point):
                    self.classification = TrackClassification.INSIDE_ROI
                else:
                    self.classification = TrackClassification.OUTSIDE_ROI

                return self.classification

        return self.classification

    def draw(self, frame):
        if not self.is_long():
            return

        color = (220, 220, 220)
        if self.classification == TrackClassification.INSIDE_ROI:
            color = (0, 0, 220)

        cv2.line(frame,
                 self.start_point[:2],
                 self.end_point[:2],
                 color=color,
                 thickness=1)
