import cv2

class InterColumnMatch:
    def __init__(self, track):
        self.track = track

        self.velocity = (self.track.end_point[0] - self.track.start_point[0],
                         self.track.end_point[1] - self.track.start_point[1])
        self.valid = self.velocity[0] > 0

    def __repr__(self):
        return f'Match: {self.track.person.id}: {self.velocity}'

    def draw(self, frame):
        self.track.draw(frame)
