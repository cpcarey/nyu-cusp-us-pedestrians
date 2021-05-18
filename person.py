from displacement import Displacement
import cv2
import util
import constants

class Person:
    """A detected person with an ID and bounding box."""

    def __init__(self, box, id):
        self.box = box
        self.centroid = util.get_centroid(box)
        self.id = id
        self.matched_displacement = Displacement((0, 0), (0, 0), 0, 0)
        self.prev_velocities = []

    def draw(self, frame):
        cv2.circle(frame, self.centroid, 2, (0, 255, 0), 2)
        # Display the matched ID of the person.
        cv2.putText(frame, str(self.id), util.translate(self.centroid, (5, -5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 255, 0),
                    thickness=2)
        # Display the rounded mean velocity in px/frame.
        cv2.putText(frame,
                    str(util.get_rounded_vector(self.get_velocity_smoothed())),
                    util.translate(self.centroid, (5, 15)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 255),
                    thickness=2)
        # Display the rounded speed in px/frame.
        cv2.putText(frame,
                    str(round(util.get_magnitude(self.get_velocity_smoothed()))),
                    util.translate(self.centroid, (5, 40)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 9, 255),
                    thickness=2)

    def append_velocity(self):
        """Updates this persons velocity buffer with its current velocity while
        ensuring a maximum velocity buffer length is maintained."""
        if len(self.prev_velocities) >= constants.MAX_VELOCITY_BUFFER_LENGTH:
            self.prev_velocities = self.prev_velocities[1:]
        self.prev_velocities.append(self.get_velocity())

    def get_velocity(self):
        """Returns the velocity vector which is the displacement from the
        previous position to the current position."""
        return self.matched_displacement.reverse().vector

    def get_velocity_smoothed(self):
        """Returns the smoothed velocity vector which is the mean velocity over
        the previous N frames (equally weighted)."""
        vx = 0
        vy = 0
        N = len(self.prev_velocities)
        if N == 0:
            return self.get_velocity()
        for velocity in self.prev_velocities:
            vx += velocity[0]
            vy += velocity[1]
        return (vx / N, vy / N)

    def get_projected_centroid(self):
        """Returns the expected centroid position of this person in the next
        frame based on its smoothed velocity."""
        velocity = self.get_velocity_smoothed()
        return util.translate(self.centroid, (velocity[0], velocity[1]))

    def get_displacement(self, person):
        """Returns the displacement vector class between this person and the
        given person's expected position based on its velocity."""
        return Displacement(self.centroid,
                            person.get_projected_centroid(),
                            self.id,
                            person.id)

    def __repr__(self):
        return '{}: {}'.format(self.id, self.centroid)

    def clone(person):
        clone = Person(person.box, person.id)
        clone.prev_velocities = person.prev_velocities
        return clone
