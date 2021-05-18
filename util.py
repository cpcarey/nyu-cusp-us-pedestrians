import math

def get_centroid(box):
    """Returns the centroid of the given box tuple (left, top, right, bottom).
    """
    return (round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2))

def get_magnitude(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

def get_rounded_vector(vector):
    return (round(vector[0]), round(vector[1]))

def translate(point, translation):
    """Returns the given point translated by the given vector."""
    return (point[0] + translation[0], point[1] + translation[1])
