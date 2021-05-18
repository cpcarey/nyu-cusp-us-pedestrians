import util

class Displacement:
    """An encapsulation of the displacement between two points used for
    comparison."""

    def __init__(self, p0, p1, id0, id1):
        self.p0 = p0
        self.p1 = p1
        self.vector = (p1[0] - p0[0], p1[1] - p0[1])
        self.id0 = id0
        self.id1 = id1
        self.magnitude = util.get_magnitude(self.vector)

    def __repr__(self):
        return f'<<{self.id0}-{self.id1}: {str((round(self.vector[0]), round(self.vector[1])))}>>'

    def __lt__(self, distance):
        return self.magnitude < distance.magnitude

    def __eq__(self, distance):
        return self.magnitude == distance.magnitude

    def reverse(self):
        return Displacement(self.p1, self.p0, self.id1, self.id0)
