class RoiMatch:
    def __init__(self, person, time):
        self.person = person
        self.locked = False
        self.time = time

        self.path = []
        self.times = []
        self.marked = False

        self.extend(person, time)

    def __repr__(self):
        if len(self.times) < 2:
            return '---'
        return f'{self.person.id}: {self.times[-1] - self.times[0]}'

    def extend(self, person, time):
        if not self.locked:
            self.path.append(person.centroid)
            self.times.append(time)

    def lock(self):
        pass
        #if len(self.path) > 100:
        #    print('locked: ', self)
        #    self.locked = True
        #    roi_matches_out.append(self)
