class InterColumnMatch:
    def __init__(self, roi_match, start_position, end_position, start_time,
                 end_time):
        self.roi_match = roi_match
        self.start_position = start_position
        self.end_position = end_position
        self.start_time = start_time
        self.end_time = end_time

        self.velocity = (self.end_position[0] - self.start_position[0],
                         self.end_position[1] - self.start_position[1])
        self.valid = self.velocity[0] > 0

    def __repr__(self):
        return f'Match: {self.roi_match.person.id}: {self.velocity}'
