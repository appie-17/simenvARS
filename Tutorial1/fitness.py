import numpy as np


class Fitness:
    def __init__(self, x, y, num_collisions):
        self.x = x
        self.y = y
        self.num_collisions = num_collisions

    def update(self, x, y, num_collisions):
        self.x = x
        self.y = y
        self.num_collisions = num_collisions

    def calculate(self):
        pass


class OurFirstFitnessFunction(Fitness):
    def __init__(self, x, y, num_collisions):
        super().__init__(x, y, num_collisions)
        self.surface_covered = {(np.round(self.x), np.round(self.y))}

    def update(self, x, y, num_collisions):
        super().update(x, y, num_collisions)
        self.surface_covered.add((np.round(x), np.round(y)))

    def calculate(self):
        return len(self.surface_covered) / (np.log(self.num_collisions + 1) + 1)
