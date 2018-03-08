class Fitness:
    def __init__(self):
        pass

    def update(self):
        raise NotImplementedError()

    def calculate(self):
        raise NotImplementedError()


class OurFirstFitnessFunction(Fitness):
    def __init__(self):
        super().__init__()

    def calculate(self):
        pass

    def update(self):
        pass
