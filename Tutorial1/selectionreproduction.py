import numpy as np
import random as rnd


class SelectionReproduction:
    def __init__(self):
        pass

    def apply(self, fitness_all: np.ndarray, population: np.ndarray):
        pass


class RankBased(SelectionReproduction):
    def __init__(self):
        super().__init__()

    def apply(self, fitness_all: np.ndarray, population: np.ndarray):
        # argsort => [index lowest val,...,highest], sorts pop bases on indices
        # individuals sorted from [individual with lowest fitness, ..., ind with highest fitness]
        population = population[fitness_all.argsort()]
        # reproduction:
        result_pop = np.copy(population)
        for i in range(population.shape[0]):
            rand_int = rnd.randrange(1, sum([x for x in range(1, population.shape[0] + 1)]))
            rank = population.shape[0]
            while rand_int - rank > 0:
                rand_int -= rank
                rank -= 1
            result_pop[i] = population[rank - 1]
        return result_pop


class TruncatedRankBased(SelectionReproduction):
    def __init__(self, offspring):
        self.offspring = offspring
        super().__init__()

    def apply(self, fitness_all: np.ndarray, population: np.ndarray):
        population_size = population.shape[0]
        # argsort:[index lowest val,...,highest], sorts pop bases on indices
        population = population[fitness_all.argsort()]
        # flips array on first dimension
        population = np.flip(population, 0)
        # keeps best 'offspring' percent of population, the rest is cut off:
        population = population[0:round(population.shape[0] * self.offspring)]

        # reproduction:
        population = population.repeat(round(1 / self.offspring), axis=0)
        if population.shape[0] < population_size:
            i = 0
            step = round(1 / self.offspring)
            while population.shape[0] < population_size:
                population = np.append(population, np.array([population[i]]), axis=0)
                i += step
        elif population.shape[0] > population_size:
            diff = population.shape[0] - population_size
            del_indices = []
            step = round(1 / self.offspring)
            i = population.shape[0] - 1
            while len(del_indices) < diff:
                del_indices.append(i)
                i -= step
            population = np.delete(population, del_indices, axis=0)
        return population


class Tournament(SelectionReproduction):
    def __init__(self, k):
        self.k = k
        super().__init__()

    def apply(self, fitness_all: np.ndarray, population: np.ndarray):
        population_size = population.shape[0]
        if self.k > population_size:
            raise Exception("k should <= population size")
        population = population[fitness_all.argsort()]
        resulting_pop = np.copy(population)
        # reproduction:
        for i in range(population_size):
            # tournament:
            resulting_pop[i] = population[max(rnd.sample(range(population_size), self.k))]
        return resulting_pop


if __name__ == "__main__":
    population_size = 10
    # ndim = [15, 10, 2]
    layers = 2
    ndim = [5, 4, 2]
    pop = np.array(
        [np.array(
            [np.random.rand(ndim[l], ndim[l + 1]) for l in range(layers)])
            for _ in range(population_size)])
    fitness = np.random.randint(100, size=population_size)
    # pop = np.random.rand(population_size, 22, 3, 15, 12, 3)
    c = Tournament(5).apply(fitness, pop)
    # c = RankBased().apply(fitness, pop)
    # c = TruncatedRankBased(0.3).apply(fitness, pop)
    print(c)
