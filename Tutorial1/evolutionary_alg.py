import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.colors import ListedColormap, BoundaryNorm
from time import time
from multiprocessing import Pool
from Simulation import Simulation
import fitness
import selectionreproduction


def evolutionaryAlgorithm(num_iter, population_size, layers, ndim, rn_range, benchmark_function, 
                          crossover_prob, mutation_prob, selection_method, graphics=False):
    # Initialise
    population = np.array(
        [np.array([np.random.rand(ndim[l], ndim[l + 1]) * rn_range[0] + rn_range[1] for l in range(layers)]) for _ in
         range(population_size)])
    average_genotype = np.mean(population[:], axis=0)
    fitness = benchmark_function(average_genotype)
    output = np.array([fitness])
    # arrays for keeping the values for the average and best fitness per iteration
    averages = np.zeros(num_iter)
    bests = np.zeros(num_iter)
    fitnesess = np.zeros((num_iter, population_size))
    diversities = np.zeros(num_iter)
    pool = Pool(processes=4)

    if graphics:
        plt.figure(2)
        fitness_plt = plt.subplot(211)
        fitness_plt.set_title("fitness")
        div_plt = plt.subplot(212)
        div_plt.set_title("diversity")
        plt.ion()

    for _ in range(num_iter):
        print(_)
        # Evaluation
        fitness_all = []
        sim_start = time()
        fitness_all = np.array(pool.map(benchmark_function, population))
        print("time for experiments: {}".format(time() - sim_start))
        print('Fitness :', fitness_all)
        print("population: {}".format(population))

        diversity = 0
        for i in range(population_size):
            for l in range(layers):
                for j in range(i, population_size):
                    diversity += np.linalg.norm(population[i][l].flatten() - population[j][l].flatten())
        print("diversity: {}".format(diversity))

        # Selection & Reproduction
        population = selection_method(fitness_all, population)

        best_genotype = population[0]
        print('Best :', best_genotype)

        # Crossover/Mutation
        reshape = [0 for x in range(layers)]
        for i in range(population_size):
            if np.random.rand() < crossover_prob:
                for l in range(layers):
                    reshape[l] = [ndim[l], ndim[l + 1]]
                    population[i][l] = np.reshape(np.array(
                        [np.mean([population[i][l][j][k], population[np.random.randint(population_size)][l][j][k]]) for
                         j in range(ndim[l]) for k in range(ndim[l + 1])]), reshape[l])
                    # population[i][l] = population[i][l].reshape(reshape[l])    

        for i in range(population_size):
            if np.random.rand() < mutation_prob:
                layer_index = np.random.randint(layers)
                print(layer_index)
                weight_index = np.random.randint(ndim[layer_index]), np.random.randint(ndim[layer_index + 1])
                mutation = np.random.normal(loc=0, scale=rn_range[0] / 10)
                print("mutating {} at [{}][{}] with {}".format(i, layer_index, weight_index, mutation))
                population[i][layer_index][weight_index] += mutation

        # Output
        average_genotype = np.mean(population[:], axis=0)
        print('Average :', average_genotype)
        fitness = benchmark_function(average_genotype)

        averages[_] = fitness_all.mean()
        bests[_] = fitness_all.max()
        diversities[_] = diversity

        if graphics:
            fitness_plt.plot(bests[0:_ + 1], color="green")
            fitness_plt.plot(averages[0:_ + 1], color="red")
            div_plt.plot(diversities[0:_ + 1], color="orange")
            plt.pause(3)
        output = np.append(output, fitness)
    return output, population[fitness_all.argmax()], diversities


# if-statement necessary for multiprocessing on Windows:
if __name__ == "__main__":
    '''
    Parameters to setup simulation for cleaning robot
    '''
    # Define range and starting point within square polygon environment
    env_range = 20
    # pos = np.random.rand(2,1)*6-3
    pos = np.array([-6, -6])
    # Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
    robot_rad = 1
    sens_range = 3
    dT = 0.1
    np.random.seed(5)
    iter_sim = 1000
    '''
    Parameters to setup evolutionary algorithm
    '''
    iter_ea = 100
    population_size = 100
    # Choose number of layers besides input layer, so hidden+output is 2 layers
    layers = 2
    # Input layer 15 nodes(12sensors/2velocities/1bias), arbitrarily number of hidden nodes, 2 output nodes (velocities)
    ndim = [15, 10, 2]
    rn_range = [10, -5]
    # Selection criteria
    offspring = 0.5
    crossover = 0.2
    mutation = 0.3
    #Selection choose between TruncatedRankBased(offspring)/Tournament(k)
    selection = selectionreproduction.Tournament(10).apply

    sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT, fitness.OurFirstFitnessFunction)

    fitness, best_individual, diversities = evolutionaryAlgorithm(iter_ea, population_size, layers, ndim, rn_range,
                                                                  sim.simulate, crossover, mutation,
                                                                  selection,
                                                                  True)
    print('Fitness :', fitness)
    print('Best :', best_individual)
    print('Diversities :', diversities)
    import os

    out_dir = "./output/{}".format(int(time()))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savetxt(out_dir + "/fit.csv", fitness, delimiter=",")
    np.savetxt(out_dir + "/diversity.csv", diversities, delimiter=",")

    f = open(out_dir + '/weights.txt', 'wb')
    for node in best_individual:
        np.savetxt(f, node, delimiter=',', footer='end_layer')

    f.close()


    # Function to import weights
    def tokenizer(fname):
        with open(fname) as f:
            weights = []
            for line in f:
                if 'end_node' in line:
                    yield weights
                    weights = []
                    continue
                weights.append(line)


    # Import weights
    weights = np.array([np.loadtxt(A, delimiter=',') for A in tokenizer('weights.txt')])
