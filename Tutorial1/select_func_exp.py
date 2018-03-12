import selectionreproduction
import fitness
import os
import numpy as np
from Simulation import Simulation
from evolutionary_alg import evolutionaryAlgorithm


# (Jan Lucas)
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
    # Define number of threads/multi-core

    '''
    Parameters to setup evolutionary algorithm
    '''
    iter_ea = 100
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

    population_size = 100
    sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT, fitness.OurFirstFitnessFunction)

    for k in [5, 10, 20]:
        selection = selectionreproduction.Tournament(k).apply
        for i in range(10):

            print("###### run {} k {}".format(i, k))

            fitness, best_individual, diversities = evolutionaryAlgorithm(iter_ea, population_size, layers, ndim, rn_range,
                                                                          sim.simulate, crossover, mutation,
                                                                          selection,
                                                                          False)
            print('Fitness :', fitness)
            print('Best :', best_individual)
            print('Diversities :', diversities)

            out_dir = "./output/tournament-{}-run-{}".format(k, i)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            np.savetxt(out_dir + "/fit.csv", fitness, delimiter=",")
            np.savetxt(out_dir + "/diversity.csv", diversities, delimiter=",")

            f = open(out_dir + '/weights.txt', 'wb')
            for node in best_individual:
                np.savetxt(f, node, delimiter=',', footer='end_layer')

            f.close()
    for offspring in [0.5, 0.25, 0.1]:
        selection = selectionreproduction.TruncatedRankBased(offspring).apply
        for i in range(10):

            print("###### run {} offspring {}".format(i, offspring))

            fitness, best_individual, diversities = evolutionaryAlgorithm(iter_ea, population_size, layers, ndim,
                                                                          rn_range,
                                                                          sim.simulate, crossover, mutation,
                                                                          selection,
                                                                          False)
            print('Fitness :', fitness)
            print('Best :', best_individual)
            print('Diversities :', diversities)

            out_dir = "./output/trunc-{}-run-{}".format(offspring, i)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            np.savetxt(out_dir + "/fit.csv", fitness, delimiter=",")
            np.savetxt(out_dir + "/diversity.csv", diversities, delimiter=",")

            f = open(out_dir + '/weights.txt', 'wb')
            for node in best_individual:
                np.savetxt(f, node, delimiter=',', footer='end_layer')

            f.close()