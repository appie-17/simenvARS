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
    # Initialise # (Jordy van Appeven)
    population = np.array(
        [np.array([np.random.normal(size=(ndim[l], ndim[l + 1]),loc=0, scale=rn_range[0]) for l in range(layers)]) for _ in
         range(population_size)])
    np.random.normal(loc=0, scale=rn_range[0])
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
        locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
        pos = locations[np.random.randint(locations.shape[0])]
        
        benchmark_function.pos = pos
        print(_)
        # Evaluation
        fitness_all = []
        sim_start = time()
        fitness_all = np.array(pool.map(benchmark_function.simulate, population))
        print("time for experiments: {}".format(time() - sim_start))
        print('Fitness :', fitness_all)

        diversity = 0
        for i in range(population_size):
            for l in range(layers):
                for w in range(ndim[l]):
                    for j in range(i,population_size):
                        diversity += np.linalg.norm(population[i][l][w]-population[j][l][w])
        print("diversity: {}".format(diversity))
        fitnesess[_] = fitness_all
        best_genotype = population[np.argmax(fitness_all)]
        print('Best :', best_genotype)
        averages[_] = fitness_all.mean()
        bests[_] = fitness_all.max()
        diversities[_] = diversity
        # Selection & Reproduction # (Sebas Higler)
        population = selection_method(fitness_all, population)
        
        # np.save('pop',population)
        # Crossover/Mutation
        cross = 0
        for i in range(population_size):
            for l in range(layers):
                for w in range(ndim[l]*ndim[l+1]):
                    if np.random.rand() < crossover_prob:
                        cross+=1
                        weight_index = np.random.randint(ndim[l]), np.random.randint(ndim[l + 1])
                        weight_temp = population[i][l][weight_index]
                        crossover_partner = np.random.randint(population_size)
                        population[i][l][weight_index] = population[crossover_partner][l][weight_index]
                        population[crossover_partner][l][weight_index] = weight_temp
        mut = 0                
        for i in range(population_size):
            for l in range(layers):
                for w in range(ndim[l]*ndim[l+1]):
                    if np.random.rand() < mutation_prob:
                        mut+=1
                        weight_index = np.random.randint(ndim[l]), np.random.randint(ndim[l + 1])
                        mutation = np.random.normal(loc=0, scale=rn_range[0])
                        population[i][l][weight_index] += mutation
        print('Crossover :',cross)
        print('Mutation : ',mut)                

        # (Jordy van Appeven)
        if graphics:
            fitness_plt.plot(bests[0:_ + 1], color="green")
            fitness_plt.plot(averages[0:_ + 1], color="red")
            div_plt.plot((diversities[0:_ + 1]), color="orange")
            div_plt.set_yscale('log')
            plt.pause(3)
        
    return fitnesess, best_genotype, diversities


# if-statement necessary for multiprocessing on Windows:
if __name__ == "__main__":
    '''
    Parameters to setup simulation for cleaning robot
    '''
    # Define range and starting point within square polygon environment
    env_range = 20
    pos = np.array([-6, -6])
    # Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
    robot_rad = 1
    sens_range = 3
    dT = 1/3
    iter_sim = 500
    '''
    Parameters to setup evolutionary algorithm
    '''
    iter_ea = 300
    population_size = 100
    # Choose number of layers besides input layer, so hidden+output is 2 layers
    layers = 1
    # Input layer 15 nodes(12 sensors/2 velocities/1 bias), arbitrarily number of hidden nodes, 2 output nodes (velocities)
    ndim = [15,2]
    rn_range = [1, -1]
    # Selection criteria
    offspring = 0.2
    crossover = 0.1
    mutation = 0.05
    # Selection choose between TruncatedRankBased(offspring)/Tournament(k)
    selection = selectionreproduction.Tournament(8).apply
    #Define map
    sim_map = np.load('Maps/'+'Map1'+'.npy')
    sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT, fitness.OurFirstFitnessFunction,sim_map)

    fitness, best_individual, diversities = evolutionaryAlgorithm(iter_ea, population_size, layers, ndim, rn_range,
                                                                  sim, crossover, mutation,
                                                                  selection,
                                                                  True)
    print('Fitness :', fitness)
    print('Best :', best_individual)
    print('Diversities :', diversities)
    import os

    # (Jan Lucas)
    out_dir = "./output/{}".format(int(time()))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savetxt(out_dir + "/fit.csv", fitness, delimiter=",")
    np.savetxt(out_dir + "/diversity.csv", diversities, delimiter=",")

    f = open(out_dir + '/weights.txt', 'wb')
    for node in best_individual:
        np.savetxt(f, node, delimiter=',', footer='end_layer')

    f.close()