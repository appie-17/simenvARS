import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.colors import ListedColormap, BoundaryNorm
from time import time
from multiprocessing import Pool
from Simulation import Simulation

def evolutionaryAlgorithm(num_iter, population_size, layers, ndim, rn_range, benchmark_function, offspring, graphics=False):
    # Initialise
    population = np.array([np.array([np.random.rand(ndim[l],ndim[l+1])* rn_range[0] + rn_range[1] for l in range(layers)]) for _ in range(population_size)])
    average_genotype = np.mean(population[:],axis=0)
    fitness = benchmark_function(average_genotype)
    output = np.array([fitness])
    # arrays for keeping the values for the average and best fitness per iteration
    averages = np.zeros(num_iter)
    bests = np.zeros(num_iter)
    fitnesess = np.zeros((num_iter, population_size))
    diversities = np.zeros(num_iter)
    mutation_prob = 0.3
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

        diversity = 0
        for i in range(population_size):
            for l in range(layers):
                for j in range(i, population_size):
                    diversity += np.linalg.norm(population[i][l].flatten() - population[j][l].flatten())
        print("diversity: {}".format(diversity))
        # Selection
        population = population[fitness_all.argsort()]
        population = np.flip(population, 0)
        best_genotype = population[0]
        print('Best :', best_genotype)
        population = population[0:round(population.shape[0] * offspring)]
        # Reproduction
        population = population.repeat(round(1 / offspring), axis=0)
        # Crossover/Mutation
        reshape= [0 for x in range(layers)]
        for i in range(population_size):
            if np.random.rand() < 0.2:
                for l in range(layers):
                    reshape[l] = [ndim[l],ndim[l+1]]
                    population[i][l] = np.array([np.mean([population[i][l][j][k],population[np.random.randint(population_size)][l][j][k]]) for j in range(ndim[l]) for k in range(ndim[l+1])])
                    population[i][l] = population[i][l].reshape(reshape[l])    
                
        for i in range(population_size):
            if np.random.rand() < mutation_prob:                
                layer_index = np.random.randint(layers)
                print(layer_index)
                weight_index = np.random.randint(ndim[layer_index]), np.random.randint(ndim[layer_index+1])
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
            div_plt.plot(diversities[0:_+1], color="orange")
            plt.pause(3)
        output = np.append(output, fitness)
    return output, population[fitness_all.argmax()], diversities


'''
Parameters to setup simulation for cleaning robot
'''
# Define range and starting point within square polygon environment
env_range = 20
# pos = np.random.rand(2,1)*6-3
pos = np.array([10, 10])
# Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
robot_rad = 1
sens_range = 3
dT = 0.1
np.random.seed(5)
iter_sim = 500
#Define number of threads/multi-core
pool = Pool(processes=4)
'''
Parameters to setup evolutionary algorithm
'''
iter_ea = 10
population_size = 100
#Choose number of layers besides input layer, so hidden+output is 2 layers
layers = 3
#Input layer 15 nodes(12sensors/2velocities/1bias), arbitrarely number of hidden nodes, 2 output nodes (velocities)
ndim = [15,5,10,2]
rn_range = [10, -5]
#Selection criteria
offspring = 0.5

sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT)

fitness, best_individual, diversities = evolutionaryAlgorithm(iter_ea, population_size, layers, ndim, rn_range, sim.simulate, offspring, True)
print('Fitness :',fitness)
print('Best :', best_individual)
print('Diversities :',diversities)
import os

out_dir = "./output/{}".format(int(time()))
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

np.savetxt(out_dir + "/fit.csv", fitness, delimiter=",")
np.savetxt(out_dir + "/best.csv", best_individual, delimiter=",")
np.savetxt(out_dir + "/diversity.csv", diversities, delimiter=",")