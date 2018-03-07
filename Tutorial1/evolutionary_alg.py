import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.colors import ListedColormap, BoundaryNorm
from time import time
from multiprocessing import Pool
from Simulation import Simulation

def evolutionaryAlgorithm(num_iter, population_size, ndim, rn_range, benchmark_function, offspring, graphics=False):
    # Initialise
    population = np.random.rand(population_size, ndim[0], ndim[1]) * rn_range[0] + rn_range[1]
    average_genotype = np.mean(population[0:], axis=0)
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
            for j in range(i, population_size):
                diversity += np.linalg.norm(population[i].flatten() - population[j].flatten())
        print("diversity: {}".format(diversity))
        # Selection
        population = population[fitness_all.argsort()]
        population = np.flip(population, 0)
        population = population[0:round(population.shape[0] * offspring)]
        # Reproduction
        population = population.repeat(round(1 / offspring), axis=0)
        # Crossover/Mutation
        for i in range(population_size):
            if np.random.rand() < 0.2:
                population[i] = np.reshape(
                    [np.mean([population[i][j][k], population[np.random.randint(population.shape[0])][j][k]]) for j in
                     range(population.shape[1]) for k in range(population.shape[2])], (2, 15))
        for i in range(population_size):
            if np.random.rand() < mutation_prob:
                weight_index = np.random.randint(ndim[1])
                layer_index = np.random.randint(ndim[0])
                mutation = np.random.normal(loc=0, scale=rn_range[0] / 10)
                print("mutating {} at [{}][{}] with {}".format(i, layer_index, weight_index, mutation))
                population[i][layer_index][weight_index] += mutation

        # Output
        average_genotype = np.mean(population[0:], axis=0)
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
pool = Pool(processes=4)
'''
Parameters to setup evolutionary algorithm
'''
iter_ea = 10
population_size = 100
ndim = 2, 15
rn_range = [10, -5]
offspring = 0.5
# fitness,num_collisions,surface_covered = simulation(env_range,pos,robot_rad,sens_range,dT,weights, graphics=False)
avg = np.array(
    [[0.15904197, 0.60632302, -0.27319458, -0.01744001, -0.46646285, -0.02565339,
      0.14300986, 0.19738202, -0.54087545, 0.5843433, -0.1273625, -0.24264141,
      -1.92671453, 0.76185036, 0.56204834],
     [0.05057849, 0.07818218, -0.43762829, 0.01238846, 0.36921576, 0.55867089,
      -0.34058266, -1.75835065, -1.15323556, -1.0287542, 0.05006896, -0.10920556
         , 0.43480835, -0.50853751, 0.11581586]]
)

sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT)

output = evolutionaryAlgorithm(iter_ea, population_size, ndim, rn_range, sim.simulate, offspring, True)
print(output)
f= open(str(int(time()))+".txt", "w")
for i in range(len(output)):
    f.write("\n####\n")
    f.write(output[i])



fmt = '{:<15}{:<25}{:<25}{}'
print(fmt.format("Iteration", "Collisions", "Surface", "Fitness"))
