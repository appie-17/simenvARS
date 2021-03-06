import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.colors import ListedColormap, BoundaryNorm
from time import time
from multiprocessing import Pool


# Check if point c lies between line (a,b)
def collision(walls, pos):
    eps = 0.01
    for wall in walls:
        ab = np.sqrt((wall[0, 0] - wall[1, 0]) ** 2 + (wall[0, 1] - wall[1, 1]) ** 2)

        ac = np.sqrt((wall[0, 0] - pos[0] - robot_rad) ** 2 + (wall[0, 1] - pos[1] - robot_rad) ** 2)
        cb = np.sqrt((wall[1, 0] - pos[0] - robot_rad) ** 2 + (wall[1, 1] - pos[1] - robot_rad) ** 2)
        if (ac + cb <= ab + eps) & (ac + cb >= ab - eps):
            return True

        ac2 = np.sqrt((wall[0, 0] - pos[0] - robot_rad) ** 2 + (wall[0, 1] - pos[1] + robot_rad) ** 2)
        cb2 = np.sqrt((wall[1, 0] - pos[0] - robot_rad) ** 2 + (wall[1, 1] - pos[1] + robot_rad) ** 2)
        if (ac2 + cb2 <= ab + eps) & (ac2 + cb2 >= ab - eps):
            return True
        ac3 = np.sqrt((wall[0, 0] - pos[0] + robot_rad) ** 2 + (wall[0, 1] - pos[1] - robot_rad) ** 2)
        cb3 = np.sqrt((wall[1, 0] - pos[0] + robot_rad) ** 2 + (wall[1, 1] - pos[1] - robot_rad) ** 2)
        if (ac3 + cb3 <= ab + eps) & (ac3 + cb3 >= ab - eps):
            return True
        ac4 = np.sqrt((wall[0, 0] - pos[0] - robot_rad) ** 2 + (wall[0, 1] - pos[1] + robot_rad) ** 2)
        cb4 = np.sqrt((wall[1, 0] - pos[0] - robot_rad) ** 2 + (wall[1, 1] - pos[1] + robot_rad) ** 2)
        if (ac4 + cb4 <= ab + eps) & (ac4 + cb4 >= ab - eps):
            return True


# Initialise positions for 12 sensors
def init_sensors(pos, theta):
    sensors = np.zeros([12, 2, 2])
    for i in range(len(sensors)):
        sensors[i] = [[pos[0] + np.sin(theta) * robot_rad,
                       pos[1] + np.cos(theta) * robot_rad],
                      [pos[0] + np.sin(theta) * (robot_rad + sens_range),
                       pos[1] + np.cos(theta) * (robot_rad + sens_range)]]
        theta += 1 / 6 * np.pi
    return sensors


def wall_distance(sensors, walls):
    distance = np.zeros(12)
    i = 0
    for sensor in sensors:

        for wall in walls:
            x1, y1, x2, y2 = sensor[0, 0], sensor[0, 1], sensor[1, 0], sensor[1, 1]
            x3, y3, x4, y4 = wall[0, 0], wall[0, 1], wall[1, 0], wall[1, 1]
            # Repair vertical wall/sensor
            if x1 == x2:
                x1 += 0.001
            if x3 == x4:
                x3 += 0.001
            # Calculate intersection point between wall and sensor line
            Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            # Check for true intersection between walls and sensors
            if (Px < np.minimum(x1, x2)) | (Px > np.maximum(x1, x2)):
                pass
            elif (Px < np.minimum(x3, x4)) | (Px > np.maximum(x3, x4)):
                pass
            else:
                distance[i] = np.sqrt((x1 - Px) ** 2 + (y1 - Py) ** 2)
                break
        i += 1
    return distance


# Velocity left- and right wheel between [-1,1]
def movement(Vl, Vr, pos, theta):
    l = robot_rad * 2
    x, y = pos[0], pos[1]
    omega = (Vr - Vl) / l
    if Vl == Vr:
        V = (Vl + Vr) / 2
        x = x + V * np.cos(theta * dT)
        y = y + V * np.sin(theta * dT)
        theta = theta
    else:
        R = l / 2 * ((Vl + Vr) / (Vr - Vl))
        ICCx = x - R * np.sin(theta)
        ICCy = y + R * np.cos(theta)
        x = np.cos(omega * dT) * (x - ICCx) - np.sin(omega * dT) * (y - ICCy) + ICCx
        y = np.sin(omega * dT) * (x - ICCy) + np.cos(omega * dT) * (y - ICCy) + ICCy
        theta = theta + omega * dT
    pos_new = np.array([x, y])
    return pos_new, theta


def ann(weights, v_left, v_right, sensor_output):
    """
    Neural network combining previous velocities and sensor distance outputs.
    (Trained) weights are multiplied with this combined vector.
    :param v_left: float [-1, 1]
    :param v_right: float [-1, 1]
    :param sensor_output: numpy array with shape (12,)
    :param weights: numpy matrix with shape (2, 14) and values [0, 1]
    :return: new velocities
    """
    # append v_left and v_right to sensor_output and set correct shape
    input_vector = np.append(sensor_output, [v_left, v_right, 1])
    # print(input_vector)
    # multiply input_input vector by weights and put through tanh activation function
    output = 1 / (1 + np.exp(-np.dot(weights, input_vector)))
    # return vector of 2x1; v_left = output[0][0] v_right = output[1][0]
    return output


def simulation(iter, env_range, pos, robot_rad, sens_range, dT, weights, graphics=False):
    # Initialise velocities for right and left wheel of robot
    Vl = 0
    Vr = 0
    # Set theta to viewing direction of robot
    theta = np.arctan(pos[1] / pos[0])
    # add walls to 4x2x2d array, giving start- & end-coordinates
    # for each wall surrounding the environment
    walls = np.array([[[0, 0], [0, env_range]]])
    walls = np.vstack((walls, np.array([[[0, 0], [env_range, 0]]])))
    walls = np.vstack((walls, np.array([[[env_range, 0], [env_range, env_range]]])))
    walls = np.vstack((walls, np.array([[[0, env_range], [env_range, env_range]]])))
    walls = np.vstack((walls, np.array([[[3, 3], [7, 3]]])))
    walls = np.vstack((walls, np.array([[[3, 3], [3, 7]]])))
    walls = np.vstack((walls, np.array([[[3, 7], [7, 7]]])))
    walls = np.vstack((walls, np.array([[[7, 7], [7, 3]]])))

    # Initialise variables to measure fitness
    num_collisions = 0
    x, y = np.asscalar(pos[0]), np.asscalar(pos[1])
    surface_covered = {(np.round(x), np.round(y))}

    # Run simulation
    if graphics is True:
        plt.ion()
        ax = plt.subplot(111)
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)
        lc_walls = mc.LineCollection(walls)

    for i in range(iter):
        # Calculate new position and viewing angle according to velocities
        pos_old, theta_old = pos, theta
        pos, theta = movement(Vl, Vr, pos, theta)
        # Add unique positions to surface_covered
        x, y = np.asscalar(pos[0]), np.asscalar(pos[1])
        surface_covered.add((np.round(x), np.round(y)))
        # When collision ignore movement orthogonal to wall only allow parralel movement
        # while collision(walls,pos):
        if collision(walls, pos):
            num_collisions += 1
            pos, theta = pos_old, theta_old
        # Define 12 sensors each separated by 30 deg,2pi/12rad and calculate distance to any object
        sensors = init_sensors(pos, theta)
        sens_distance = wall_distance(sensors, walls)
        if graphics is True:
            ax.clear()
            _ = plt.xlim(-2, 12)
            _ = plt.ylim(-2, 12)
            robot = plt.Circle(pos, robot_rad)
            linecolors = ['red' if i == 0 else 'blue' for i in range(12)]
            lc_sensors = mc.LineCollection(sensors, colors=linecolors)
            _ = ax.add_artist(robot)
            _ = ax.add_collection(lc_walls)
            _ = ax.add_collection(lc_sensors)
            plt.pause(1e-40)
            # When 1/dT=1 run controller and calculate new velocities according to old vel. and sensor output
        if (i * dT) % 1 == 0:
            Vl, Vr = ann(weights, Vl, Vr, sens_distance)
    fitness = len(surface_covered) / (np.log(num_collisions + 1) + 1)
    # plt.close()
    return fitness


def evolutionaryAlgorithm(num_iter, population_size, ndim, rn_range, benchmark_function, offspring):
    # Initialise
    population = np.random.rand(population_size, ndim[0], ndim[1]) * rn_range[0] + rn_range[1]
    average_genotype = np.mean(population[0:], axis=0)
    fitness = benchmark_function(iter_sim, env_range, pos, robot_rad, sens_range, dT, weights=average_genotype,
                                 graphics=False)
    output = np.array([fitness])
    # arrays for keeping the values for the average and best fitness per iteration
    averages = np.zeros(num_iter)
    bests = np.zeros(num_iter)
    fitnesess = np.zeros((num_iter, population_size))
    mutation_prob = 0.1
    plt.figure(2)
    plt.ion()

    for _ in range(num_iter):
        print(_)
        # Evaluation
        fitness_all = []
        for i in range(population_size):
            sim_start = time()
            fitness = benchmark_function(iter_sim, env_range, pos, robot_rad, sens_range, dT, weights=population[i],
                                         graphics=False)
            print("simulation {} took {}".format(i, time() - sim_start))
            fitness_all = np.append(fitness_all, fitness)
        print('Fitness :', fitness_all)
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
        fitness = benchmark_function(iter_sim, env_range, pos, robot_rad, sens_range, dT, weights=average_genotype,
                                     graphics=False)

        averages[_] = fitness_all.mean()
        bests[_] = fitness_all.max()

        plt.plot(bests[0:_ + 1], color="green")
        plt.plot(averages[0:_ + 1], color="red")
        plt.pause(3)
        output = np.append(output, fitness)
    return output


'''
Parameters to setup simulation for cleaning robot
'''
# Define range and starting point within square polygon environment
env_range = 20
# pos = np.random.rand(2,1)*6-3
pos = np.array([2, 2])
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
iter_ea = 100
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

simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT, weights=avg, graphics=False)

output = evolutionaryAlgorithm(iter_ea, population_size, ndim, rn_range, simulation, offspring)
print(output)

fmt = '{:<15}{:<25}{:<25}{}'
print(fmt.format("Iteration", "Collisions", "Surface", "Fitness"))
