import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import localization

# (Jordy van Appeven)
class Simulation:
    def __init__(self, iter_sim, env_range, robot_rad, sens_range, dT, fitness, graphics=False):
        self.graphics = graphics
        self.sens_range = sens_range
        self.robot_rad = robot_rad
        self.dT = dT
        self.env_range = env_range
        self.iter_sim = iter_sim
        self.fitness = fitness
        self.map = None
        self.pos = None

    def simulate(self, weights, sim_map, pos):
        # Initialise velocities for right and left wheel of robot
        Vl = 0
        Vr = 0
        self.pos = pos
        
        
        env_range = self.env_range
        graphics = self.graphics
        # add walls to 4x2x2d array, giving start- & end-coordinates
        # for each wall surrounding the environment
        self.map = sim_map
        walls = self.map
        # Set theta to viewing direction of robot
        theta = 0

        # Initialise variables to measure fitness
        num_collisions = 0
        x, y = np.asscalar(pos[0]), np.asscalar(pos[1])
        surface_covered = {(np.round(x), np.round(y))}
        fitness = 0
        # Initialize instance of fitness class:
        # self.fitness = self.fitness(x, y, num_collisions)
        #Initialize alpha's for kalmanFilter
        alphas = [0.01,0.01,0.01,0.01]
        localization_iter = 5
        kalman = localization.kalmanFilter(self.map,pos,theta,alphas)
        # Run simulation
        if graphics is True:
            plt.figure(1)
            ax = plt.subplot(111)
            plt.ion()
            lc_walls = mc.LineCollection(walls)

        for i in range(self.iter_sim):
            # Calculate new position and viewing angle according to velocities
            
            pos_old, theta_old = pos, theta
            pos, theta = self.movement(Vl, Vr, pos, theta)
            
            # Add unique positions to surface_covered
            x, y = np.asscalar(pos[0]), np.asscalar(pos[1])
            surface_covered.add((np.round(x), np.round(y)))
            
            if self.collision(walls, pos):
                num_collisions += 1
                pos, theta = pos_old, theta_old
                Vl,Vr = 0,0
            sensors = self.init_sensors(pos, theta)
            sens_distance = self.wall_distance(sensors, walls)
            # Define 12 sensors each separated by 30 deg,2pi/12rad and calculate distance to any object
            
            #Update kalmanFilter
            kalman.updateKalmanFilter(pos,theta)
            
            if graphics is True:
                view = [[pos[0], pos[0] + np.cos(theta) * self.robot_rad],
                        [pos[1], pos[1] + np.sin(theta) * self.robot_rad]]
                ax.clear()
                robot_self = plt.Circle(kalman.state,self.robot_rad,color='r')
                _ = ax.add_artist(robot_self)
                for j in range(localization_iter):
                    
                    robot = plt.Circle(kalman.sampleKalmanFilter(),self.robot_rad,fill=False)
                    _ = ax.add_artist(robot)    

                lc_sensors = mc.LineCollection(sensors, linestyle='dotted')
                robot = plt.Circle(pos, self.robot_rad)
                _ = ax.add_artist(robot)
                _ = ax.add_collection(lc_walls)
                
                _ = ax.add_collection(lc_sensors)
                _ = ax.plot(view[0], view[1], c='r')
                # xlim,ylim = ax.get_xlim(),ax.get_ylim()
                # _ = plt.xlim(xlim)
                # _ = plt.ylim(ylim)
                plt.pause(1e-40)
                
            # Update fitness
            # self.fitness.update(x, y, num_collisions)
            # Calculate new velocities
            
            if i*self.dT%1 == 0:
                Vl_old,Vr_old = Vl,Vr
                Vl, Vr = self.ann(weights, Vl, Vr, sens_distance)
            
            if max(sens_distance)-self.sens_range*2/3> 0: 
                col_dist = max(sens_distance)-self.sens_range*2/3
                if col_dist>1:
                    col_dist = 1
            else: 
                col_dist = 0
            fitness += (abs(Vl)+abs(Vr)) * (1-np.sqrt(abs(Vl-Vr)))*(1-col_dist)
            
        # Calculate final fitness:
        # fitness = self.fitness.calculate()
        # fitness = len(surface_covered) / (np.log(num_collisions + 1) + 1)
        # print('Fitness :',fitness/self.iter_sim)
        return len(surface_covered)*fitness/self.iter_sim

    def movement(self, Vl, Vr, pos, theta):

        l = self.robot_rad * 2
        dT = self.dT
        x, y = pos[0], pos[1]
        omega = (Vr - Vl) / l
        if Vl == Vr:
            V = (Vl + Vr) / 2
            x = x + V * np.cos(theta) * dT
            y = y + V * np.sin(theta) * dT
            theta = theta
        else:
            R = l / 2 * ((Vl + Vr) / (Vr - Vl))
            ICCx = x - R * np.sin(theta)
            ICCy = y + R * np.cos(theta)
            x = np.cos(omega * dT) * (x - ICCx) - np.sin(omega * dT) * (y - ICCy) + ICCx
            y = np.sin(omega * dT) * (x - ICCx) + np.cos(omega * dT) * (y - ICCy) + ICCy
            theta = theta + omega * dT
        pos_new = np.array([x, y])
        theta = theta % (2*np.pi)
        
        return pos_new, theta

    def collision(self, walls, pos):
        robot_rad = self.robot_rad

        for wall in walls:
            # Compute distance from position to linesegment (wall)
            x0, y0 = pos[0], pos[1]
            x1, y1, x2, y2 = wall[0, 0], wall[0, 1], wall[1, 0], wall[1, 1]
            
            px, py = x2 - x1, y2 - y1
            u = ((x0 - x1) * px + (y0 - y1) * py) / (px * px + py * py)
            if u > 1:
                u = 1
            elif u < 0:
                u = 0
            x, y = x1 + u * px, y1 + u * py
            dx, dy = x - x0, y - y0
            distance = np.sqrt(dx * dx + dy * dy)
            # print(distance)
            if distance < robot_rad:
                return True

    # Initialise positions for 12 sensors
    def init_sensors(self, pos, theta):
        eps = 0.0001
        robot_rad = self.robot_rad-eps
        sens_range = self.sens_range
        sensors = np.zeros([12, 2, 2])
        for i in range(len(sensors)):
            sensors[i] = [[pos[0] + np.cos(theta) * robot_rad,
                           pos[1] + np.sin(theta) * robot_rad],
                          [pos[0] + np.cos(theta) * (robot_rad + sens_range),
                           pos[1] + np.sin(theta) * (robot_rad + sens_range)]]
            theta += 1 / 6 * np.pi
        return sensors

    def wall_distance(self, sensors, walls):
        distance = np.zeros(12)
        sens_range = self.sens_range
        i = 0
        for sensor in sensors:

            for wall in walls:
                x1, y1, x2, y2 = sensor[0, 0], sensor[0, 1], sensor[1, 0], sensor[1, 1]
                x3, y3, x4, y4 = wall[0, 0], wall[0, 1], wall[1, 0], wall[1, 1]
                # Repair vertical wall/sensor
                if x1 == x2:
                    x1 += 0.000001
                if x3 == x4:
                    x3 += 0.000001
                if y1 == y2:
                    y1 += 0.000001
                if y3 == y4:
                    y3 += 0.000001
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
                    distance[i] = sens_range-(np.sqrt((x1 - Px) ** 2 + (y1 - Py) ** 2)) 
                    break
            i += 1
        
        return distance

    # (Sebas Higler)
    def ann(self, weights, v_left, v_right, sensor_output):
        """
        Neural network combining previous velocities and sensor distance outputs.
        (Trained) weights are multiplied with this combined vector.
        :param v_left: float [-1, 1]
        :param v_right: float [-1, 1]
        :param sensor_output: numpy array with shape (12,)
        :param weights: numpy matrix with shape (2, 14) and values [0, 1]
        :return: new velocities
        """
        layers = weights.shape[0]

        # append v_left and v_right to sensor_output and set correct shape
        input_vector = np.append(sensor_output, [v_left, v_right,1])
        # Calculate 10 node hidden layer with sigmoid activation
        for l in range(layers - 1):
            input_vector = 1 / (1 + np.exp(-np.matmul(input_vector, weights[l])))
        # Calculate output nodes by hyperbolic tangent activation
        # output = np.tanh(np.dot(input_vector, weights[layers - 1]))
        
        # multiply input_input vector by weights and put through tanh activation function
        output = 1 / (1 + np.exp(-np.matmul(input_vector, weights[layers - 1])))
        # return vector of 2x1; v_left = output[0][0] v_right = output[1][0]
        
        output = output-0.5        
        
        return output
