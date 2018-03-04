import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.colors import ListedColormap, BoundaryNorm

#Check if point c lies between line (a,b)
def collision(walls,pos):
	eps = 0.02
	for wall in walls:
		ab = np.sqrt((wall[0,0] - wall[1,0])**2 + (wall[0,1] - wall[1,1])**2)
		
		ac = np.sqrt((wall[0,0] - pos[0]-robot_rad)**2 + (wall[0,1] - pos[1]-robot_rad)**2)
		cb = np.sqrt((wall[1,0] - pos[0]-robot_rad)**2 + (wall[1,1] - pos[1]-robot_rad)**2)
		if (ac+cb <= ab+eps) & (ac+cb >= ab-eps):
			return True
		
		ac2 = np.sqrt((wall[0,0] - pos[0]+robot_rad)**2 + (wall[0,1] - pos[1]+robot_rad)**2)
		cb2 = np.sqrt((wall[1,0] - pos[0]+robot_rad)**2 + (wall[1,1] - pos[1]+robot_rad)**2)
		if (ac2+cb2 <= ab+eps) & (ac2+cb2 >= ab-eps):
			return True
		ac3 = np.sqrt((wall[0,0] - pos[0]+robot_rad)**2 + (wall[0,1] - pos[1]-robot_rad)**2)
		cb3 = np.sqrt((wall[1,0] - pos[0]+robot_rad)**2 + (wall[1,1] - pos[1]-robot_rad)**2)
		if (ac3+cb3 <= ab+eps) & (ac3+cb3 >= ab-eps):
			return True
		ac4 = np.sqrt((wall[0,0] - pos[0]-robot_rad)**2 + (wall[0,1] - pos[1]+robot_rad)**2)
		cb4 = np.sqrt((wall[1,0] - pos[0]-robot_rad)**2 + (wall[1,1] - pos[1]+robot_rad)**2)
		if (ac4+cb4 <= ab+eps) & (ac4+cb4 >= ab-eps):
			return True

#Initialise positions for 12 sensors
def init_sensors(pos,theta):
	sensors = np.zeros([12,2,2])	
	for i in range(len(sensors)):
		sensors[i] = [[pos[0] + np.sin(theta)*robot_rad, 
		pos[1] + np.cos(theta)*robot_rad],
		[pos[0] + np.sin(theta)*(robot_rad+sens_range), 
		pos[1] + np.cos(theta)*(robot_rad+sens_range)]]
		theta += 1/6 * np.pi
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
	l = robot_rad*2
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
	pos_new = [x, y]
	return pos_new, theta

def ann(v_left, v_right, sensor_output):
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
    input_vector = np.reshape(np.append(sensor_output, [v_left, v_right]), (14, 1))
    # multiply input_input vector by weights and put through tanh activation function
    output = np.tanh(np.dot(weights, input_vector))
    # return vector of 2x1; v_left = output[0][0] v_right = output[1][0]
    return output

'''
Parameters to setup simulation for cleaning robot
'''
#Define range and starting point within square polygon environment
env_range = 10
pos = [2,8]
#Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
robot_rad = 0.25
sens_range = 0.5
dT = 0.1

np.random.seed(5)
#Initialise weights for ann
weights = np.random.rand(2, 14)
#Initialise velocities for right and left wheel of robot
Vl = np.random.randint(0,11)/100
Vr = np.random.randint(0,11)/100
# Vr = Vl

#Set theta to viewing direction of robot
theta = np.arctan(pos[1] / pos[0])

#add walls to 4x2x2d array, giving start- & end-coordinates 
#for each wall surrounding the environment
walls = np.array([[[0,0],[0,env_range]]])
walls = np.vstack((walls,np.array([[[0,0],[env_range,0]]])))
walls = np.vstack((walls,np.array([[[env_range,0],[env_range,env_range]]])))
walls = np.vstack((walls,np.array([[[0,env_range],[env_range,env_range]]])))

# Run simulation
plt.ion()
ax = plt.subplot(111)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
lc_walls = mc.LineCollection(walls)

for i in range(2000):
	#Calculate new position and viewing angle according to velocities
	pos, theta = movement(Vl,Vr,pos,theta)
	#When collision neglect movement orthogonal to wall only allow parralel movement
	while collision(walls,pos):
	# if collision(walls,pos):
		theta += 1/2*np.pi
		print('collision')
		pos, theta = movement(Vl,Vr,pos,theta)
	#Define 12 sensors each separated by 30 deg,2pi/12rad and calculate distance to any object
	sensors = init_sensors(pos,theta)
	sens_distance = wall_distance(sensors,walls)
	ax.clear()
	_ = plt.xlim(-2,12)
	_ = plt.ylim(-2,12)
	robot = plt.Circle(pos,robot_rad)
	linecolors = ['red' if i==0 else 'blue' for i in range(12)]
	lc_sensors = mc.LineCollection(sensors,colors = linecolors)
	_ = ax.add_artist(robot)
	_ = ax.add_collection(lc_walls)
	_ = ax.add_collection(lc_sensors)
	plt.pause(1e-40)
	print(sens_distance)
	#When 1/dT=1 run controller and calculate new velocities according to old vel. and sensor output
	if (i*dT)%1 == 0:
		print(Vl,Vr)
	# Vl, Vr = ann(Vl,Vr,sens_distance)