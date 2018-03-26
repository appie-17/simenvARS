import numpy as np
from Simulation import Simulation

# Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
env_range = 20
robot_rad = 1
sens_range = 3
dT = 1/3
iter_sim = 1000

def tokenizer(fname):
    with open(fname) as f:
            weights= []
            for line in f:
                    if 'end_layer' in line:
                            yield weights                            
                            weights = []
                            continue
                    weights.append(line)
                    
#Test on Map 1
# sim_map = np.load('Maps/'+'Map1'+'.npy')
# locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6]])
# pos = locations[np.random.randint(locations.shape[0])]   
# weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map1.txt')])
# sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
# sim.simulate(weights, sim_map, pos, localiz=1)

#Test on Map 2
# sim_map = np.load('Maps/'+'Map2'+'.npy')
# locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
# pos = locations[np.random.randint(locations.shape[0])]   
# weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map2.txt')])
# sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
# sim.simulate(weights, sim_map, pos, localiz=1)

# Test on Map 4
sim_map = np.load('Maps/'+'Map4'+'.npy')
locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
pos = locations[np.random.randint(locations.shape[0])]   
# pos = np.array([6,6])
weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map4.txt')])
sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
sim.simulate(weights, sim_map, pos,localiz=1)
#Test on any map
sim_map = np.load('Maps/'+'Map3'+'.npy')
locations = np.array([[3,5],[17.5,3],[3,17]])
pos = locations[np.random.randint(locations.shape[0])]   
weights = np.array(
[[[-1.04746922,  1.41313515],
  [ 0.07980493, -0.71853415],
  [-0.89241298,  0.30547098],
  [-0.36028469,  0.82745653],
  [-2.42469271, -0.10056928],
  [ 1.64762453, -0.70256259],
  [ 1.60205111,  0.09800635],
  [ 4.09033603, -0.30117652],
  [ 0.91532369, -0.64346112],
  [-0.56963408, -2.22844021],
  [-2.67422521, -0.39716665],
  [-0.82585395, -1.48817701],
  [-1.70141903,  1.44971092],
  [ 0.39313447,  1.32933255],
  [ 0.29463374, -0.56627154]]]
	)
sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
sim.simulate(weights, sim_map, pos,localiz=1)




	

