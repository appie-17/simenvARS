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
# sim.simulate(weights, sim_map, pos)

#Test on Map 2
# sim_map = np.load('Maps/'+'Map2'+'.npy')
# locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
# pos = locations[np.random.randint(locations.shape[0])]   
# weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map2.txt')])
# sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
# sim.simulate(weights, sim_map, pos)

#Test on any map
sim_map = np.load('Maps/'+'Map4'+'.npy')
locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
pos = locations[np.random.randint(locations.shape[0])]   
weights = np.array(
[[[ -5.54532718e-01 ,  3.34266635e-01],
  [ -2.50167844e+00 ,  7.13613799e-01],
  [ -3.18377032e-01 ,  1.44594677e+00],
  [ -2.11074574e+00 ,  2.44124541e-03],
  [ -1.35192283e+00 ,  3.85418016e+00],
  [ -1.01565586e+00 ,  1.89812350e+00],
  [  1.11408965e+00 , -9.25233653e-01],
  [  1.53478820e+00 , -2.90552858e-01],
  [  5.01485139e-01 , -4.93053305e+00],
  [ -5.02201612e+00 , -7.98786593e+00],
  [ -1.77896485e+00 , -3.06744917e-01],
  [ -5.54390872e-02  , 8.96889934e-01],
  [  1.21510163e+00 ,  6.18756769e-01],
  [ -5.25597262e-01 ,  8.99705477e-01],
  [ -8.43220577e-02 , -8.10100325e-01]]]
	)
sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
sim.simulate(weights, sim_map, pos)




	

