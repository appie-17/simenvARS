import numpy as np
from Simulation import Simulation

env_range = 20


# Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
robot_rad = 1
sens_range = 3
dT = 1/3
iter_sim = 1000
#Define datetime of directory with saved weights
datetime = '1520520627'
sim_map = np.load('Maps/'+'Map1'+'.npy')
locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
pos = locations[np.random.randint(locations.shape[0])]   
# pos = np.array([-6,-6])
sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT, None,sim_map,graphics = True)

def tokenizer(fname):
    with open(fname) as f:
            weights= []
            for line in f:
                    if 'end_layer' in line:
                            yield weights                            
                            weights = []
                            continue
                    weights.append(line)
                    
weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights.txt')])
sim.simulate(weights)
weights = np.array(
[[[ 0.21883954,0.00909977]
,[-1.51675099,0.07377015]
,[-0.17709574,0.33242105]
,[ 1.94879141,-1.0214378 ]
,[-0.05296046,-0.87517783]
,[ 1.72608846,-1.72454358]
,[-0.9792224,-1.35882227]
,[ 0.32527566,-0.92881706]
,[-1.66015946,3.71900956]
,[ 1.26785087,-1.46103617]
,[ 1.51995277,-0.73618862]
,[ 0.91361177,-1.42997914]
,[ 1.41564102,-1.53094714]
,[ 2.44230664,-2.00473708]
,[-1.92879941,-5.05219016]]]
)

sim.simulate(weights)


	

