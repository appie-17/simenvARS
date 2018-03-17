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
# sim.simulate(weights)
weights = np.array(
[[[-2.863194996236616552e-01,-5.823901804348237121e+00],
[-1.931146305707156507e+00,-3.195205535850386802e+00],
[-8.422625530906630020e-01,-5.371829560859618269e+00],
[1.714799978548887438e+00,2.442306377191790556e+00],
[1.267571490148619695e+00,-6.553947981063446449e+00],
[2.279013915764636078e+00,3.029129398929800487e+00],
[-8.426655270297324085e-01,-4.698829669489203553e+00],
[-2.300814151698820920e-01,-3.198063517154974722e+00],
[-3.076148972809680249e+00,-8.542100506608720778e-01],
[2.643654715807363664e-01,-1.966232965697041868e+00],
[3.916939724099173681e+00,6.777286081524553918e-01],
[1.961296425645149499e+00,-1.418407684621802778e+00],
[5.267982904095528163e+00,1.189870442739433404e-01],
[5.474931843191590097e+00,3.597182201640269916e-01],
[-1.653308185480586223e+00,-8.131844447358332673e+00]]]
)

sim.simulate(weights)


	

