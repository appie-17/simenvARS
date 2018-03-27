import numpy as np

robot = [12,6]
theta = (1 / 4) *np.pi

beacons = np.array([
    (2.0001, 2)
    , (2, 2)
    , (0.5, -2)    
])







#Input approximated beacon location according to sensor measurements
def triangulation(beacons):
    alphas = []
    for b in beacons:    
        alphas.append(
            np.arctan2(robot[1]-b[1], robot[0]-b[0]))

    alpha1, alpha2, alpha3 = alphas
    x1,y1,x2,y2,x3,y3 = np.array(beacons).flatten()
    
    alpha12 = alpha2 - alpha1
    alpha23 = alpha3 - alpha2 

    d12 = np.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    d23 = np.sqrt( (x3-x2)*(x3-x2) + (y3-y2)*(y3-y2))

    delta12 = np.arctan2( (y2-y1), (x2-x1))
    ra = d12 / (2 * np.sin(alpha12))
    cax =  x1 - ra*np.sin(delta12 - alpha12)
    cay = y1 + ra * np.cos(delta12-alpha12)

    delta23 = np.arctan2(y3-y2, x3-x2)
    rb = d23 / (2*np.sin(alpha23))
    cbx = x2 - rb* np.sin(delta23-alpha23)
    cby = y2 + rb*np.cos(delta23-alpha23)

    d = np.sqrt((cbx-cax)*(cbx-cax) + (cby-cay)*(cby-cay))

    d2r = (rb*rb + d*d - ra*ra) / (2*d)

    gamma = np.arccos(d2r/rb)

    phi = np.arctan2((cay-cby), (cax-cbx))

    R1x = cbx + rb * np.cos(gamma-phi)
    R1y = cby - rb * np.sin(gamma-phi)

    v1 = np.sqrt( (R1x-x2)*(R1x-x2) + (R1y-y2)*(R1y-y2))

    R2x = cbx + rb * np.cos( phi + gamma )
    R2y = cby + rb * np.sin( phi + gamma )

    v2 = np.sqrt( (R2x-x2)*(R2x-x2) + (R2y-y2)*(R2y-y2) ) 
    if( v1 > v2 ):
        x = R1x
        y = R1y
          
    else:
        x = R2x
        y = R2y
    state_ = [x, y]
    return state_

def sensorMeasurement( pos, theta):
    sensorModel, landmark_id = [], 0
    x, y = pos[0], pos[1]
    # beacons = np.random.choice(self.landmarksInSight,3,replace=False)

    for landmark in beacons:

        r = np.sqrt((landmark[0] - x) ** 2 + (landmark[1] - y) ** 2) #+ np.random.normal(0, self.r_std)
        phi = np.arctan2(landmark[1] - y, landmark[0] - x)-theta #+ np.random.normal(0, self.phi_std)

        beacons[landmark_id] = [r*np.cos(phi), r*np.sin(phi)]
        landmark_id += 1
    sensorModel = np.append(triangulation(beacons),theta)
    return np.array(sensorModel)
pos, theta = [0,0], np.pi

