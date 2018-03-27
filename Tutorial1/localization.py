import numpy as np


from f_LinePoint import Line, Point, intersect_segments
class localization:
	def __init__(self):
		pass
		
	def sampleFilter(self):
		return np.random.multivariate_normal(self.state, self.state_cov)
    # Andrea Sica
	def updateLandmarkInSight(self, pos, max_distance=100):
		self.landmarksInSight = []
		for mark in self.landmarks:
			markPoint = Point(mark[0], mark[1])
			d = np.hypot(mark[0] - pos[0], mark[1] - pos[1])
			if d > max_distance:
				continue

			line = Line(Point(pos[0],pos[1]), markPoint)
			in_sign = True
			for wall in self.map:
				if np.array_equal(mark, wall[0]) or np.array_equal(mark, wall[1]):
					continue
				if intersect_segments(line, Line(Point(wall[0][0], wall[0][1]), Point(wall[1][0], wall[1][1]))) is not None:
					in_sign = False
					break
			if in_sign:
				self.landmarksInSight.append(mark)
    # Ruhui Zhao
	def updateOdometryMotion(self, pos, theta):

		def odometryMotion(pos, theta):
			x_, y_, theta_ = self.state[0], self.state[1], self.state[2]
			
			x, y = pos[0], pos[1]
			dRot1 = np.arctan2(y - y_, x - x_) - theta_
			dTrans = np.sqrt((x_ - x) ** 2 + (y_ - y) ** 2)
			dRot2 = theta - theta_ - dRot1
			return dRot1, dTrans, dRot2

		dRot1, dTrans, dRot2 = odometryMotion(pos, theta)

		alpha1, alpha2, alpha3, alpha4 = [self.error_params[i] for i in range(4)]
		
		error = np.random.normal(0, alpha1*abs(dRot1) + alpha2*abs(dTrans))
		
		
		dRot1 = dRot1 - error						
		dTrans = dTrans - np.random.normal(0, (alpha3 * abs(dTrans) + alpha4 * (abs(dRot1+dRot2))))
		dRot2 = dRot2 - np.random.normal(0, (alpha1 * abs(dRot2) + alpha2 * abs(dTrans)))
		dx = dTrans * np.cos(theta + dRot1)
		dy = dTrans * np.sin(theta + dRot1)
		dtheta = dRot1 + dRot2
		
		self.state = np.array(self.state) + np.array([dx, dy, dtheta])		
		self.state_cov = self.state_cov + self.R

# Sebas Higler and Jan Lucas
class kalmanFilter(localization):
    def __init__(self, map, pos, theta, alpha1,alpha2,alpha3,alpha4,r_std,phi_std):
        self.landmarks = map.reshape(map.shape[0] * map.shape[1], 2)
        self.landmarks = [list(x) for x in set(tuple(x) for x in self.landmarks)]
        self.landmarksInSight = []
        self.map = map

        self.error_params = [alpha1,alpha2,alpha3,alpha4]
        self.state = np.array([pos[0], pos[1], theta])
        self.state_cov = np.diag([999. for i in range(3)])
        self.R = np.diag([1,1,1])*1000
        if (r_std > 0.00) & (phi_std>0.00): self.Q = np.diag([r_std,phi_std,1])
        else: self.Q = np.identity(3)*0.1
        self.r_std = r_std
        self.phi_std = phi_std

    def updateKalmanFilter(self, pos, theta):

        # Prediction
        self.updateOdometryMotion(pos, theta)        
        self.updateLandmarkInSight(pos)
                                       
        # Correction        
        z = self.sensorMeasurement(pos, theta)        
        print(self.state_cov)
    
        C = np.identity(3)
        K = self.state_cov.dot(C.T).dot(np.linalg.inv(C.dot(self.state_cov).dot(C.T) + self.Q))            

        self.state += K.dot(z-C.dot(self.state))
        self.state_cov = (np.identity(3) - K.dot(C)).dot(self.state_cov)
        

    def landmarkMeasurement(self):
        landmarkModel, landmark_id = np.array([]), 0
        x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

        for landmark in self.landmarksInSight:
            r = np.sqrt((landmark[0] - x_) ** 2 + (landmark[1] - y_) ** 2)
            phi = np.arctan2(landmark[1] - y_, landmark[0] - x_)

            landmarkModel = np.append(landmarkModel,np.array([x_,y_,theta_]).reshape(-1,1))
        return landmarkModel

    def triangulation(self,beacons,pos,theta):
        alphas = []
        for b in beacons:    
            alphas.append(
                np.arctan2(pos[1]-b[1], pos[0]-b[0]))

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
        state_ = np.array([x, y, theta])
        return state_

    def sensorMeasurement(self, pos, theta):
        sensorModel, landmark_id = [], 0
        x, y = pos[0], pos[1]
    
        beacons = np.array(self.landmarksInSight)
        beacons = beacons[np.random.choice(beacons.shape[0],3,replace=False)]

        for landmark in beacons:

            r = np.sqrt((landmark[0] - x) ** 2 + (landmark[1] - y) ** 2) + np.random.normal(0, self.r_std)
            phi = np.arctan2(landmark[1] - y, landmark[0] - x)-theta + np.random.normal(0, self.phi_std)

            beacons[landmark_id] = [r*np.cos(phi), r*np.sin(phi)]
            landmark_id += 1
        sensorModel = self.triangulation(beacons,pos,theta)
        return sensorModel    

# Jordy van Appeven
class extendedKalmanFilter(localization):
    def __init__(self, map, pos, theta, alpha1, alpha2, alpha3, alpha4, r_std, phi_std):
        self.landmarks = map.reshape(map.shape[0] * map.shape[1], 2)
        self.landmarks = [list(x) for x in set(tuple(x) for x in self.landmarks)]
        self.landmarksInSight = []
        self.map = map
        
        self.error_params = [alpha1,alpha2,alpha3,alpha4]
        self.state = np.array([pos[0], pos[1], theta])        
        self.state_cov = np.diag([999. for i in range(3)])
        self.R = np.diag([1,1,1])*100
        if (r_std > 0.00) & (phi_std>0.00): self.Q = np.diag([r_std,phi_std,0.01])
        else: self.Q = np.diag([1,1,1000])
        self.r_std = r_std
        self.phi_std = phi_std
        
    
    def updateKalmanFilter(self, pos, theta):
        
        # Prediction
        self.updateOdometryMotion(pos, theta)        
        self.updateLandmarkInSight(pos)
        
        # Correction
        z_hat,H = self.landmarkMeasurement()
        z = self.sensorMeasurement(pos, theta)
        dmu, dsigma = 0, 0
        print(self.state_cov)
        
        for i in range(len(z)):

            K = self.state_cov.dot(H[i].T).dot(np.linalg.inv(H[i].dot(self.state_cov).dot(H[i].T) + self.Q))
            # print(z[i], z_hat[i])            
            # dmu += K.dot(z[i]-z_hat[i])            
            # dsigma += K.dot(H[i])
            
            self.state += K.dot(z[i]-z_hat[i])                                   
            self.state_cov = (np.identity(3) - K.dot(H[i])).dot(self.state_cov)
            self.state_cov = np.diag(self.state_cov.diagonal())
            
        # self.state += dmu        
        # self.state_cov = (np.identity(3)-dsigma).dot(self.state_cov)
        # self.state_cov = np.diag(self.state_cov.diagonal())
        
    def sensorMeasurement(self, pos, theta):        
        sensorModel, landmark_id = [], 0
        x, y = pos[0], pos[1]

        for landmark in self.landmarksInSight:
            r = np.sqrt((landmark[0] - x) ** 2 + (landmark[1] - y) ** 2) + np.random.normal(0, self.r_std)
            phi = np.arctan2(landmark[1] - y, landmark[0] - x) -theta + np.random.normal(0, self.phi_std)
            
            sensorModel.append([r, phi, landmark_id])
            landmark_id += 1

        return np.array(sensorModel)

    def landmarkMeasurement(self):
        landmarkModel, H, landmark_id = [], [], 0
        x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

        for landmark in self.landmarksInSight:
            delta = np.array([landmark[0] - x_, landmark[1] - y_])
            q = delta.dot(delta)
            
            r = np.sqrt(q)

            phi = (np.arctan2(delta[1], delta[0]))-theta_
            landmarkModel.append([r, phi, landmark_id])
            landmark_id += 1

            H.append(np.array([[np.sqrt(q)*delta[0], - np.sqrt(q)*delta[1], 0],
            	[delta[1], delta[0], -1],
            	[0, 0, 0]]).dot(1/q))
            
        return np.array(landmarkModel), H