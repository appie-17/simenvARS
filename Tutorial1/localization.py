import numpy as np


from f_LinePoint import Line, Point, intersect_segments
class localization:
	def __init__(self):
		pass
		
	def sampleFilter(self):
		return np.random.multivariate_normal(self.state, self.state_cov)

	def updateLandmarkInSight(self, pos, max_distance=50):
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
		# self.state[2] = self.state[2]%(2*np.pi)

		# if self.state[2] <0: self.state[2] = self.state[2]%-(2*np.pi)
		# else: self.state[2] = self.state[2]%(2*np.pi)
		self.state_cov = self.state_cov + self.R
		
class kalmanFilter(localization):
    def __init__(self, map, pos, theta, alpha1,alpha2,alpha3,alpha4,r_std,phi_std):
        self.landmarks = map.reshape(map.shape[0] * map.shape[1], 2)
        self.landmarks = [list(x) for x in set(tuple(x) for x in self.landmarks)]
        self.landmarksInSight = []
        self.map = map

        self.error_params = [alpha1,alpha2,alpha3,alpha4]
        self.state = np.array([pos[0], pos[1], theta])
        self.state_cov = np.diag([999. for i in range(3)])
        self.R = np.diag([1,1,1])
        if (r_std > 0.00) & (phi_std>0.00): self.Q = np.diag([r_std,phi_std,1])
        else: self.Q = np.identity(3)
        self.r_std = r_std
        self.phi_std = phi_std

    def updateKalmanFilter(self, pos, theta):

        # Prediction
        self.updateOdometryMotion(pos, theta)        
        self.updateLandmarkInSight(pos)
                                       
        # Correction
        z_hat = self.landmarkMeasurement()
        z = self.sensorMeasurement(pos, theta)
        Q = np.diag([self.r_std,self.phi_std,1]) #* np.array([self.r_std,self.phi_std,1])        
        print(self.state_cov)
        d_mu, d_sigma = 0, 0

        for i in range(len(z)):

            C = np.identity(3)
            K = self.state_cov.dot(C.T).dot(np.linalg.inv(C.dot(self.state_cov).dot(C.T) + Q))
            
            # d_mu += K.dot(z[i]-C.dot(self.state))
            # d_sigma += K.dot(C)

            self.state += K.dot(z[i]-C.dot(self.state))
            self.state_cov = (np.identity(3) - K.dot(C)).dot(self.state_cov)
        # self.state += d_mu
        # self.state_cov = (np.identity(3) - d_sigma).dot(self.state_cov)

    def landmarkMeasurement(self):
        landmarkModel, landmark_id = np.array([]), 0
        x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

        for landmark in self.landmarksInSight:
            r = np.sqrt((landmark[0] - x_) ** 2 + (landmark[1] - y_) ** 2)
            phi = np.arctan2(landmark[1] - y_, landmark[0] - x_)

            landmarkModel = np.append(landmarkModel,np.array([x_,y_,theta_]).reshape(-1,1))
        return landmarkModel

    def sensorMeasurement(self, pos, theta):
        sensorModel, landmark_id = [], 0
        x, y = pos[0], pos[1]

        for landmark in self.landmarksInSight:
            r = np.sqrt((landmark[0] - x) ** 2 + (landmark[1] - y) ** 2) + np.random.normal(0, self.r_std)
            phi = np.arctan2(landmark[1] - y, landmark[0] - x) -theta + np.random.normal(0, self.phi_std)

            sensorModel.append([x,y,theta]) 
            landmark_id += 1           

        return np.array(sensorModel)

class extendedKalmanFilter(localization):
    def __init__(self, map, pos, theta, alpha1, alpha2, alpha3, alpha4, r_std, phi_std):
        self.landmarks = map.reshape(map.shape[0] * map.shape[1], 2)
        self.landmarks = [list(x) for x in set(tuple(x) for x in self.landmarks)]
        self.landmarksInSight = []
        self.map = map
        
        self.error_params = [alpha1,alpha2,alpha3,alpha4]
        self.state = np.array([pos[0], pos[1], theta])
        # self.state = np.array([10.10,0.,5.0])
        self.state_cov = np.diag([999. for i in range(3)])
        self.R = np.diag([1,1,1])*100
        if (r_std > 0.00) & (phi_std>0.00): self.Q = np.diag([r_std,phi_std,0.01])
        else: self.Q = np.diag([1,1,1000])*10
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
        # print(self.state_cov)
        
        for i in range(len(z)):

            K = self.state_cov.dot(H[i].T).dot(np.linalg.inv(H[i].dot(self.state_cov).dot(H[i].T) + self.Q))
            # print(z[i], z_hat[i])
            # print(K)
            # dmu += K.dot(z[i]-z_hat[i])            
            # dsigma += K.dot(H[i])
            
            self.state += K.dot(z[i]-z_hat[i])
            # self.state[2] = self.state[2]%(2*np.pi)
            # if self.state[2] <0: self.state[2] = self.state[2]%-(2*np.pi)
            # else: self.state[2] = self.state[2]%(2*np.pi)        
            self.state_cov = (np.identity(3) - K.dot(H[i])).dot(self.state_cov)
            self.state_cov = np.diag(self.state_cov.diagonal())
            # self.state_cov = self.state_cov - K.dot(H[i]).dot(self.state_cov)
        # self.state += dmu
        
        # self.state[2] = self.state[2]%(2*np.pi)
        
        # if self.state[2] <0: self.state[2] = self.state[2]%-(2*np.pi)
        # else: self.state[2] = self.state[2]%(2*np.pi)        
        # self.state_cov = (np.identity(3)-dsigma).dot(self.state_cov)
        # self.state_cov = np.diag(self.state_cov.diagonal())
        print(self.state_cov)
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