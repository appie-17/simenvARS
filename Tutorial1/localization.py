import numpy as np
from f_LinePoint import Line, Point, intersect_segments

class kalmanFilter:
    def __init__(self, map, pos, theta, alpha1,alpha2,alpha3,alpha4,r_std,phi_std):
        self.landmarks = map.reshape(map.shape[0] * map.shape[1], 2)
        self.landmarks = [list(x) for x in set(tuple(x) for x in self.landmarks)]

        self.landmarksInSight = self.landmarks
        self.map = map
        self.error_params = [alpha1,alpha2,alpha3,alpha4]
        self.state = np.array([pos[0], pos[1], theta])
        self.state_cov = np.diag([0.009 for i in range(3)])
        self.r_std = r_std
        self.phi_std = phi_std

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


    def updateKalmanFilter(self, pos, theta):
        self.updateLandmarkInSight(pos)
        A, B = [np.identity(3) for _ in range(2)]

        R = np.identity(3)
        Q = np.diag([self.r_std,self.phi_std,1]) #* np.array([self.r_std,self.phi_std,1])
        mu, sigma = self.state, self.state_cov
        u = self.sampleOdometryMotion(pos, theta)

        z_hat = self.landmarkMeasurement()
        z = self.sensorMeasurement(pos, theta)

        # Prediction
        # mu = np.matmul(A, mu) + np.matmul(B, u)
        mu = A.dot(mu) + B.dot(u)
        # sigma = A * sigma * A.transpose() + R
        sigma = A.dot(sigma).dot(A.T) + R

        # Correction
        d_mu, d_sigma = 0, 0

        for i in range(len(z)):
            gamma = np.random.rand() * 2 * np.pi
            # C = np.array([[np.cos(gamma), 0, 0], [np.sin(gamma),0, 0], [0, -1, 0]])

            C = np.identity(3)
            K = sigma.dot(C.T).dot(np.linalg.inv(C.dot(sigma).dot(C.T) + Q))
            # K = sigma * C.transpose() * np.linalg.inv(C * sigma * C.transpose() + Q)
            # print(sigma)
            # d_mu += K.dot(z[i]-C.dot(mu))
            # d_sigma += K.dot(C)

            mu = mu + K.dot(z[i]-C.dot(mu))
            sigma = (np.identity(3) - K.dot(C)).dot(sigma)
        # mu = mu + d_mu
        # sigma = (np.identity(3) - d_sigma).dot(sigma)
        print(sigma)
        

        self.state, self.state_cov = mu, sigma

    def sampleKalmanFilter(self):
        return np.random.multivariate_normal(self.state, self.state_cov)

    def odometryMotion(self, pos, theta):
        x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

        x, y = pos[0], pos[1]

        dRot1 = np.arctan2(y - y_, x - x_) - theta_
        dRot2 = theta - theta_ - dRot1
        dTrans = np.sqrt((x - x_) ** 2 + (y - y_) ** 2)

        return dRot1, dRot2, dTrans

    def sampleOdometryMotion(self, pos, theta):

        dRot1, dRot2, dTrans = self.odometryMotion(pos, theta)

        x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

        alpha1, alpha2, alpha3, alpha4 = [self.error_params[i] for i in range(4)]

        dRot1 = dRot1 - np.random.normal(0, alpha1 * abs(dRot1) + alpha2 * dTrans)

        dTrans = dTrans - np.random.normal(0, alpha3 * dTrans + alpha4 * (abs(dRot1) + abs(dRot2)))

        dRot2 = dRot2 - np.random.normal(0, alpha1 * abs(dRot2) + alpha2 * dTrans)

        dx = dTrans * np.cos(theta + dRot1)
        dy = dTrans * np.sin(theta + dRot1)
        dtheta = (dRot1 + dRot2)

        state = np.array([dx, dy, dtheta])

        return state

    def landmarkMeasurement(self):
        landmarkModel, landmark_id = np.array([]), 0
        x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

        for landmark in self.landmarksInSight:
            r = np.sqrt((landmark[0] - x_) ** 2 + (landmark[1] - y_) ** 2)
            phi = np.arctan2(landmark[1] - y_, landmark[0] - x_)

            # landmarkModel.append([r, phi, landmark_id])
            # landmarkModel.append([x_,y_,theta_].reshape(-1,1))
            landmarkModel = np.append(landmarkModel,np.array([x_,y_,theta_]).reshape(-1,1))
        return landmarkModel

    def sensorMeasurement(self, pos, theta):
        sensorModel, landmark_id = [], 0
        x, y = pos[0], pos[1]

        for landmark in self.landmarksInSight:
            r = np.sqrt((landmark[0] - x) ** 2 + (landmark[1] - y) ** 2) + np.random.normal(0, self.r_std)

            # phi = np.arctan2(landmark[1] - y, landmark[0] - x) + np.random.normal(0, self.phi_std)
            # gamma = np.random.rand()*2*np.pi
            # x = landmark[0] + r*np.cos(gamma) 
            # y = landmark[1] + r*np.sin(gamma)
            # theta = gamma - np.pi - phi
            # sensorModel.append([r, phi, landmark_id])
            sensorModel.append([x,y,theta])

            # sensorModel = np.append(sensorModel,np.array([x,y,theta]))

        return np.array(sensorModel)

class extendedKalmanFilter:
    def __init__(self, map, pos, theta, alpha1, alpha2, alpha3, alpha4, r_std, phi_std):
        self.landmarks = map.reshape(map.shape[0] * map.shape[1], 2)
        self.landmarks = [list(x) for x in set(tuple(x) for x in self.landmarks)]
        self.landmarksInSight = []
        self.map = map
        
        self.error_params = [alpha1,alpha2,alpha3,alpha4]
        self.state = np.array([pos[0], pos[1], theta])
        self.state_cov = np.diag([999. for i in range(3)])
        self.R = np.diag([1,1,1])*0.1
        self.Q = np.identity(3)
        self.r_std = r_std
        self.phi_std = phi_std
        # self.Q = np.diag([r_std,phi_std,1])
    
    def updateKalmanFilter(self, pos, theta):
        
        # Prediction
        self.updateOdometryMotion(pos, theta)
        # print(self.state)
        self.updateLandmarkInSight(pos)
        
        # Correction
        z_hat,H = self.landmarkMeasurement()
        z = self.sensorMeasurement(pos, theta)
        dmu, dsigma = 0, 0
        print(self.state_cov)
        for i in range(len(z)):

            K = self.state_cov.dot(H[i].T).dot(np.linalg.inv(H[i].dot(self.state_cov).dot(H[i].T) + self.Q))
            # print(z[i], z_hat[i])
            dmu += K.dot(z[i]-z_hat[i])
            dsigma += K.dot(H[i])
            self.state += K.dot(z[i]-z_hat[i])
            if self.state[2] <0: self.state[2] = self.state[2]%-(2*np.pi)
            else: self.state[2] = self.state[2]%(2*np.pi)
            self.state_cov = (np.identity(3) - K.dot(H[i])).dot(self.state_cov)
        
        # self.state += dmu
        # if self.state[2] <0: self.state[2] = self.state[2]%-(2*np.pi)
        # else: self.state[2] = self.state[2]%(2*np.pi)
        print((self.state))
        # self.state_cov = (np.identity(3)-dsigma).dot(self.state_cov)
        # print((np.identity(3)-dsigma).dot(self.state_cov))
        
        

    def sampleKalmanFilter(self):
        return np.random.multivariate_normal(self.state, self.state_cov)

    def updateOdometryMotion(self, pos, theta):

    	def odometryMotion(pos, theta):
    		x_, y_, theta_ = self.state[0], self.state[1], self.state[2]

    		x, y = pos[0], pos[1]
    		
    		dRot1 = np.arctan2(y - y_, x - x_) - theta_
    		dTrans = np.sqrt((x - x_) ** 2 + (y - y_) ** 2)
    		dRot2 = theta - theta_ - dRot1

    		return dRot1, dRot2, dTrans

    	dRot1, dRot2, dTrans = odometryMotion(pos, theta)
    	
    	alpha1, alpha2, alpha3, alpha4 = [self.error_params[i] for i in range(4)]

    	dRot1 = dRot1 - np.random.normal(0, (alpha1 * abs(dRot1) + alpha2 * dTrans))
    	
    	dTrans = dTrans - np.random.normal(0, (alpha3 * dTrans + alpha4 * (abs(dRot1) + abs(dRot2))))
    	dRot2 = dRot2 - np.random.normal(0, (alpha1 * abs(dRot2) + alpha2 * dTrans))
    	
    	dx = dTrans * np.cos(theta + dRot1)
    	dy = dTrans * np.sin(theta + dRot1)
    	dtheta = (dRot1 + dRot2)
    	
    	self.state = self.state + np.array([dx, dy, dtheta])
    	if self.state[2] <0: self.state[2] = self.state[2]%-(2*np.pi)
    	else: self.state[2] = self.state[2]%(2*np.pi)
    	# self.R = np.diag([dRot1**2,dTrans**2,dRot2**2])
    	self.state_cov += self.R
        
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
            # if phi <0: phi = phi%-(2*np.pi)
            # else: phi = phi%(2*np.pi)
            landmarkModel.append([r, phi, landmark_id])
            landmark_id += 1

            H.append(np.array([[np.sqrt(q)*delta[0], - np.sqrt(q)*delta[1], 0],
            	[delta[1], delta[0], -1],
            	[0, 0, 0]]).dot(1/q))
                        
        return np.array(landmarkModel), H