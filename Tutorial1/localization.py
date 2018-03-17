import numpy as np

class kalmanFilter:
	def __init__(self,map,pos,theta,error_params):
		self.landmarks = map.reshape(map.shape[0]*map.shape[1],2)
		self.error_params = error_params
		self.state = np.array([pos[0],pos[1],theta])
		self.state_cov = np.zeros([3,3])
	
	def updateKalmanFilter(self,pos,theta):
		A, B, C = [np.identity(3) for _ in range(3)]
		R, Q = [np.identity(3) for _ in range(2)]
		mu, sigma = self.state, self.state_cov
		u = self.sampleOdometryMotion(pos,theta)
		z = self.sampleSensorModel()
	#Prediction
		mu = np.matmul(A,mu) + np.matmul(B,u)
		sigma = A*sigma*A.transpose() + R
	#Correction
		K = sigma*C.transpose()*np.linalg.inv(C*sigma*C.transpose()+Q)
		for observation in z:
			mu = mu + np.matmul(K,(observation-np.matmul(C,mu)))
		sigma = (np.identity(3)-K*C)*sigma

		self.state, self.state_cov = mu, sigma
		
	def sampleKalmanFilter(self):
		return np.random.multivariate_normal(self.state,self.state_cov)

	def odometryMotion(self,pos,theta):
		x, y, theta = self.state[0],self.state[1],self.state[2]
		x_, y_, theta_ = pos[0], pos[1], theta

		dRot1 = np.arctan2(y_-y,x_-x)-theta
		dRot2 = theta_-theta-dRot1
		dTrans = np.sqrt((x_-x)**2 + (y_-y)**2)

		return dRot1, dRot2, dTrans

	def sampleOdometryMotion(self,pos,theta):
		dRot1,dRot2,dTrans = self.odometryMotion(pos,theta)
		
		x,y,theta = self.state[0],self.state[1],self.state[2]

		alpha1,alpha2,alpha3,alpha4 = [self.error_params[i] for i in range(4)]
		
		dRot1 = dRot1 + np.random.normal(0,alpha1*abs(dRot1)+alpha2*dTrans)

		dRot2 = dRot2 + np.random.normal(0,alpha1*abs(dRot2)+alpha2*dTrans)

		dTrans = dTrans + np.random.normal(0,alpha3*dTrans+alpha4*(abs(dRot1)+abs(dRot2)))

		x = x + dTrans*np.cos(theta+dRot1)
		y = y + dTrans*np.sin(theta+dRot1)
		theta = theta + dRot1 + dRot2
		
		state = np.array([x,y,theta])
		
		return state

	def sensorModel(self):
		distance,landmark_id = [], 0
		x, y, theta = self.state[0], self.state[1], self.state[2]
		for landmark in self.landmarks:
			distance.append([landmark_id,np.sqrt((landmark[0]-x)**2+(landmark[1]-y)**2),
			np.arctan2(landmark[1]-y,landmark[0]-x)])
			landmark_id += 1
		return distance

	def sampleSensorModel(self):
		landmarks_dist = self.sensorModel()
		states= []
		for landmark in landmarks_dist:
			
			gamma = np.random.rand()*2*np.pi
			r = landmark[0] + np.random.normal(0,1)
			phi = landmark[1] + np.random.normal(0,1)
			
			x = self.landmarks[landmark[0]][0] + r*np.cos(gamma)
			y = self.landmarks[landmark[0]][1] + r*np.sin(gamma)
			theta = gamma - np.pi - phi
			
			states.append([x,y,theta])

		return states