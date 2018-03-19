import numpy as np

class kalmanFilter:
	def __init__(self,map,pos,theta,error_params):
		self.landmarks = map.reshape(map.shape[0]*map.shape[1],2)
		self.error_params = error_params
		self.state = np.array([pos[0],pos[1],theta])
		self.state_cov = np.zeros([3,3])
	
	def updateKalmanFilter(self,pos,theta):

		A, B = [np.identity(3) for _ in range(2)]
		
		R, Q = [np.identity(3) *0.01 for _ in range(2)]
		mu, sigma = self.state, self.state_cov
		u = self.sampleOdometryMotion(pos,theta)

		z_hat = self.landmarkMeasurement()
		z = self.sensorMeasurement(pos,theta)
		
	#Prediction
		mu = np.matmul(A,mu) + np.matmul(B,u)
		sigma = A*sigma*A.transpose() + R
	#Correction
		for i in range(len(z)):
			
			gamma = np.random.rand()*2*np.pi
			C = np.array([[np.cos(gamma),0,0],[np.sin(gamma),0,0],[0,-1,0]])
			K = sigma*C.transpose()*np.linalg.inv(C*sigma*C.transpose()+Q)
			
			mu = mu + np.matmul(K,(z[i]-z_hat[i])) 
			sigma = (np.identity(3)-K*C)*sigma	
		
		
		self.state, self.state_cov = mu, sigma
		
	def sampleKalmanFilter(self):
		return np.random.multivariate_normal(self.state,self.state_cov)

	def odometryMotion(self,pos,theta):
		x_, y_, theta_ = self.state[0],self.state[1],self.state[2]
		
		x, y = pos[0], pos[1]
		
		dRot1 = np.arctan2(y-y_,x-x_)-theta_
		dRot2 = theta-theta_-dRot1
		dTrans = np.sqrt((x-x_)**2 + (y-y_)**2)

		return dRot1, dRot2, dTrans

	def sampleOdometryMotion(self,pos,theta):

		dRot1,dRot2,dTrans = self.odometryMotion(pos,theta)
		
		x_,y_,theta_ = self.state[0],self.state[1],self.state[2]

		alpha1,alpha2,alpha3,alpha4 = [self.error_params[i] for i in range(4)]
		
		dRot1 = dRot1 - np.random.normal(0,alpha1*abs(dRot1)+alpha2*dTrans)
		
		dRot2 = dRot2 - np.random.normal(0,alpha1*abs(dRot2)+alpha2*dTrans)

		dTrans = dTrans - np.random.normal(0,alpha3*dTrans+alpha4*(abs(dRot1)+abs(dRot2)))

		dx =  dTrans*np.cos(theta+dRot1)
		dy =  dTrans*np.sin(theta+dRot1)
		dtheta =  (dRot1 + dRot2)
		
		state = np.array([dx,dy,dtheta])
		
		return state

	def landmarkMeasurement(self):
		landmarkModel,landmark_id = [], 0
		x_, y_, theta_ = self.state[0], self.state[1], self.state[2]
		
		for landmark in self.landmarks:
			
			r = np.sqrt((landmark[0]-x_)**2+(landmark[1]-y_)**2)
			phi = np.arctan2(landmark[1]-y_,landmark[0]-x_)
			
			landmarkModel.append([r,phi,landmark_id])

		return np.array(landmarkModel)

	def sensorMeasurement(self, pos, theta):
		sensorModel,landmark_id = [], 0
		x, y = pos[0], pos[1]

		
		for landmark in self.landmarks:			
			
			r = np.sqrt((landmark[0]-x)**2+(landmark[1]-y)**2) + np.random.normal(0,0.01)
			
			phi = np.arctan2(landmark[1]-y,landmark[0]-x) + np.random.normal(0,0.01)
			
			sensorModel.append([r,phi,landmark_id])
		
		return np.array(sensorModel)