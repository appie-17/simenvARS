import numpy as np

#Check if point c lies between line (a,b)
def collision(walls,pos):
	eps = 0.01
	for wall in walls:
		ab = np.sqrt((wall[0,0] - wall[1,0])**2 + (wall[0,1] - wall[1,1])**2)
		
		ac = np.sqrt((wall[0,0] - pos[0]-robot_rad)**2 + (wall[0,1] - pos[1]-robot_rad)**2)
		cb = np.sqrt((wall[1,0] - pos[0]-robot_rad)**2 + (wall[1,1] -pos[1]-robot_rad)**2)
		if (ac+cb <= ab+eps) & (ac+cb >= ab-eps):
			return True
		
		ac2 = np.sqrt((wall[0,0] - pos[0]+robot_rad)**2 + (wall[0,1] - pos[1]+robot_rad)**2)
		cb2 = np.sqrt((wall[1,0] - pos[0]+robot_rad)**2 + (wall[1,1] - pos[1]+robot_rad)**2)
		if (ac2+cb2 <= ab+eps) & (ac2+cb2 >= ab-eps):
			return True
		

def movement(x,y,R,omega, theta):
		deltaT = 1
		
		ICCx = x - R*np.sin(theta)
		ICCy = y + R*np.cos(theta)
		x = np.cos(omega*deltaT) * (x-ICCx) + ICCx
		y = np.sin(omega*deltaT) * (y-ICCy) + ICCy
		theta = theta + omega*deltaT 
		pos_new = [x,y]
		angle = theta
		return pos_new, angle


#Velicty left- and right wheel between [-1,1]

#Initialise positions for 12 sensors, still need to apply omega as start angle
def init_sensors(pos,omega):
	sensors = np.zeros([12,2,2])
	unit_circle = 0
	for i in range(len(sensors)):
		sensors[i] = [[pos[0] + np.sin(unit_circle*2*np.pi)*robot_rad, 
		pos[1] + np.cos(unit_circle*2*np.pi)*robot_rad],
		[pos[0] + np.sin(unit_circle*2*np.pi)*(robot_rad+sens_range), 
		pos[1] + np.cos(unit_circle*2*np.pi)*(robot_rad+sens_range)]]
		unit_circle += 1/12
	return sensors

def wall_distance(sensors,walls):
	distance = np.zeros(12)
	for sensor in sensors:
		i = 0
		for wall in walls:
			x1,y1,x2,y2 = sensor[0,0],sensor[0,1],sensor[1,0],sensor[1,1]
			x3,y3,x4,y4 = wall[0,0],wall[0,1],wall[1,0],wall[1,1]
			#Repair vertical wall/sensor
			if x1 == x2:
				x1 += 0.1
			if x3 == x4:
				x3 += 0.1
			#Calculate intersection point between wall and sensor line
			Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
			Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
			#Check for true intersection between walls and sensors
			if (Px <= np.minimum(x1,x2)) | (Px >= np.maximum(x1,x2)):
				break
			if (Px <= np.minimum(x3,x4)) | (Px >= np.maximum(x3,x4)):
				break
			distance[i] = np.sqrt((x1-Px)**2+(y1-Py)**2)
		i += 1
	return distance	

#Determine range of square environment polygon
env_range = 10
pos_start = [4,3]
robot_rad = 0.5
sens_range = 5
#add walls to 4x2x2d array, giving start- & end-coordinates 
#for each wall surrounding the environment
walls = np.array([[[0,0],[0,env_range]]])
walls = np.vstack((walls,np.array([[[0,0],[env_range,0]]])))
walls = np.vstack((walls,np.array([[[env_range,0],[env_range,env_range]]])))
walls = np.vstack((walls,np.array([[[0,env_range],[env_range,env_range]]])))

l = robot_rad
pos = pos_start
theta = np.arctan(pos[1]/pos[0])/(2*np.pi)*360
for _ in range(100):
	print(pos)
	Vl = np.random.rand()-0.5
	Vr = np.random.rand()-0.5
	R = l/2 * (Vl+Vr)/(Vr-Vl)
	omega = (Vr-Vl)/l
	pos, angle = movement(pos[0],pos[1],R,omega,theta)
	if collision(walls,pos):
		print('collision')
		Vl = 0
		Vr = 0
	sensors = init_sensors(pos,omega)
	sens_distance = wall_distance(sensors,walls)
	print(sens_distance)