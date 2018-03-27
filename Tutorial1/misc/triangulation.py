import math
import numpy as np

robot = [2, 2]
theta = (1 / 4) * math.pi

beacons = [
    [1, 5]
    , [5, 1]
    , [33, 2]
    # , [4, 0]
    # , [2, 0]
]

alphas = []

for b in beacons:
    if b[0] == robot[0]:
        b[0] += 0.000001
    if b[1] == robot[1]:
        b[0] += 0.000001
    alphas.append(
        (math.atan2(b[1] - robot[1], b[0] - robot[0]) - theta) % (math.pi * 2)
    )

# print(alphas)

alpha12 = alphas[1] - alphas[0]
alpha23 = alphas[2] - alphas[1]
alpha31 = alphas[0] - alphas[2]

print(alpha12, alpha23, alpha31)

x_prime1 = beacons[0][0] - beacons[1][0]
y_prime1 = beacons[0][1] - beacons[1][1]
x_prime3 = beacons[2][0] - beacons[1][0]
y_prime3 = beacons[2][1] - beacons[1][1]

t12 = 1 / math.tan(alphas[1] - alphas[0])
t23 = 1 / math.tan(alphas[2] - alphas[1])
t31 = (1 - (t12 * t23)) / (t12 + t23)

x_prime12 = x_prime1 + t12 * y_prime1
y_prime12 = y_prime1 - t12 * x_prime1

x_prime23 = x_prime3 + t23 * y_prime3
y_prime23 = y_prime3 - t23 * x_prime3

x_prime31 = (x_prime3 + x_prime1) + t31 * (y_prime3 - y_prime1)
y_prime31 = (y_prime3 + y_prime1) - t31 * (x_prime3 - x_prime1)

k_prime31 = (x_prime1 * x_prime3) + (y_prime1 * y_prime3) + t31 * ((x_prime1 * y_prime3) - (x_prime3 * y_prime1))

d = ((x_prime12 - x_prime23) * (y_prime23 - y_prime31)) - \
    ((y_prime12 - y_prime23) * (x_prime23 - x_prime31))

print("d =", d)

if d == 0:
    raise Exception("d not allowed to be 0")

x_r = beacons[1][0] + ((k_prime31 * (y_prime12 - y_prime23)) / d)
y_r = beacons[1][1] + ((k_prime31 * (x_prime23 - x_prime12)) / d)

print(x_r, y_r)


def sensor_measurement_triangulation(self, pos, theta):
    sensorModel, landmark_id = [], 0
    x, y = pos[0], pos[1]

    for landmark in self.landmarksInSight:
        r = np.sqrt((landmark[0] - x) ** 2 + (landmark[1] - y) ** 2) + np.random.normal(0, 0.01)

        phi = np.arctan2(landmark[1] - y, landmark[0] - x) + np.random.normal(0, 0.01)

        sensorModel.append([r, phi, landmark_id])

    # output [x, y, theta]
    return np.array(sensorModel)
