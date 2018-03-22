import numpy as np

class Line(object):
    a = None
    m = None # Slope
    P2 = None
    def __init__(self, P1, P2, Slope=None):
        self.x = P1.x # for short after the call to x and y
        self.y = P1.y
        self.P1 = P1
        self.P2 = P2

        # None is a type of slope! and is different from 0
        # But still the parameter can be omitted in the function, so I recalculate it anyway
        if(P2 is not None and Slope is None):
            Slope = slope(P1, P2)

        if Slope is not None:
            self.m = Slope
            self.calculate_abc()

    def calculate_abc(self):
        self.a = -self.m
        self.b = 1
        self.c = self.m*self.P1.x - self.P1.y

    def get_y_intersect(self, X): # when one of the 2 slope are undefined because it is vertical
        if self.m is None:
            return None # just in case
        return (-self.c-self.a*X)/self.b

    def distance(self, P): # only used for rays!!
        return math.hypot(P.x - self.P1.x, P.y - self.P1.y)

    def intersect_circle(self, pos_circle, radius):
        if self.m is None: # the line is horizontal, no Slope then, you have to check in another way
            if (pos_circle.x-radius <= self.x <= pos_circle.x+radius) and (
                min(self.P1.y, self.P2.y) < pos_circle.y + radius < max(self.P1.y, self.P2.y) or
                min(self.P1.y, self.P2.y) < pos_circle.y - radius < max(self.P1.y, self.P2.y)
            ):
             return True

        else:
            nearest_point = self.nearest_line_point__to_point(pos_circle)

            dist = math.hypot(nearest_point.x - pos_circle.x, nearest_point.y - pos_circle.y)

            # Checking if the distance is less than,
            # greater than or equal to radius.
            if radius == dist:
                return True # print("Touch")
            elif radius > dist:
                return True  # print("Intersect ", dist)
            # else is Outside
        return False

    def segment_intersect_circle(self, pos_circle, r0): # r0 = radius
        # p is the circle parameter, lsp and lep is the two end of the line
        x0, y0 = pos_circle.x, pos_circle.y
        x1, y1 = self.P1.x, self.P1.y
        x2, y2 = self.P2.x, self.P2.y

        if x1 == x2:
            if abs(r0) >= abs(x1 - x0):
                p1 = x1, y0 - np.sqrt(r0 ** 2 - (x1 - x0) ** 2)
                p2 = x1, y0 + np.sqrt(r0 ** 2 - (x1 - x0) ** 2)
                inp = [p1, p2]
                # select the points lie on the line segment
                inp = [p for p in inp if p[1] >= min(y1, y2) and p[1] <= max(y1, y2)]
            else:
                inp = []
        else:
            k = (y1 - y2) / (x1 - x2)
            b0 = y1 - k * x1
            a = k ** 2 + 1
            b = 2 * k * (b0 - y0) - 2 * x0
            c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
            delta = b ** 2 - 4 * a * c
            if delta >= 0:
                p1x = (-b - np.sqrt(delta)) / (2 * a)
                p2x = (-b + np.sqrt(delta)) / (2 * a)
                p1y = k * x1 + b0
                p2y = k * x2 + b0
                inp = [[p1x, p1y], [p2x, p2y]]
                # select the points lie on the line segment
                inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
            else:
                inp = []
        return inp

    def segment_intersect_semicircle(self, pos_circle, r0, start_circle, end_circle):
        ints = self.segment_intersect_circle(pos_circle, r0)
        for int in ints:
            if min(start_circle.x, end_circle.x) < int[0] < max(start_circle.x, end_circle.x) and  min(start_circle.y, end_circle.y) < int[1] < max(start_circle.y, end_circle.y):
                return True
        return False

    def nearest_line_point__to_point(self, point):
        perpendicolar_line = self.perpendicolar_line(point)
        perpendicolar_point = intersect(self, perpendicolar_line)  # point on the line

        if (min(self.P1.x, self.P2.x) <= perpendicolar_point.x <= max(self.P1.x, self.P2.x) and
                min(self.P1.y, self.P2.y) <= perpendicolar_point.y <= max(self.P1.y, self.P2.y)):
            return perpendicolar_point
        else:  # it is not, I have then to find the nearest point on the segment
            nearest_point = Point()
            if ( min(self.P1.x, self.P2.x) <= perpendicolar_point.x <= max(self.P1.x, self.P2.x) and
                 min(self.P1.y, self.P2.y) <= perpendicolar_point.y <= max(self.P1.y, self.P2.y)
            ):
                return nearest_point

            if perpendicolar_point.x > max(self.P1.x, self.P2.x):
                nearest_point.x = max(self.P1.x, self.P2.x)
            else:
                nearest_point.x = min(self.P1.x, self.P2.x)
            if perpendicolar_point.y > max(self.P1.y, self.P2.y):
                nearest_point.y = max(self.P1.y, self.P2.y)
            else:
                nearest_point.y = min(self.P1.y, self.P2.y)

            return nearest_point

    def distance_to_point(self, point):
        nearest_p = self.nearest_line_point__to_point(point)
        return nearest_p.distance(point)

    def perpendicolar_line(self, point):
        if self.m is None:
            new_m = 0
        elif self.m == 0:
            new_m = None
        else:
            new_m = -1/self.m
        return Line(point, None, new_m)

    def print(self):
        print( 'P1(',self.P1.x,', ',self.P1.y,') - P2(',self.P2.x,', ',self.P2.y,') - slope(', self.m, ') - xy(',self.x,', ',self.y,')')

def slope(P1, P2):
    if(P2.x != P1.x):
        return (P2.y - P1.y) / (P2.x - P1.x)
    else:
        return None


def intersect(line1, line2): # this not use segment, but lines!
    if line1.m == line2.m:
        return None # same slope

    if(line1.m is None):
        return Point(line1.P1.x, line2.get_y_intersect(line1.P1.x))

    if(line2.m is None):
        return Point(line2.P1.x, line1.get_y_intersect(line2.P1.x))

    a = np.array(((line1.a, line1.b), (line2.a, line2.b)))
    b = np.array((-line1.c, -line2.c))
    x, y = np.linalg.solve(a, b)

    return Point(x, y)

def intersect_segments(line1, line2):
    Point = intersect(line1, line2)
    if(Point is None):
        return None
    # check if the intersection is in the bounds of the segments
    if(Point.x < max(min(line1.P1.x, line1.P2.x), min(line2.P1.x, line2.P2.x)) or Point.x > min(max(line1.P1.x, line1.P2.x), max(line2.P1.x, line2.P2.x)) or
        Point.y < max(min(line1.P1.y, line1.P2.y), min(line2.P1.y, line2.P2.y)) or Point.y > min(max(line1.P1.y, line1.P2.y), max(line2.P1.y, line2.P2.y))):
        return None
    else:
        return Point

import math

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def distance(self, P):
        return math.hypot(P.x - self.x, P.y - self.y)

    def print(self):
        print('(', self.x, ', ', self.y,')')