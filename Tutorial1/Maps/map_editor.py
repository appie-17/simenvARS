import matplotlib.pyplot as plt
import numpy as np
import sys, time, os, glob
import matplotlib.patches as patches
from matplotlib.path import Path

sys.setrecursionlimit(100000) # problems with too many object and memory


# MapEditor - How it works:
# Click: add a point
# double click: close object  (without adding a point)
# middle click: save current active lines, but don't close them
# right click delete last added point or line
#
# When you close the window as output you will have the array in the terminal e a new file in Maps
# if you want load the last map uncomment map.load()
# if you want a base square board put the x, [y] like: map = MapEditor(20, [20])
# if you don't put anything it won't have the boarder and the map will have self.min_width and self.max_width as view

class MapEditor:
    # MAZE MODE ON/OFF
    # Maze ON =>  Only line, no closed obstacles
    MazeMode = False

    active_points = []
    active_points_draw = []
    active_lines_draw = []

    last_number_lines = 0
    lines = []

    min_width = min_height = 0
    max_width = max_height = 20

    fig = None
    ax = None

    def __init__(self, x=None, y=None):
        if x is not None: # premade border
            if y is None:
                y = x
            self.lines.append([[0,0],[0,y]])
            self.lines.append([[0,y],[x,y]])
            self.lines.append([[x,y],[x,0]])
            self.lines.append([[x,0],[0,0]])
            self.max_width = x
            self.max_height = y

        # so the boarder are not deletable:
        self.n_undeletable = len(self.lines)

        self.lines_patch = []

    def plot_editor(self):
        if self.fig is None:
            self.fig = plt.figure()

            self.ax = self.fig.add_subplot(111, aspect="equal")

            self.ax.set_xlim(self.min_width, self.max_width)
            self.ax.set_ylim(self.min_height, self.max_height)

            cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Obstacles
        dif = len(self.lines) - self.last_number_lines
        if dif > 0:
            self.create_lines_patch()
            for patch in self.lines_patch:
                self.ax.add_patch(patch)

            self.last_number_lines = len(self.lines)

        plt.show()

    def onclick(self, event):
        x = int(event.xdata)
        y = int(event.ydata)

        if event.dblclick:
            self.close_obstacle(True)
        if event.button == 2:
            self.close_obstacle(False)
        elif event.button == 3:  # click right
            self.remove_last()
        else:
            self.add_active_point(Point(x, y))

        self.plot_editor()

    def add_active_point(self, point):
        for stored_point in self.active_points:
            if stored_point == point:
                return

        self.active_points.append(point)
        self.active_points_draw.append(self.ax.plot(point.x, point.y, 'ro'))
        # add the line between the two points

        if len(self.active_points) > 1:
            last_ap = self.active_points[-2:-1][0]

            self.active_lines_draw.append(self.ax.plot([last_ap.x, point.x], [last_ap.y, point.y]))

    def remove_last(self):
        if len(self.active_points) == 0:
            if self.n_undeletable >= len(self.lines):
                return
            # rimuovo l'ultimo object
            if len(self.lines) == 0:
                return
            plt.close('all')
            self.fig = None
            self.lines = self.lines[:-1]
            self.last_number_lines = 0
            # self.ax.clear()
            self.plot_editor()

            return

        self.remove_active_point()

    def remove_active_point(self):
        self.active_points_draw[-1:][0].pop(0).remove()
        if len(self.active_lines_draw) > 0:
            self.active_lines_draw[-1:][0].pop(0).remove()

        self.active_points = self.active_points[:-1]
        self.active_points_draw = self.active_points_draw[:-1]
        self.active_lines_draw = self.active_lines_draw[:-1]

    def create_lines_patch(self):
        lines_patch = []
        for line in self.lines:
            lines_patch.append(patches.PathPatch(Path([(line[0][0], line[0][1]), (line[1][0], line[1][1])], [Path.MOVETO, Path.LINETO]), lw=1))
        self.lines_patch = lines_patch

    def close_obstacle(self, close=False):
        if close: # then remove the lastone, dblclick!
            self.remove_active_point()

        if len(self.active_points) < 2:
            print('Minimo 2 punti! :P')
            return
        else:
            if not self.MazeMode:
                last_p = None
                for p in self.active_points:
                    if last_p is None:
                        last_p = p
                    else:
                        self.lines.append([[last_p.x, last_p.y], [p.x, p.y]])
                        last_p = p
                if close and len(self.active_points) > 2:
                    p = self.active_points[0]
                    self.lines.append([[last_p.x, last_p.y], [p.x, p.y]])

            # resetto:
            self.active_points = []
            for d in self.active_points_draw:
                d.pop(0).remove()
            for d in self.active_lines_draw:
                d.pop(0).remove()

            self.active_points_draw = []
            self.active_lines_draw = []

            self.plot_editor()

    def print_maze_data(self):
        print(self.lines)


    out_dir = './'
    def save(self):
        if len(self.lines) > self.n_undeletable:
            np.save(self.out_dir + 'Map_' + str(time.time()), self.lines)

    def load(self, name=None):
        if name is None:
            list_of_files = glob.glob(self.out_dir + '*.npy')  # * means all if need specific format then *.csv
            if len(list_of_files) == 0:
                print('I can\'t load, the Folder is empty!')
                return
            latest_file = max(list_of_files, key=os.path.getctime)
            name = latest_file.split('\\')[1]

        stuff = np.load(self.out_dir + name)
        self.lines = []
        for line in stuff:
            self.lines.append(line)

        x,y=[],[]
        for line in self.lines:
            x+=[line[0][0], line[1][0]]
            y+=[line[0][1], line[1][1]]
        self.min_width = min(x)
        self.max_width = max(x)
        self.min_height = min(y)
        self.max_height = max(y)

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# map = MapEditor(15)
map = MapEditor(20) # without any border
map.load()

map.plot_editor()

map.print_maze_data()
map.save()
