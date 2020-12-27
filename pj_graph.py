import numpy as np
import matplotlib.pyplot as plt
from collections import deque

j = complex(0, 1)

RAY_COLORS = deque(['tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']) #this deque list can rotate!!

class System:
    def __init__(self, ray):
        assert isinstance(ray, Ray)
        self.source_ray = ray
        self.layers = []

    def add(self, layer):
        assert is_layer(layer)
        self.layers.append(layer)

    def summary(self):
        Log.colourprint('Summary of system:', Log.FG_BLACK, Log.BG_CYAN, Log.UNDERLINE)
        Log.colourprint('Source ray:', Log.UNDERLINE)
        print(self.source_ray.j_vec)
        n = 1
        for layer in self.layers:
            Log.colourprint('Layer {0}: {1}'.format(n, layer.name), Log.UNDERLINE)
            print(layer.mat)
            n+=1

    def calc_end_ray(self):
        res = self.source_ray.j_vec
        for layer in self.layers:
            res = layer.mat.dot(res).T
        return res

    def solve_system(self):
        Log.colourprint('Solve system:', Log.FG_BLACK, Log.BG_CYAN, Log.UNDERLINE)
        Log.colourprint('Source ray:', Log.UNDERLINE)
        res = self.source_ray.j_vec
        print(res)
        n = 1
        for layer in self.layers:
            Log.colourprint('After layer {0} ({1}):'.format(n, layer.name), Log.UNDERLINE)
            res = layer.mat.dot(res).T
            print(res)
            n+=1
        

    def graph_system(self):
        layer_dist = 30
        resolution = 100

        sections = [Layers.Air()]
        for layer in self.layers:
            sections.append(layer)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        res = self.source_ray.j_vec
        n = 0
        
        for sec in sections:
            res = sec.mat.dot(res).T
            z = np.linspace(n*layer_dist, (n+1)*layer_dist, resolution)
            x, y = graph(res, z)
            ax.plot(x, y, z, color=RAY_COLORS[0], zorder=10)
            RAY_COLORS.rotate(-1)
            n+=1

        ax.plot([0, 0], [0, 0], [0, layer_dist*n], color='tab:green', zorder=5)
        yl = ax.get_ylim()
        xl = ax.get_xlim()
        m = max(abs(max(xl)), abs(max(yl)))
        ax.set_xlim3d([-m, m])
        ax.set_ylim3d([-m ,m])
        for i in range(1, n):
            X, Y = np.meshgrid([-m/2, m/2], [-m/2, m/2])
            Z = (i * layer_dist) * np.ones(X.shape)
            ax.plot_surface(X, Y, Z, color='tab:blue', shade=False)
        ax.scatter(0, 0, 0, color='tab:red', label='Soruce')
        plt.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
            
class Layers:
    class LinearPolarizer:
        def __init__(self, param):
            if param == 'vertical':
                lp = np.array([
                    [0, 0],
                    [0, 1]
                ])
            elif param == 'horizontal':
                lp = np.array([
                    [1, 0],
                    [0, 0]
                ])
            elif is_number(param): #angle in degrees
                rad = np.radians(param)
                lp = np.array([
                    [np.cos(rad)**2, np.sin(rad)*np.cos(rad)],
                    [np.sin(rad)*np.cos(rad), np.sin(rad)**2]
                ])
            else:
                Log.error('bad parameter in linear polarizer')
                assert False
            self.mat = lp
            self.name = ''
            if isinstance(param, str):
                self.name = 'Polarizador lineal (' + str(param) + ')'
            elif is_number(param):
                self.name = 'Polarizador lineal (' + str(param) + '°)'

    class Lamina4Onda:
        def __init__(self, param):
            com = complex(0, 1)
            if param == 'vertical':
                lp = np.array([
                    [1, 0],
                    [0, -com]
                ])
            elif param == 'horizontal':
                lp = np.array([
                    [1, 0],
                    [0, com]
                ])
            elif is_number(param): #angle in degrees
                rad = np.radians(param)
                lp = np.array([
                    [np.cos(rad)**2 + com*np.sin(rad)**2, (1 - com)*np.sin(rad)*np.cos(rad)],
                    [(1 - com)*np.sin(rad)*np.cos(rad), np.sin(rad)**2 + com*np.cos(rad)**2]
                ])
            else:
                Log.error('bad parameter in lamina de cuarto de onda')
                assert False
            self.mat = lp
            self.name = ''
            if isinstance(param, str):
                self.name = 'Lámina λ/4 (' + str(param) + ')'
            elif is_number(param):
                self.name = 'Lámina λ/4 (' + str(param) + '°)'

    class Lamina2Onda:
        def __init__(self, param):
            
            if param == 'vertical' or param == 'horizontal':
                lp = np.array([
                    [1, 0],
                    [0, -1]
                ])
            elif is_number(param): #angle in degrees
                rad = np.radians(param)
                lp = np.array([
                    [np.cos(2*rad), np.sin(2*rad)],
                    [np.sin(2*rad), -np.cos(2*rad)]
                ])
            else:
                Log.error('bad parameter in lamina de media onda')
                assert False
            self.mat = lp
            self.name = ''
            if isinstance(param, str):
                self.name = 'Lámina λ/2 (' + str(param) + ')'
            elif is_number(param):
                self.name = 'Lámina λ/2 (' + str(param) + '°)'

    class Air:
        def __init__(self, param=0):
            self.mat = np.identity(2)
            self.name = 'Air'

class Ray:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.j_vec = np.array([self.x, self.y]).T
        self.I = (self.j_vec.dot(self.j_vec.conj()))/2 # a chequear!!!

class Log:
    RESET = '\u001b[0m'

    FG_BLACK = '\u001b[30m'
    FG_RED = '\u001b[31m'
    FG_GREEN = '\u001b[32m'
    FG_YELLOW = '\u001b[33m'
    FG_BLUE = '\u001b[34m'
    FG_MAGENTA = '\u001b[35m'
    FG_CYAN = '\u001b[36m'
    FG_WHITE = '\u001b[37m'

    BG_BLACK = '\u001b[40m'
    BG_RED = '\u001b[41m'
    BG_GREEN = '\u001b[42m'
    BG_YELLOW = '\u001b[43m'
    BG_BLUE = '\u001b[44m'
    BG_MAGENTA = '\u001b[45m'
    BG_CYAN = '\u001b[46m'
    BG_WHITE = '\u001b[47m'

    BOLD = '\u001b[1m'
    UNDERLINE = '\u001b[4m'
    REVERSED = '\u001b[7m'

    @staticmethod
    def info(message):
        print(Log.FG_GREEN + message + Log.RESET)

    @staticmethod
    def warning(message):
        print(Log.FG_YELLOW + Log.UNDERLINE + 'WARNING' + Log.RESET + Log.FG_YELLOW + ': ' + message + Log.RESET)

    @staticmethod
    def error(message):
        print(Log.FG_BLACK + Log.BG_RED + Log.UNDERLINE + 'ERROR' + Log.RESET + Log.FG_BLACK + Log.BG_RED + ': ' + message + Log.RESET)

    @staticmethod
    def colourprint(message, *args):
        print(''.join(args) + message + Log.RESET)

def graph(j_vec, zs):
    # para t = 0
    A = 1
    k = 0.5
    wave = lambda comp, z: comp * A * np.exp(j * k * z)

    x = j_vec.T[0]
    y = j_vec.T[1]

    xs = [wave(x, z) for z in zs]
    ys = [wave(y, z) for z in zs]

    return xs, ys

def is_number(n):
    if isinstance(n, int) or isinstance(n, float) or isinstance(n, complex):
        return True
    return False

def is_real(n):
    if isinstance(n, int) or isinstance(n, float):
        return True
    elif isinstance(n, complex):
        if n.imag == 0:
            return True
    return False

def is_complex(n):
    if isinstance(n, complex):
        if n.imag != 0:
            return True
    return False

def is_layer(l):
    if isinstance(l, Layers.LinearPolarizer) or isinstance(l, Layers.Lamina4Onda) or isinstance(l, Layers.Lamina2Onda) or isinstance(l, Layers.Air):
        return True
    return False

###

def demo():
    ray = Ray(1, -2*j+1)
    s = System(ray)
    s.add(Layers.Lamina2Onda('vertical'))
    s.add(Layers.Lamina4Onda('horizontal'))
    s.add(Layers.LinearPolarizer(30))
    
    s.graph_system()
    s.solve_system()

if __name__ == '__main__':
    demo()
