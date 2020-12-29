import numpy as np

def Ray(x, y):
    return np.array([x, y], dtype=np.csingle)

def PolLineal(param):
    mat = np.zeros((2, 2), dtype=np.csingle)
    if param == 'h' or param == 'horizontal':
        mat[0][0] = 1
    elif param == 'v' or param == 'vertical':
        mat[1][1] = 1
    elif _is_digit(param):
        alpha = np.radians(param)
        mat[0][0] = np.cos(alpha)**2
        mat[0][1] = np.sin(alpha) * np.cos(alpha)
        mat[1][0] = mat[0][1]
        mat[1][1] = np.sin(alpha)**2
    else:
        print('Bad Parameter in PolLineal!')
        assert False
    return mat

def Lamina4Onda(param):
    mat = np.zeros((2, 2), dtype=np.csingle)
    if param == 'h' or param == 'horizontal':
        mat[0][0] = 1
        mat[1][1] = 1j
    elif param == 'v' or param == 'vertical':
        mat[0][0] = 1
        mat[1][1] = -1j
    elif _is_digit(param):
        alpha = np.radians(param)
        mat[0][0] = np.cos(alpha)**2 + 1j*np.sin(alpha)**2
        mat[0][1] = (1-1j) * np.sin(alpha) * np.cos(alpha)
        mat[1][0] = mat[0][1]
        mat[1][1] = np.sin(alpha)**2 + 1j*np.cos(alpha)**2
    else:
        print('Bad Parameter in Lamina4Onda!')
        assert False
    return mat

def Lamina2Onda(param):
    mat = np.zeros((2, 2), dtype=np.csingle)
    if param == 'h' or param == 'horizontal' \
        or param == 'v' or param == 'vertical':
        mat[0][0] = 1
        mat[1][1] = -1
    elif _is_digit(param):
        alpha = np.radians(param)
        mat[0][0] = np.cos(2*alpha)
        mat[0][1] = np.sin(2*alpha)
        mat[1][0] = mat[0][1]
        mat[1][1] = -mat[0][0]
    else:
        print('Bad Parameter in Lamina2Onda!')
        assert False
    return mat

def Rotation(angle):
    if not _is_digit(angle):
        print('Bad Parameter in Rotation!')
        assert False
    alpha = np.radians(angle)
    mat = np.zeros((2, 2), dtype=np.csingle)
    mat[0][0] = np.cos(alpha)
    mat[0][1] = -np.sin(alpha)
    mat[1][0] = -mat[0][1]
    mat[1][1] = mat[0][0]
    return mat

def Identity():
    return np.identity(2)

def Solve(*layers):
    layers = list(layers)
    while True:
        res = layers[-2].dot(layers[-1])
        layers = layers[:-2] + [res]
        if len(layers) <= 1:
            return Zero(layers[0])

def Graph(ray):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = np.linspace(0, 30, 100)
    X = ray[0]*np.exp(-1j*z)
    Y = ray[1]*np.exp(-1j*z)
    ax.plot(X, Y, z)
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def _is_digit(n):
    if isinstance(n, int) or \
            isinstance(n, float) or \
            isinstance(n, complex) or \
            isinstance(n, np.float32) or \
            isinstance(n, np.int16):
        return True
    return False

def _zero_vect(x):
    eps = min(np.finfo(np.csingle).eps, np.finfo(float).eps)
    real = x.real
    imag = x.imag * 1j
    if abs(real) <= eps:
        real = 0.0
    if abs(imag) <= eps:
        imag = 0j
    return real + imag
Zero = np.vectorize(_zero_vect)

if __name__ == '__main__':
    rayo = Ray(1, 1j)
    lam1 = Lamina2Onda('vertical')
    pol1 = PolLineal(45)
    lam2 = Lamina4Onda(30)
    s = Solve(lam2, pol1, lam1, rayo)
    Graph(s)