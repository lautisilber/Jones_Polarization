# works well with pj_mat library
import numpy as np
import matplotlib.pyplot as plt
from pj_mat import PolLineal

def EnergyGraph(ray):
    # energy graph after linear polarizator of I vs. θ
    theta = np.linspace(0, 180, 500)
    T = []
    I = []
    for t in theta:
        e = PolLineal(t).dot(ray)
        i = GetEnergy(e)
        I.append(i)
        T.append(t)
    max_i_y = max(I[10:-9])
    max_index = I.index(max_i_y, 10, -9)
    max_i_x = T[max_index]
    min_i_y = min(I[10:-9])
    min_index = I.index(min_i_y, 10, -9)
    min_i_x = T[min_index]

    plt.xlim((0, 180))
    plt.xticks(range(0, 181, 45))
    plt.xlabel('Angulo θ')
    plt.ylabel('Intensidad I')
    plt.title('Rayo a través de pol. lineal')
    plt.plot(T, I, color='tab:orange')
    plt.plot([0, 180], [max_i_y, max_i_y], ls=':', color='tab:gray')
    plt.plot([0, 180], [min_i_y, min_i_y], ls=':', color='tab:gray')
    ylim = plt.ylim()
    plt.plot([max_i_x, max_i_x], [-10, max_i_y], color='tab:green', ls='--',
                label='max ({}, {})'.format(round(max_i_x, 2), round(max_i_y.real, 2)))
    plt.plot([min_i_x, min_i_x], [-10, min_i_y], color='tab:blue', ls='--',
                label='min ({}, {})'.format(round(min_i_x, 2), round(min_i_y.real, 2)))
    plt.ylim(ylim)
    plt.legend()
    plt.show()

def GetEnergy(ray):
    energy = ray.dot(np.conj(ray)) / 2
    assert energy.imag == 0
    return energy.real

if __name__ == '__main__':
    from pj_mat import Ray
    rayo = Ray(1, 1j + 1)
    EnergyGraph(rayo)