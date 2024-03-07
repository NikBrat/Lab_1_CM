import matplotlib.pyplot as plt
import numpy
import numpy as np


def dot_product(f, g, x):
    dx = x[1] - x[0]
    product = np.dot(f, g(x))
    return product*dx


def g_coefficients(func, t, T, n):
    g = []
    for i in range(-n, n+1):
        exp = lambda x: np.exp(-(1j * 2 * np.pi * i * x) / T)
        g.append((1/T)*(dot_product(func, exp, t)))
    return np.array(g)


def fourier_g_series(x, n, T, gc):
    N = np.arange(-n, n+1)
    y = np.sum(gc.reshape(-1, 1)*np.exp(1j * 2 * np.pi * N.reshape(-1, 1) * x / T), axis=0)
    return y.flatten()


def original_plot(label, x, y):
    fig = plt.figure(figsize=(7.0, 4.8))
    plt.plot(x, y, color='red')
    plt.margins(y=0.1, x=0.1)
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('Im(f(t))', fontfamily='serif', fontsize='large')
    plt.title(label, fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.grid()
    plt.show()
    fig.savefig(label, dpi=250)


def fourier_g_plot(x, n, T, gc):
    y_gc = fourier_g_series(x, n, T, gc)
    fig = plt.figure(figsize=(7.0, 4.8))
    plt.margins(y=0.1, x=0.1)
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('Im(Gn(t))', fontfamily='serif', fontsize='large')
    plt.title(f'График Im(Gn(t)) при n={n}', fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.grid()
    plt.plot(x, y_gc.imag, color='black')
    fig.savefig(f'Im({n})_g', dpi=250)
    # plt.show()


def group_plot(x_or, y_or, gc, n, T):
    y_gc = fourier_g_series(x_or, n, T, gc)
    fig = plt.figure(figsize=(10.34, 6.9))
    plt.grid()
    plt.margins(y=0.1, x=0.1)
    plt.plot(x_or, y_or.imag, label='График Im(f(t))', color='red')
    plt.plot(x_or, y_gc.imag, label=f'График Im(Gn(t)) при n={n}', color='black', linestyle='--')
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('Im(f(t))', fontfamily='serif', fontsize='large')
    plt.title(f'График функции f(t) (n={n})', fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.legend(loc=4)
    # plt.show()
    fig.savefig(f'ОбщаяIm_{n}', dpi=250)


def pars_check(func, n, t, label, T=2*np.pi,):
    norm_f = np.abs(np.dot(func, numpy.conjugate(func))*(t[1] - t[0]))
    g_c = g_coefficients(func, t, T, n)
    sm_c = 2*np.pi*sum(abs(g_c[i])**2 for i in range(len(g_c)))
    print(f'{label}:\nКвадрат нормы:{norm_f}\nc:{sm_c}\n')


# Выбираем постоянные
R, T = 2, 8


def function(t):
    r, img = -1, -2
    if -(T/8) <= t < (T/8):
        r, img = R, 8*R*t/T
    elif (T/8) <= t < (3*T/8):
        r, img = 2*R-8*R*t/T, R
    elif (3*T/8) <= t < (5*T/8):
        r, img = -R, 4*R-8*R*t/T
    elif (5*T/8) <= t < (7*T/8):
        r, img = -6*R+8*R*t/T, -R
    return r + img*1j


func = np.vectorize(function)
x_f = np.linspace(-np.pi, np.pi, 1000)
y_f = func(x_f)
T_f = x_f[-1] - x_f[0]
# n = 3


# original_plot('График Re(f(t))', x_f, y_f.real)


'''
N = [1, 2, 3, 10]
for n in N:
    g_f = g_coefficients(y_f, x_f, T_f, n)
    fourier_g_plot(x_f, n, T_f, g_f)
    group_plot(x_f, y_f, g_f, n, T_f)
'''

pars_check(y_f, 10, x_f, '10')

