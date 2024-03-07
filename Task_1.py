import matplotlib.pyplot as plt
import numpy as np


def dot_product(f, g, x):
    dx = (x[1] - x[0])
    product = np.dot(f, g(x))
    return product*dx


def calculating_a(func, t, T, n):
    exp = lambda x: np.cos(2 * x * np.pi * n / T)
    result = np.round((2 / T) * dot_product(func, exp, t), 4)
    return result


def calculating_b(func, t, T, n):
    exp = lambda x: np.sin(2 * np.pi * n * x / T)
    result = np.round((2 / T) * dot_product(func, exp, t), 4)
    return result


def fourier_coefficients(func, t, T, n):
    c = [calculating_a(func, t, T, 0)]
    for i in range(1, n+1):
        c.append([calculating_a(func, t, T, i), calculating_b(func, t, T, i)])
    return c


def fourier_series(x, n, T, fc):
    N = np.arange(1, n+1)
    As = []
    Bs = []
    for i in N:
        As.append(fc[i][0])
        Bs.append(fc[i][1])
    As = np.array(As)
    Bs = np.array(Bs)
    y = np.sum(As.reshape(-1, 1)*np.cos(x*2*np.pi*N.reshape(-1, 1)/T)+Bs.reshape(-1, 1)*np.sin(x*2*np.pi*N.reshape(-1, 1)/T), axis=0)
    return y.flatten()


def g_coefficients(func, t, T, n):
    g = []
    for i in range(-n, n+1):
        exp = lambda x: np.exp(-(1j * 2 * np.pi * i * x) / T)
        g.append((1/T)*(dot_product(func, exp, t)))
    return g


def fourier_g_series(x, n, T, gc):
    N = np.arange(-n, n+1)
    y = np.sum(gc.reshape(-1, 1)*np.exp(1j * 2 * np.pi * N.reshape(-1, 1) * x / T), axis=0)
    return y.flatten()


def original_plot(label, x, y):
    fig = plt.figure(figsize=(7.0, 4.8))
    plt.plot(x, y, color='red')
    plt.margins(y=0.1, x=-0.03)
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('f(t)', fontfamily='serif', fontsize='large')
    plt.title(label, fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.grid()
    plt.show()
    fig.savefig(label, dpi=250)


def fourier_plot(x, n, T, fc, label):
    y_fc = fourier_series(x, n, T, fc) + fc[0] / 2
    fig = plt.figure(figsize=(7.0, 4.8))
    plt.margins(y=0.1, x=-0.01)
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('f(t)', fontfamily='serif', fontsize='large')
    plt.title(f'Частичные суммы Фурье при n={n}', fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.grid()
    plt.plot(x, y_fc, color='black')
    fig.savefig(label + f' {n}', dpi=250)
    plt.show()


def fourier_g_plot(x, n, T, gc, label):
    y_gc = fourier_g_series(x, n, T, gc)
    fig = plt.figure(figsize=(7.0, 4.8))
    plt.margins(y=0.1, x=-0.01)
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('f(t)', fontfamily='serif', fontsize='large')
    plt.suptitle(f'Частичная сумма Фурье при n={n}', fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.title('(Комплексная форма)', fontfamily='serif')
    plt.grid()
    plt.plot(x, y_gc, color='black')
    fig.savefig(label + f' {n}_g', dpi=250)
    plt.show()


def group_plot(x_or, y_or, fc, gc, n, T, label):
    y_fc = fourier_series(x_or, n, T, fc) + fc[0] / 2
    y_gc = fourier_g_series(x_or, n, T, gc)
    fig = plt.figure(figsize=(10.93, 6.8))
    plt.grid()
    plt.margins(y=0.1, x=-0.02)
    plt.plot(x_or, y_fc, label='Частичная сумма Фурье', color='yellow')
    plt.plot(x_or, y_gc, label='Частичная сумма Фурье (комплексная форма)', color='green', linestyle='--')
    plt.plot(x_or, y_or, label='Исходная функция', color='red')
    plt.xlabel('t', fontfamily='serif', fontsize='large')
    plt.ylabel('f(t)', fontfamily='serif', fontsize='large')
    plt.title(f'График функции f(t) (n={n})', fontweight='regular', fontsize='x-large', fontfamily='serif')
    plt.legend(loc=1)
    plt.show()
    fig.savefig(label + f'_Общая_{n}', dpi=250)


def pars_check(func, n, t, label, T=2*np.pi,):
    norm_f = np.dot(func, func)*(t[1] - t[0])
    f_c = fourier_coefficients(func, t, T, n)
    g_c = g_coefficients(func, t, T, n)
    sm_ab = np.pi*((f_c[0] ** 2)/2 + sum(f_c[i][0] ** 2 + f_c[i][1] ** 2 for i in range(1, n+1)))
    sm_c = 2*np.pi*sum(abs(g_c[i])**2 for i in range(len(g_c)))
    print(f'{label}:\nКвадрат нормы:{norm_f}\na, b: {sm_ab}\nc:{sm_c}\n')


# придуманные постоянные
a, b = 2, 1
t0, t1, t2 = 1, 3, 5
n = 10

original_info = []
fourier_info = []

# квадратная функция
x_s = np.linspace(0, 8, 1000)
T_s = x_s[-1] - x_s[0]
func = np.vectorize(lambda x: a if t0 <= (x - 1)%4 < t1 else b)
y_s = func(x_s)
original_info.append(['График функции f(t) - Меандр', x_s, y_s])
c_s = fourier_coefficients(y_s, x_s, T_s, n)
g_s = g_coefficients(y_s, x_s, T_s, n)
fourier_info.append([T_s, c_s, g_s, 'Меандр'])


# четная функция
x_e = np.linspace((-np.pi-2), 2*np.pi, 1000)
y_e = np.absolute(np.cos(x_e))
original_info.append(['График функции f(t)=|cos(t)|', x_e, y_e])
T_e = x_e[-1] - x_e[0]
c_e = fourier_coefficients(y_e, x_e, T_e, n)
g_e = g_coefficients(y_e, x_e, T_e, n)
fourier_info.append([T_e, c_e, g_e, 'Четная'])


# нечетная функция
x_o = np.linspace((-np.pi-2), 2*np.pi, 1000)
y_o = np.multiply(np.sin(3 * x_o), np.cos(4 * x_o))
original_info.append(['График функции f(t)=sin(3t)*cos(4t)', x_o, y_o])
T_o = x_o[-1] - x_o[0]
c_o = fourier_coefficients(y_o, x_o, T_o, n)
g_o = g_coefficients(y_o, x_o, T_o, n)
fourier_info.append([T_o, c_o, g_o, 'Нечетная'])

# функция общего вида
x_n = np.linspace((-np.pi-2), 2*np.pi, 1000, endpoint=True)
y_n = np.add(np.sin(3 * x_n), np.cos(x_n - 5))
original_info.append(['График функции f(t)=sin(3t)+cos(t-5)', x_n, y_n])
T_n = x_n[-1] - x_n[0]
c_n = fourier_coefficients(y_n, x_n, T_n, n)
g_n = g_coefficients(y_n, x_n, T_n, n)
fourier_info.append([T_n, c_n, g_n, 'Общая'])


# вывод графиков исходных функций, частичных фурье сумм и общих графиков
for i in range(1, 4):
    # original_plot(original_info[i][0], original_info[i][1], original_info[i][2])
    # fourier_plot(original_info[i][1], n, fourier_info[i][0], fourier_info[i][1], fourier_info[i][3])
    # fourier_g_plot(original_info[i][1], n, fourier_info[i][0], np.array(fourier_info[i][2]), fourier_info[i][3])
    group_plot(original_info[i][1], original_info[i][2], fourier_info[i][1], np.array(fourier_info[i][2]), n, fourier_info[i][0], fourier_info[i][3])
'''
# записываем коэффициенты для n=3   
f = open(f'N={n}_info.txt', 'w')
for i in range(4):
    f.write(f'Коэффициенты для {fourier_info[i][3]} при n={n}:\n{fourier_info[i][1]}\nПоказательная форма:\n{fourier_info[i][2]}\n')
f.close()
'''

'''# Проверяем равенство Парсеваля:
for i in range(4):
    pars_check(original_info[i][2], 200, original_info[i][1], fourier_info[i][3])'''



