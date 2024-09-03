import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 50  # кол-во измерений

U1 = np.linspace(0, n, n)
U2 = np.linspace(0, n, n)  # этой мы даем наклон
U3 = np.linspace(0, n, n) + np.random.normal(0, 0.015, 50)
U_1 = np.hstack([U2, U3])

df = pd.DataFrame(data=[U1.transpose(), U2.transpose(), U3.transpose()])
df = df.transpose()

"""
j = 0
U_LS = np.zeros((k, 1))
for i in range(k - 1):
    L = U1[j: int((i + 1) * n / k)]
    j = int((i + 1) * n / k)
    A = np.zeros((L.size, 2))
    # задаем массив эпох
    t = np.arange(0, L.size, 1)
    # цикл формирования матрицы коэффициентов, хотя можно было обойтись и без него
    for m in range(L.size):
        ti = t[m]
        A[m] = np.array([1, ti])
    # решаем СЛАУ
    N = -A.transpose().dot(A)
    X = np.linalg.inv(N).dot(A.transpose().dot(L))  # вектор параметров кинематической модели
    U_LS[i] = X[0] + X[1] * t[-1]
    # вычисляем вектор невязок
    V = A.dot(X) + L
    #  СКП единицы веса
    mu = np.sqrt(np.sum(V * V) / (V.shape[0] - 2))
    mu2 = np.mean(V)
"""

def geometric_chi_test_calc(time_series_frag, sigma, sigma_0):
    L = time_series_frag
    A = np.zeros((L.size, 2))  # матрица нулей с размерностью L
    # задаем массив эпох
    t = np.arange(0, L.size, 1)
    # цикл формирования матрицы коэффициентов, хотя можно было обойтись и без него
    for m in range(L.size):
        ti = t[m]
        A[m] = np.array([1, ti])
    # формирование матрицы весов
    P = np.diag(sigma) / sigma_0
    # решаем СЛАУ
    N = A.transpose().dot(P).dot(A)  # со знаком Я сомневаюсь
    X = np.linalg.inv(N).dot(A.transpose().dot(P).dot(L))  # вектор параметров кинематической модели
    x_LS = X[0] + X[1] * t[-1]
    # вычисляем вектор невязок
    V = A.dot(X) + L  #
    Qx = np.linalg.inv(N)
    # СКП единицы веса
    mu = np.sqrt(np.sum(V.transpose().dot(P).dot(V)) / (V.shape[0] - 2))  # я тут не уверен
    Qv = Qx[1, 1]
    return (x_LS, Qv, mu)


def geometric_chi_test_statictics(time_series_df, window,
                                  sigma_0, sigma=None):  # в проге нужно сделать так, чтобы сигмы брать из pos-файла. в тестовом скрипте этого нет
    X_WLS = np.zeros((window, time_series_df.shape[1]))
    Qv_WLS = np.zeros((window, time_series_df.shape[1]))

    '''if sigma is None:
        sigma = np.ones(time_series_df.shape[0] // window) * sigma_0'''


    for i in range(time_series_df.shape[1]):
        end = 0
        for j in range(window):
            st = end
            end = int((j + 1) * time_series_df.shape[0] / window)

            x_LS, Qx, mu = geometric_chi_test_calc(time_series_frag=time_series_df[i][st:end], sigma=sigma,
                                                   sigma_0=sigma_0)
            X_WLS[j, i] = x_LS
            Qv_WLS[j, i] = Qx
    test_statistic = np.zeros((window - 1))
    for l in range(window - 1):  # здесь косяки скорее всего. Нужно их исправлять
        Qv = Qv_WLS[l] + Qv_WLS[l + 1]
        d = X_WLS[l] - X_WLS[l + 1]
        Qdd = np.diag(Qv)
        test_statistic[l] = d.transpose().dot(Qdd).dot(d) / (sigma_0 ** 2)
        # d.transpose().dot(pinv(Qdd)).dot(d) / (sigma_0 ** 2)
    return X_WLS, Qv_WLS, test_statistic


df = pd.read_csv('Data/merged_2024-08-16_30s(feature).csv', delimiter=';')
df = df.iloc[:, 1:4]
df.columns = [0, 1, 2]

window = 5
sigma = np.ones(int(df.shape[0]/window))*0.01

X_WLS, Qv_WLS, test_statistic = geometric_chi_test_statictics(df, window, 0.015, sigma)

initial_values = df.iloc[0].values
X_WLS = np.insert(X_WLS, 0, initial_values, axis=0)

print('test_statistic: ', test_statistic)
print('X_WLS: ', X_WLS)
print('Qv_WLS: ', Qv_WLS)


fig, axs = plt.subplots(3, 1, figsize=(10, 10))

window_size = df.shape[0] // window

print(X_WLS.shape[0])
print(window_size)
print(df.shape[0])

for i, ax in enumerate(axs):
    ax.plot(df.iloc[:, i], label='Time Series')
    ax.plot(np.arange(0, df.shape[0]+1, window_size)[:X_WLS.shape[0]], X_WLS[:, i], color='r', label='WLS Estimate')
    for j in range(X_WLS.shape[0]):
        ax.axvline(j * window_size, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f'Coordinate {i}')
    ax.legend()

'''
for i, ax in enumerate(axs):
    ax.plot(df.iloc[:, i], label='Time Series')
    ax.plot(np.arange(0, df.shape[0], window_size), X_WLS[:, i], color='r', label='WLS Estimate')
    for j in range(X_WLS.shape[0]):
        ax.axvline(j * window_size, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f'Coordinate {i}')
    ax.legend()
'''

plt.tight_layout()
plt.show()

