import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import inv
from numpy import fft

N = 365 * 10                     # Количество дней наблюдений
t = np.linspace(0, N/365.25, N)  # Время в годах

np.random.seed(0)                # ключ генерации

kappa = -1                       # Flicker noise
h = np.zeros(2*N)                # Note the size : 2N
h[0] = 0.5                         # Eq. (25)


for i in range(1, N):                                       # Генерация шума
    h[i] = (i-kappa/2-1)/i * h[i-1]

v = np.zeros(2*N)                                           # Again zero-padded N:2N
v[0:N] = np.random.normal(loc=0.0, scale=0.5, size=N)

w = np.real(fft.ifft(fft.fft(v) * fft.fft(h)))              # Разложение по методу Фурье
''' 
Метод Фурье конвертирует сигнал в индивидуальные спектральные  компоненты и тем самым предоставляет частотную информацию
о сигнале. Здесь сигналы разлагаются на ряд Фурье и умножаются, затем конвертируются обратно.
'''
y = (6 + 3*t) + w[0:N]                                      # траектория (линейный тренд волн) + noise

plt.plot(t, y, "b-")                                        # plot the time series
print('h =', h)
print('v =', v)
print('w =', w)
print('y =', y)
plt.show()
'''
# The design matrix (Матрица с двумя столбцами значений: в первом - единицы, во втором - время)
A = np.empty((N, 2))
for i in range(0, N):
    A[i, 0] = 1
    A[i, 1] = t[i]

# Old white noise method (Метод исследования для белого шума)
C = np.identity(N)
x = inv(A.T @ inv(C) @ A) @ (A.T @ inv(C) @ y)           # Eq. (14)
y_hat = A @ x
r = y - y_hat                                            # residuals
C_x = np.var(r) * inv(A.T @ inv(C) @ A)                  # Eq. (15)
print('White noise approximation')
print('a = {0:6.3f} +/- {1:5.3f} mm'.format(x[0], math.sqrt(C_x[0, 0])))
print('b = {0:6.3f} +/- {1:5.3f} mm/yr'.format(x[1], math.sqrt(C_x[1, 1])))


# power-law noise covariance matrix (Степенной закон)
def create_C(sigma_pl, kappa):
    U = np.identity(N)
    h_prev = 1
    for i in range(1, N):
        h = (i-kappa/2-1)/i * h_prev        # Eq. (25)
        for j in range(0, N-i):
            U[j, j+i] = h
        h_prev = h
    U *= sigma_pl                           # scale noise
    return U.T @ U                          # Eq. (26)


# weighted least-squares (Взвешанный метод наименьших квадратов)
def leastsquares(C, A, y):
    U = np.linalg.cholesky(C).T                 # Разложение Холецкого
    U_inv = inv(U)
    B = U_inv.T @ A
    z = U_inv.T @ y
    x = inv(B.T @ B) @ B.T @ z                  # Eq. (14)

    # variance of the estimated parameters
    C_x = inv(B.T @ B)                          # Eq. (15)

    # Compute log of determinant of C
    ln_det_C = 0.0
    for i in range(0, N):
        ln_det_C += 2*math.log(U[i, i])

    return [x, C_x, ln_det_C]


# The correct flicker noise covariance matrix (Корректировка фликкер-шума)
sigma_pl = 0.495
kappa = -1.004
C = create_C(sigma_pl, kappa)
[x, C_x, ln_det_C] = leastsquares(C, A, y)
print('Correct Flicker noise')
print('a = {0:6.3f} +/- {1:5.3f} mm'.format(x[0], math.sqrt(C_x[0, 0])))
print('b = {0:6.3f} +/- {1:5.3f} mm/yr'.format(x[1], math.sqrt(C_x[1, 1])))


# Log-likelihood (with opposite sign) (Метод наибольшего правдоподобия)
def log_likelihood(x_noise):
    sigma_pl = x_noise[0]
    kappa = x_noise[1]
    C = create_C(sigma_pl, kappa)
    [x, C_x, ln_det_C] = leastsquares(C, A, y)
    r = y - A @ x                                 # residuals

    #--- Eq. (12)
    logL = -0.5*(N*math.log(2*math.pi) + ln_det_C + r.T @ inv(C) @ r)
    return -logL


x_noise0 = np.array([1, 1])                       # sigma_pl and kappa guesses
res = minimize(log_likelihood, x_noise0, method='nelder-mead', options={'xatol': 0.01})

print('sigma_pl={0:6.3f}, kappa={1:6.3f}'.format(res.x[0], res.x[1]))


S = np.empty((21, 21))
for i in range(0, 21):
    sigma_pl = 1.2 - 0.05*i
    for j in range(0, 21):
        kappa = -1.9 + 0.1*j
        x_noise0 = [sigma_pl, kappa]
        S[i, j] = math.log(log_likelihood(x_noise0))

plt.imshow(S, extent=[-1.9, 0.1, 0.2, 1.2], cmap='nipy_spectral', aspect='auto')

plt.colorbar()
plt.ylabel('sigma_pl')
plt.xlabel('kappa')


plt.show()
'''