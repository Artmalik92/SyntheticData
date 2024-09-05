import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import medfilt
import statsmodels.api as sm


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
    # Mask NaN values
    mask = ~np.isnan(time_series_frag)
    L = time_series_frag[mask]

    sigma = np.ones(L.size) * 0.01
    # задаем массив эпох
    t = np.arange(0, L.size, 1)
    A = np.column_stack((np.ones(L.size), t))
    P = np.diag(1 / (sigma * sigma_0))

    """A = np.zeros((L.size, 2))  # матрица нулей с размерностью L
    
    # цикл формирования матрицы коэффициентов, хотя можно было обойтись и без него
    for m in range(L.size):
        ti = t[m]
        A[m] = np.array([1, ti])
    # формирование матрицы весов
    P = np.diag(sigma) / sigma_0"""


    # решаем СЛАУ
    N = A.transpose().dot(P).dot(A)  # со знаком Я сомневаюсь
    X = np.linalg.pinv(N).dot(A.transpose().dot(P).dot(L))  # вектор параметров кинематической модели
    x_LS = X[0] + X[1] * t[-1]
    x_LS_first = X[0] + X[1] * t[0]
    # вычисляем вектор невязок
    V = A.dot(X) + L  #
    Qx = np.linalg.pinv(N)
    # СКП единицы веса
    mu = np.sqrt(np.sum(V.transpose().dot(P).dot(V)) / (V.shape[0] - 2))  # я тут не уверен
    Qv = Qx[1, 1]
    return (x_LS_first, x_LS, Qv, mu)


def geometric_chi_test_statictics(time_series_df, resample_period,
                                  sigma_0):  # в проге нужно сделать так, чтобы сигмы брать из pos-файла. в тестовом скрипте этого нет
    X_WLS = []
    Qv_WLS = []
    wls_times = []
    initial_values_X = []
    initial_values_Qv = []

    #sigma = np.ones(int(time_series_df.shape[0] / window)) * 0.01

    time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
    time_series_df.set_index('Date', inplace=True)

    # Calculate the initial values
    start_time = time_series_df.index[0]
    end_time = start_time + pd.Timedelta(resample_period)
    fragment = time_series_df[(time_series_df.index >= start_time) & (time_series_df.index < end_time)]
    for col in time_series_df.columns:
        x_LS_first, _, Qx, _ = geometric_chi_test_calc(time_series_frag=fragment[col],
                                                      sigma=np.ones(fragment.shape[0]) * 0.01,
                                                      sigma_0=sigma_0)
        initial_values_X.append(x_LS_first)
        initial_values_Qv.append(Qx)

    X_WLS.append(initial_values_X)
    Qv_WLS.append(initial_values_Qv)
    wls_times.append(start_time)

    i = 0

    while end_time <= time_series_df.index[-1]:
        wls_times.append(end_time)
        # Extract the fragment
        fragment = time_series_df[(time_series_df.index >= start_time) & (time_series_df.index < end_time)]

        x_LS_values = []
        Qx_values = []

        # Apply the least squares code to the fragment
        for col in fragment.columns:
            x_LS_first, x_LS, Qx, mu = geometric_chi_test_calc(time_series_frag=fragment[col],
                                                               sigma=np.ones(fragment.shape[0]) * 0.01, sigma_0=sigma_0)

            x_LS_values.append(x_LS)
            Qx_values.append(Qx)

        X_WLS.append(x_LS_values)
        Qv_WLS.append(Qx_values)

        # Move to the next fragment
        start_time = end_time
        end_time = start_time + pd.Timedelta(resample_period)
        i += 1

    # Convert the lists to numpy arrays
    X_WLS = np.array(X_WLS)
    Qv_WLS = np.array(Qv_WLS)

    wls_df = pd.DataFrame({'Date': wls_times})
    wls_df[['Coord 0', 'Coord 1', 'Coord 2']] = X_WLS

    # Calculate the test statistic
    test_statistic = np.zeros((X_WLS.shape[0] - 1))
    for l in range(X_WLS.shape[0] - 1):
        Qv = Qv_WLS[l] + Qv_WLS[l + 1]
        d = X_WLS[l] - X_WLS[l + 1]
        Qdd = np.diag(Qv)
        test_statistic[l] = d.transpose().dot(Qdd).dot(d) / (sigma_0 ** 2)

    return X_WLS, Qv_WLS, test_statistic, wls_df


def interpolate_missing_values(df):
    """
    Interpolate missing values in the dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with interpolated missing values.
    """
    df_interpolated = df.interpolate(method='linear', limit_direction='both')
    return df_interpolated


df = pd.read_csv('Data/merged_2024-08-29_with_na_nozeroepoch.csv', delimiter=';')
df = df.iloc[:, [0, 4, 5, 6]]

df.columns = ['Date', 0, 1, 2]
# Interpolate missing values
df = interpolate_missing_values(df)
df_filtered = df
df_filtered.iloc[:, 1:] = df_filtered.iloc[:, 1:].apply(lambda x: medfilt(x, kernel_size=11))

window_length = '10min'
X_WLS, Qv_WLS, test_statistic, wls_df = geometric_chi_test_statictics(df_filtered, window_length, 0.015)

print(wls_df)


fig, axs = plt.subplots(3, 1, figsize=(10, 10))

window_length = pd.Timedelta(window_length)



for i, ax in enumerate(axs):
    ax.plot(df.index, df.iloc[:, i], label='Raw data', color='lightgray')  # plot the coordinates
    ax.plot(df_filtered.index, df_filtered.iloc[:, i], label='Filtered', color='b')

    # Plot the WLS estimates
    ax.plot(wls_df['Date'], wls_df[f'Coord {i}'], color='red', label='WLS Estimate')

    # Plot the vertical lines
    for j in wls_df['Date']:
        ax.axvline(j, color='r', linestyle='--', alpha=0.2)

    # Set x-axis tick labels to show dates of vertical lines
    ax.set_xticks(wls_df['Date'])
    ax.set_xticklabels(wls_df['Date'].dt.strftime('%H:%M:%S'))

    ax.set_title(f'Coordinate {i}')
    ax.legend()
    fig.autofmt_xdate()

plt.tight_layout()
# Get the current figure manager
manager = plt.get_current_fig_manager()
# Get the window object
window = manager.window
# Show the window maximized
window.showMaximized()
# Show the plot
plt.show()


