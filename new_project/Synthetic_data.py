import pandas as pd
from pandas import DataFrame
import numpy as np
import math as m
from numpy import random, fft
from geopy.distance import geodesic
from scipy.spatial import cKDTree, Delaunay
from scipy.signal import unit_impulse
import pymap3d
import matplotlib.pyplot as plt


# скорости пунктов для линейного тренда
vx = [0, -0.0248, -0.0282]  # m/year
vy = [0, 0.0022, 0.0048]    # m/year
vz = [0, -0.0005, 0.0024]   # m/year

# синусоидальные и косинусоидальные компоненты ежегодной периодичности
X_cosine_annual = -9.22073228840851e-04
X_sine_annual = -8.33375846509620e-04
Y_cosine_annual = -3.77540217555887e-03
Y_sine_annual = -1.56021023058487e-03
Z_cosine_annual = -7.95353842309861e-03
Z_sine_annual = -5.24528624062300e-03
X_cosine_semiannual = 4.43980484775477e-04
X_sine_semiannual = 5.56854095417971e-04
Y_cosine_semiannual = 8.88035304955231e-04
Y_sine_semiannual = 1.92258274437124e-03
Z_cosine_semiannual = 4.22368034377985e-04
Z_sine_semiannual = 1.37821014348371e-03


class SyntheticData(DataFrame):
    def __init__(self):
        super().__init__()

    # BLH to XYZ
    def my_geodetic2ecef(df):
        # создание нового DataFrame с обновленными координатами
        df_new = pd.DataFrame({'Date': df['Date'],
                               'Station': df['Station'],
                               'X': 0,
                               'Y': 0,
                               'Z': 0})

        # перевод координат и добавление в новый DataFrame
        for i, row in df.iterrows():
            X, Y, Z = pymap3d.geodetic2ecef(row['B'], row['L'], row['H'])
            df_new.at[i, 'X'] = X
            df_new.at[i, 'Y'] = Y
            df_new.at[i, 'Z'] = Z

        return df_new

    # XYZ to BLH
    def my_ecef2geodetic(df):
        # создание нового DataFrame с обновленными координатами
        df_new = pd.DataFrame({'Date': df['Date'],
                               'Station': df['Station'],
                               'B': 0,
                               'L': 0,
                               'H': 0})

        # перевод координат и добавление в новый DataFrame
        for i, row in df.iterrows():
            B, L, H = pymap3d.ecef2geodetic(row['X'], row['Y'], row['Z'])
            df_new.at[i, 'B'] = B
            df_new.at[i, 'L'] = L
            df_new.at[i, 'H'] = H

        return df_new

    # вспомогательная функция для создания временных рядов
    def unique_names(df):
        unique_names = []
        for name in df["Station"]:
            if name not in unique_names:
                unique_names.append(name)
            else:
                break
        return unique_names

    # Случайная сеть пунктов
    def random_points(B, L, H, zone, amount, method):
        k = 1  # k - количество ближайших соседей для проверки расстояния
        start_point = {'Date': 'None', 'Station': 'NSK1', 'B': B, 'L': L, 'H': H}  # исходный пункт

        points = [start_point]

        while len(points) < amount:
            Station = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4))
            B = np.random.uniform(start_point['B'] - zone, start_point['B'] + zone)
            L = np.random.uniform(start_point['L'] - zone, start_point['L'] + zone)
            H = np.random.uniform(start_point['H'] - 25, start_point['H'] + 25)
            point = {'Date': 'None', 'Station': Station, 'B': B, 'L': L, 'H': H}

            distances, indices = cKDTree(list([p['B'], p['L']] for p in points)).query([[B, L]], k=k)
            '''
            if len(points) < 6:
                if all(20000 < geodesic((p['B'], p['L']), (B, L)).meters < 30000 for p in points):
                    print(f"Distances to nearest neighbors of {Station}: {distances}")
                    start_point = point
                    points.append(point)
            else:
                if sum(1 for p in points if 20000 < geodesic((p['B'], p['L']), (B, L)).meters < 30000) >= 6:
                    print(f"Distances to nearest neighbors of {Station}: {distances}")
                    start_point = point
                    points.append(point)
                    k += 1
            '''
            if all(20000 < geodesic((p['B'], p['L']), (B, L)).meters for p in points):
                # if any(20000 < geodesic((p['B'], p['L']), (B, L)).meters < 30000 for p in points)
                # if np.all(20 < d < 30 for d in distances):
                # print(f"Distances to nearest neighbors of {Station}: {distances}")
                if method == 'consistent':
                    start_point = point
                elif method == 'centralized':
                    pass
                points.append(point)
                k += 1

        return pd.DataFrame(points)

    # Схема сети (метод триангуляции)
    def triangulation(df):
        fig, ax = plt.subplots()

        # рисуем точки на графике
        ax.scatter(df['L'], df['B'], c='blue', zorder=3)
        for i, row in df.iterrows():
            ax.annotate(f"{row['H']:.0f} m\n{row['Station']}", (row['L'], row['B'] + 0.01), fontsize=10, color='blue')

        # строим геодезическую сеть триангуляции
        tri = Delaunay(df[['B', 'L']])
        edges = set()
        # получаем все ребра треугольников и добавляем их в множество edges
        for n in range(tri.nsimplex):
            simplex = tri.simplices[n]
            for i in range(3):
                for j in range(i, 3):
                    if i != j:
                        edges.add((simplex[i], simplex[j]))

        # рисуем ребра треугольников на графике и добавляем текст с длиной линий
        edge_labels = set()
        for edge in edges:
            x1, y1 = df.iloc[edge[0]][['L', 'B']]
            x2, y2 = df.iloc[edge[1]][['L', 'B']]
            length = geodesic((y1, x1), (y2, x2)).meters
            if length <= 80000:  # проверяем длину линии
                label = f'{length:.0f} m'
                edge_labels.add((min(edge), max(edge), label))
                ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
            '''
            label = f'{geodesic((y1, x1), (y2, x2)).meters:.0f} m'
            edge_labels.add((min(edge), max(edge), label))
            ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
            '''
        # рисуем текст с информацией о длинах линий
        for label in edge_labels:
            edge = label[:2]
            label_text = label[2]
            x1, y1 = df.iloc[edge[0]][['L', 'B']]
            x2, y2 = df.iloc[edge[1]][['L', 'B']]
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2
            rotation = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if rotation > 90:
                rotation -= 180
            elif rotation < -90:
                rotation += 180
            bbox = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'pad': 0.3}
            ax.annotate(label_text, (label_x, label_y), rotation=rotation, bbox=bbox, ha='center', va='center',
                        fontsize=6,
                        color='red', zorder=4)

            '''
            dist = geodesic((y1, x1), (y2, x2)).meters
            ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
            ax.annotate(f"{dist:.0f} m", ((x1 + x2) / 2, (y1 + y2) / 2), fontsize=8, color='red')
            '''
        plt.show()

    # Заполняем DataFrame координатами для каждой даты и каждого геодезического пункта
    def create_dataframe(df, date_list):
        points = df.to_dict('records')
        coordinates = [(p['X'], p['Y'], p['Z']) for p in points]
        stations = [p['Station'] for p in points]
        rows = []
        for date in date_list:
            for i, station in enumerate(stations):
                new_row = {'Date': date,
                           'Station': station,
                           'X': coordinates[i][0],
                           'Y': coordinates[i][1],
                           'Z': coordinates[i][2]}
                rows.append(new_row)
        return rows

    # добавляем годовые и полугодовые колебания
    def harmonics(df, date_list, periods_in_year):
        stations = SyntheticData.unique_names(df)
        x_harmonic_array = []
        y_harmonic_array = []
        z_harmonic_array = []
        t = 0
        for date in date_list:
            t += 1
            for i, station in enumerate(stations):
                x_harmonic = X_cosine_annual * m.cos(2 * m.pi * t / periods_in_year) \
                    + X_sine_annual * m.sin(2 * m.pi * t / periods_in_year) + \
                    X_cosine_semiannual * m.cos(4 * m.pi * t / periods_in_year) + \
                    X_sine_semiannual * m.sin(4 * m.pi * t / periods_in_year)

                y_harmonic = Y_cosine_annual * m.cos(2 * m.pi * t / periods_in_year) \
                    + Y_sine_annual * m.sin(2 * m.pi * t / periods_in_year) + \
                    Y_cosine_semiannual * m.cos(4 * m.pi * t / periods_in_year) + \
                    Y_sine_semiannual * m.sin(4 * m.pi * t / periods_in_year)

                z_harmonic = Z_cosine_annual * m.cos(2 * m.pi * t / periods_in_year) \
                    + Z_sine_annual * m.sin(2 * m.pi * t / periods_in_year) + \
                    Z_cosine_semiannual * m.cos(4 * m.pi * t / periods_in_year) + \
                    Z_sine_semiannual * m.sin(4 * m.pi * t / periods_in_year)

                x_harmonic_array.append(x_harmonic)
                y_harmonic_array.append(y_harmonic)
                z_harmonic_array.append(z_harmonic)

        df['X'] = df['X'].add(x_harmonic_array, axis=0)
        df['Y'] = df['Y'].add(y_harmonic_array, axis=0)
        df['Z'] = df['Z'].add(z_harmonic_array, axis=0)
        return df

    # линейный тренд
    def linear_trend(df, date_list, periods_in_year):
        stations = SyntheticData.unique_names(df)
        row_vx = []
        row_vy = []
        row_vz = []
        t = 0

        for date in date_list:
            t += 1
            for i, station in enumerate(stations):
                vx_on_period = vx[2] * (t / periods_in_year)
                vy_on_period = vy[2] * (t / periods_in_year)
                vz_on_period = vz[2] * (t / periods_in_year)

                row_vx.append(vx_on_period)
                row_vy.append(vy_on_period)
                row_vz.append(vz_on_period)
        df['X'] = df['X'].add(row_vx, axis=0)
        df['Y'] = df['Y'].add(row_vy, axis=0)
        df['Z'] = df['Z'].add(row_vz, axis=0)
        return df

    # шум
    def noise(df, num_periods):
        stations = SyntheticData.unique_names(df)
        kappa = -1                       # Flicker noise
        N = num_periods * len(stations)  # size of the file
        h = np.zeros(2 * N)              # Note the size : 2N
        h[0] = 1                         # Eq. (25)

        for i in range(1, N):  # Генерация шума
            h[i] = (i - kappa / 2 - 1) / i * h[i - 1]

        v = np.zeros(2 * N)  # zero-padded N:2N
        v[0:N] = np.random.normal(loc=0.0, scale=0.002, size=N)

        w = np.real(fft.ifft(fft.fft(v) * fft.fft(h)))  # Разложение по методу Фурье

        df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].add(w[0:N], axis=0)
        return df

    # импульс (скачок измерений)
    def impulse(df, num_periods):
        stations = SyntheticData.unique_names(df)
        N = num_periods * len(stations)  # size of the file
        impulse_array = unit_impulse(N, 520) * 0.02
        df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].add(impulse_array, axis=0)
        return df

'''
data = SyntheticData.random_points(56.012255, 82.985018, 141.687, 0.5, 10)
data_xyz = SyntheticData.my_geodetic2ecef(data)
Synthetic_data = pd.DataFrame(SyntheticData.create_dataframe(data_xyz, date_list))
print(Synthetic_data)
SyntheticData.impulse(Synthetic_data)
plt.show()
'''
'''
SyntheticData.harmonics(Synthetic_data, date_list)

SyntheticData.linear_trend(Synthetic_data, date_list)

SyntheticData.noise(Synthetic_data)

Synthetic_data.to_csv('data_test.csv', sep=';', header=True, index=True)

data_blh = SyntheticData.my_ecef2geodetic(Synthetic_data)

print(data_blh)

'''


