import pandas as pd
from pandas import DataFrame
import numpy as np
import math as m
import random
from numpy import fft
import geopy.distance
from geopy.distance import geodesic
from scipy.spatial import KDTree, Delaunay
from scipy.signal import unit_impulse, butter, lfilter
import pymap3d
import matplotlib.pyplot as plt
import colorednoise


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
    def random_points(B, L, H, zone, amount, method, min_dist, max_dist):
        start_point = {'Date': 'None', 'Station': 'NSK1', 'B': B, 'L': L, 'H': H}  # исходный пункт

        points = [start_point]

        while len(points) < amount:
            Station = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4))
            B = np.random.uniform(start_point['B'] - zone, start_point['B'] + zone)
            L = np.random.uniform(start_point['L'] - zone, start_point['L'] + zone)
            H = np.random.uniform(start_point['H'] - 25, start_point['H'] + 25)
            point = {'Date': 'None', 'Station': Station, 'B': B, 'L': L, 'H': H}

            # Convert all points to Cartesian coordinates
            cartesian_points = [pymap3d.geodetic2ecef(point['B'], point['L'], point['H']) for point in points]

            # Build a KD tree from the Cartesian coordinates
            tree = KDTree(cartesian_points)

            if len(points) < 2:
                neighbours = 1
            else:
                neighbours = 2

            # Find the nearest point to the new point
            dist, idx = tree.query(pymap3d.geodetic2ecef(point['B'], point['L'], point['H']), k=neighbours)

            # Convert the distance from Cartesian to geodetic distance
            geod_dist = geopy.distance.distance(kilometers=dist).meters / 1000

            # Check if the distance is within the desired range
            if np.all(min_dist <= geod_dist) and np.all(geod_dist <= max_dist):
                # Add new point to list
                points.append(point)
                if method == 'consistent':
                    start_point = point
                elif method == 'centralized':
                    pass

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
            '''
        return pd.DataFrame(points)

    # Схема сети (метод триангуляции)
    def triangulation(df, subplot, canvas, max_baseline):
        fig = canvas
        ax = subplot
        ax.clear()  # очищаем график перед новой отрисовкой

        # рисуем точки на графике
        scatter = ax.scatter(df['L'], df['B'], c='blue', zorder=3)
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
            if length <= max_baseline:  # проверяем длину линии
                label = f'{length:.0f} m'
                edge_labels.add((min(edge), max(edge), label))
                ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)

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
        canvas.draw()

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

    # импульс старая версия
    '''def impulse(df, num_periods):
        stations = SyntheticData.unique_names(df)
        N = num_periods * len(stations)  # size of the file

        """
        это код фильтра (Butterworth lowpass filter)
        imp = unit_impulse(N, int(N/2)) * 0.1
        b, a = butter(4, 0.2)
        response = lfilter(b, a, imp)
        """

        impulse_array = unit_impulse(N, int(N/2)) * 0.1
        df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].add(impulse_array, axis=0)
        return df'''

    # импульс (скачок измерений)
    def impulse(df, impulse_size, target_date=None, num_stations=1, random_dates=0):
        """ Функция генерации импульсов во временном ряде.
        Входные данные:
        df - файл с временным рядом (DataFrame),
        impulse_size - размер импульса (в метрах),
        target_date - дата измерений, в которые вносится импульс (можно задать списком несколько дат),
        num_stations - количество станций в сети, в которые вносится импульс (по умолчанию 1),
        random_dates - количество случайных дат для создания импульсов (по умолчанию 0) """

        if target_date is None and random_dates <= 0:
            raise ValueError("Either 'target_date' or 'random_dates' must be provided.")

        if target_date is not None and random_dates > 0:
            raise ValueError("Only one of 'target_date' or 'random_dates' can be provided.")

        if isinstance(target_date, list):
            target_dates = target_date
        elif target_date is not None:
            target_dates = [target_date]
        else:
            random_dates = min(random_dates, len(df['Date'].unique()))
            target_dates = random.sample(list(df['Date'].unique()), random_dates)
        impulse_array = [impulse_size] * 3  # список с импульсами

        # Выбираем случайные станции из DataFrame и сохраняем их в переменной
        random_stations = df.drop_duplicates(subset='Station')
        random_stations = random_stations.sample(n=num_stations, replace=False)

        for target_date in target_dates:
            # Используем выбранные случайные станции для фильтрации DataFrame по заданной дате
            filtered_df = df[(df['Date'] == target_date) & df['Station'].isin(random_stations['Station'])]

            # Обновляем координаты в основном DataFrame
            for index, row in filtered_df.iterrows():
                station = row['Station']
                date = row['Date']
                df.loc[(df['Station'] == station) & (df['Date'] == date), ['X', 'Y', 'Z']] += impulse_array

        if random_dates > 0:
            return target_dates, random_stations['Station']


    # вывод графика временного ряда
    def time_series_plot(file_name, station_name):
        df = pd.read_csv(file_name, delimiter=';')
        df = file_name.set_index('Station')
        df_station = df.loc[station_name]
        df_station = df_station.reset_index()
        df_station_X = df_station['X']
        df_station_Y = df_station['Y']
        df_station_Z = df_station['Z']

        plt.figure(1)
        df_station_X.plot()
        plt.xlabel('Weeks')
        plt.ylabel('Amplitude')
        plt.title(f'{station_name} X-coordinate')

        plt.figure(2)
        df_station_Y.plot()
        plt.xlabel('Weeks')
        plt.ylabel('Amplitude')
        plt.title(f'{station_name} Y-coordinate')

        plt.figure(3)
        df_station_Z.plot()
        plt.xlabel('Weeks')
        plt.ylabel('Amplitude')
        plt.title(f'{station_name} Z-coordinate')

        plt.show()

