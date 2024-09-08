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
# import colorednoise


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


class SyntheticData:
    @staticmethod
    def my_geodetic2ecef(df):
        """
        Converts coordinates from the BLH (latitude, longitude, height) system
        to the XYZ (ECEF) system.

        Args:
            df (DataFrame): DataFrame containing coordinates in the BLH system

        Returns:
            DataFrame: DataFrame containing coordinates in the XYZ system
        """

        df = pd.DataFrame(df)

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

    @staticmethod
    def my_ecef2geodetic(df):
        """
        Converts coordinates from the XYZ (ECEF) system to the BLH (latitude,
        longitude, height) system.

        Args:
            df (DataFrame): DataFrame containing coordinates in the XYZ system

        Returns:
            DataFrame: DataFrame containing coordinates in the BLH system
        """
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

    def random_points_old(B, L, H, zone, amount, method, min_dist, max_dist):
        """
        Генерирует случайную сеть пунктов в заданной зоне.

        Параметры:
        B (float): начальная широта
        L (float): начальная долгота
        H (float): начальная высота
        zone (float): зона генерации пунктов
        amount (int): количество пунктов для генерации
        method (str): метод генерации ('consistent' или 'centralized')
        min_dist (float): минимальное расстояние между пунктами
        max_dist (float): максимальное расстояние между пунктами

        Возвращает:
        DataFrame: DataFrame с информацией о случайных пунктах
        """
        if min_dist > max_dist:
            raise ValueError("Minimum distance cannot be greater than maximum distance")

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
        return pd.DataFrame(points)

    def random_points(X, Y, Z, amount, min_dist, max_dist):
        """
        Generates a random network of points within a specified zone.

        Args:
            X (float): Initial longitude
            Y (float): Initial latitude
            Z (float): Initial height
            amount (int): Number of points to generate
            min_dist (float): Minimum distance between points
            max_dist (float): Maximum distance between points

        Returns:
            DataFrame: DataFrame with information about the random points
    """
        if min_dist > max_dist:
            raise ValueError("Minimum distance cannot be greater than maximum distance")

        start_point = {'Date': 'None', 'Station': 'NSK1', 'X': X, 'Y': Y, 'Z': Z}  # исходный пункт

        points = [start_point]

        while len(points) < amount:
            # Generate a random direction
            direction = np.random.uniform(0, 2 * np.pi)

            # Generate a random distance within the specified range
            distance = np.random.uniform(min_dist, max_dist)

            # Calculate the new point's coordinates
            prev_x, prev_y, prev_z = points[-1]['X'], points[-1]['Y'], points[-1]['Z']

            x = prev_x + distance * np.cos(direction)
            y = prev_y + distance * np.sin(direction)
            z = prev_z + np.random.uniform(-25, 25)

            station_name = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4))
            point = {'Date': 'None', 'Station': station_name, 'X': x, 'Y': y, 'Z': z}

            # Build a KD tree from the Cartesian coordinates
            tree = KDTree([[point['X'], point['Y'], point['Z']] for point in points])

            if len(points) < 2:
                neighbours = 1
            else:
                neighbours = 2

            # Find the nearest point to the new point
            dist, idx = tree.query((x, y, z), k=neighbours)

            if np.all(min_dist <= dist) and np.all(dist <= max_dist):
                # Add new point to list
                points.append(point)

        return points

    def triangulation(df, subplot, canvas, max_baseline=None):
        """
        Builds a triangulation scheme of the network.

        Args:
            df (DataFrame): DataFrame with coordinates of points
            subplot (matplotlib.axes.Axes): Axes object for plotting
            canvas (matplotlib.figure.FigureCanvas): Canvas object for plotting
            max_baseline (float): Maximum length of the baseline

        Returns:
            None
        """
        fig = canvas
        ax = subplot
        ax.clear()  # очищаем график перед новой отрисовкой

        # рисуем точки на графике
        ax.scatter(df['L'], df['B'], c='blue', zorder=5)
        for i, row in df.iterrows():
            ax.annotate(f"{row['H']:.0f} m\n{row['Station']}", (row['L'], row['B'] + 0.01),
                        fontsize=10,
                        color='blue',
                        zorder=6)

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
            if max_baseline:
                if length <= max_baseline:  # проверяем длину линии
                    label = f'{length:.0f} m'
                    edge_labels.add((min(edge), max(edge), label))
                    ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
            else:
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

        ax.set_aspect('equal')

        return fig, ax

    def create_dataframe(points, date_list):
        """
        Creates a DataFrame with coordinates for each date and each geodetic point.

        Args:
            points (list): List of points with coordinates
            date_list (list): List of dates

        Returns:
            DataFrame: DataFrame with coordinates for each date and each geodetic point
        """
        coordinates = [(p['X'], p['Y'], p['Z']) for p in points]
        stations = [p['Station'] for p in points]
        data = []

        for date in date_list:
            for i, station in enumerate(stations):

                data.append({'Date': date,
                             'Station': station,
                             'X': coordinates[i][0],
                             'Y': coordinates[i][1],
                             'Z': coordinates[i][2]})

        df = pd.DataFrame(data)

        return df

    def create_dataframe_old(df, date_list):
        """
        Creates a DataFrame with coordinates for each date and each geodetic point.

        Args:
            df (DataFrame): DataFrame with coordinates
            date_list (list): List of dates

        Returns:
            DataFrame: DataFrame with coordinates for each date and each geodetic point
        """
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

    def harmonics(df, date_list, periods_in_year):
        """
        Добавление годовых и полугодовых колебаний в временном ряде.

        Параметры:
        df (DataFrame): файл с временным рядом
        date_list (list): список дат
        periods_in_year (int): количество периодов измерений в году

        Возвращает:
        DataFrame: файл с временным рядом, содержащим годовые и полугодовые колебания
        """
        stations = df['Station'].unique()
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

    @staticmethod
    def c(t: float):
        """
        Calculates an amplitude change factor for seasonal signal generation

        Args:
            t (float): Time value

        Returns:
            float: Coefficient value
        """
        return 2 * np.exp(0.3 * np.sin(t))

    @staticmethod
    def harmonics_new(interval, periods, i, j, k, l):
        """
        Generates a harmonic signal based on user input parameters.

        Returns:
            tuple: Time array and simulated data array
        """

        num = (end_date - start_date).days * interval

        t = np.linspace(start_date, end_date, num)

        s = (i * np.sin(2 * np.pi * t) + j * np.cos(2 * np.pi * t) + SyntheticData.c(t) * np.sin(2 * np.pi * t)
             + SyntheticData.c(t) * np.cos(2 * np.pi * t) + k * np.sin(4 * np.pi * t) + l * np.cos(4 * np.pi * t))

        harmonic = s

        return t, harmonic

    def linear_trend(df, date_list, periods_in_year):
        """
        Adds a linear trend to a time series.

        Args:
            df (DataFrame): DataFrame containing the time series
            date_list (list): List of dates
            periods_in_year (int): Number of periods in a year

        Returns:
            DataFrame: DataFrame with the added linear trend
        """
        stations = df['Station'].unique()
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

    def noise(df, num_periods):
        """
        Injects noise into a time series.

        Args:
            df (DataFrame): DataFrame containing the time series
            num_periods (int): Number of periods in the time series

        Returns:
            DataFrame: DataFrame with the added noise
        """
        stations = df['Station'].unique()
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
        stations = df['Station'].unique()
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

    def impulse(df, impulse_size, target_date=None, num_stations=1, random_dates=0):
        """
        Generates impulses in the time series.

        Args:
            df (DataFrame): DataFrame with time series
            impulse_size (float): Size of the impulse in meters
            target_date (datetime.date or list of datetime.date): Date of measurements to add impulse
            num_stations (int): Number of stations to add impulse
            random_dates (int): Number of random dates to add impulse

        Returns:
            None or list of datetime.date: If random_dates > 0, returns a list of random dates
        """
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


