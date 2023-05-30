import numpy as np
from numpy.linalg import pinv
import pandas as pd
from pandas import DataFrame
from colorama import init, Fore
from scipy.stats import ttest_1samp, shapiro, chi2
from scipy import signal
from Synthetic_data import SyntheticData
import matplotlib.pyplot as plt


class Tests(DataFrame):
    def __init__(self):
        super().__init__()

    def congruency_test(df, method="line_based", calculation="all_dates",
                        start_date=None, end_date=None, threshold=0.05):
        """ Функция геометрического теста конгруэнтности геодезической сети на начальную (start_date)
        и i-ую (end_date) эпохи. Точность вычисляется при помощи T-теста (ttest) и теста Хи-квадрат (chi2) """
        if start_date is None:
            start_date = df['Date'].min()  # Если парам. start_date не задан, программа задает начальную дату документа
        if end_date is None:
            end_date = df['Date'].max()  # Если парам. end_date не задан, программа задает конечную дату документа

        # Отбор уникальных станций в DataFrame
        stations = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'Station'].unique()

        def baselines(date):
            """ функция для вычисления базовых линий в сети на заданную эпоху """
            lines = []  # Список для хранения базовых линий
            computed_pairs = set()  # Множество для хранения уже вычисленных пар пунктов
            for station in stations:  # Цикл для определения БЛ
                station_df = df.loc[(df['Date'] == date) & (df['Station'] == station)]  # координаты на дату
                for other_station in stations:  # проверяем, чтобы алгоритм не сравнивал один и тот же пункт сам с собой
                    if other_station == station:
                        continue
                    pair = tuple(sorted([station, other_station]))  # сортировка уникальных пар пунктов
                    if pair in computed_pairs:
                        continue  # базовая линия уже вычислена, пропускаем повторное вычисление
                    other_station_df = df.loc[(df['Date'] == date) & (df['Station'] == other_station)]
                    dist = np.sqrt((station_df['X'].values - other_station_df['X'].values) ** 2  # вычисление БЛ
                                   + (station_df['Y'].values - other_station_df['Y'].values) ** 2
                                   + (station_df['Z'].values - other_station_df['Z'].values) ** 2)
                    if len(dist) > 0:
                        line = [station, other_station, dist.mean()]
                        lines.append(line)
                        computed_pairs.add(pair)  # добавляем пару во множество уже вычисленных пар
            return lines

        def coordinates(date):
            """ функция для получения координат станций на заданную эпоху """
            coords = {}  # Словарь для хранения координат станций
            for station in stations:
                station_df = df.loc[(df['Date'] == date) & (df['Station'] == station)]
                coords[station] = (station_df['X'].values[0], station_df['Y'].values[0], station_df['Z'].values[0])
            return coords

        values_0, values_i = None, None
        raz_list = []  # Список разностей на 0 и i эпохи

        ''' метод, основанный на вычислении разностей базовых линий на разные эпохи '''
        if method == 'line_based':
            values_0 = baselines(start_date)  # Словари для базовых линий на начальную
            values_i = baselines(end_date)    # и i-ую эпоху
            # Заполнение списка разностями
            for line1 in values_0:
                for line2 in values_i:
                    if line1[0] == line2[0] and line1[1] == line2[1]:
                        raz_list.append(line1[2] - line2[2])

        ''' метод, основанный на вычислении разностей координат пунктов на разные эпохи '''
        if method == 'coordinate_based':
            values_0 = coordinates(start_date)  # Словарь координат на начальную эпоху
            values_i = coordinates(end_date)  # Словарь координат на i-ую эпоху
            # Заполнение списка разностями
            for station, coord_0 in values_0.items():
                if station in values_i:
                    coord_i = values_i[station]
                    raz = np.subtract(coord_i, coord_0)
                    raz_list.extend(raz)

        print(values_0, '\n', values_i)
        print("raz_list:", raz_list)

        pvalue = ttest_1samp(a=raz_list, popmean=0, nan_policy='omit')[1]  # T-тест

        # Chi2 test
        sigma_0 = 0.005
        d = np.array(raz_list)
        Qdd = np.eye(d.shape[0])
        K = d.transpose().dot(pinv(Qdd)).dot(d) / (sigma_0 ** 2)
        test_value = chi2.ppf(df=d.shape[0], q=threshold)

        # Shapiro test
        print('Shapiro:\n', shapiro(raz_list), '\n')

        print("T-test:")
        if pvalue <= threshold:
            print(Fore.RED + f"Нулевая гипотеза отвергается, pvalue = {round(pvalue, 6)}")
        else:
            print(Fore.GREEN + f"Нулевая гипотеза не отвергается, pvalue = {round(pvalue, 6)}")
        print(Fore.RESET)

        print("chi2-test:")
        print("testvalue:", test_value)
        if K > test_value:
            print(Fore.RED + f"Нулевая гипотеза отвергается, K = {round(K, 6)}")
        else:
            print(Fore.GREEN + f"Нулевая гипотеза не отвергается, K = {round(K, 6)}")
        print(Fore.RESET)

    def detrend_df(df):
        print(df)
        # Получение уникальных названий пунктов
        station_names = df['Station'].unique()

        # Вывод
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df.loc[df['Station'] == 'NSK1', 'X'])
        plt.title('NSK1 X С трендом')
        plt.xlabel('Время')
        plt.ylabel('Значение')

        # Итерация по каждому уникальному пункту
        for station_name in station_names:

            # Выбор только строк, относящихся к текущему пункту
            station_rows = df[df['Station'] == station_name]

            # Получение значений координат для текущего пункта
            x = station_rows['X'].values
            y = station_rows['Y'].values
            z = station_rows['Z'].values

            # Удаление линейного тренда для каждой координаты
            detrended_x = signal.detrend(x)
            detrended_y = signal.detrend(y)
            detrended_z = signal.detrend(z)

            # Обновление DataFrame с результатами
            df.loc[df['Station'] == station_name, 'X'] = detrended_x
            df.loc[df['Station'] == station_name, 'Y'] = detrended_y
            df.loc[df['Station'] == station_name, 'Z'] = detrended_z

        print(df)

        plt.subplot(2, 1, 2)
        plt.plot(df.loc[df['Station'] == 'NSK1', 'X'])
        plt.title('NSK1 X Без тренда')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.tight_layout()
        plt.show()

        return df


"""print('файл с аномалией (10см):')
df = pd.read_csv('Data/testfile_big_impulse.csv', delimiter=';')
Tests.congruency_test(df=df, start_date='2022-12-25', end_date='2023-01-01')

print('зашумленный файл без аномалий:')
df = pd.read_csv('Data/testfile_just_noise.csv', delimiter=';')
Tests.congruency_test(df=df, start_date='2022-12-25', end_date='2023-01-01')

print('чистый файл без аномалий:')
df = pd.read_csv('Data/testfile_no_anomalies.csv', delimiter=';')
Tests.congruency_test(df=df, start_date='2022-12-25', end_date='2023-01-01')"""

df = pd.read_csv('Data/testfile_10points+noise(only)+impulse5cm.csv', delimiter=';')
Tests.congruency_test(df=df, method='coordinate_based', start_date='2022-12-25', end_date='2023-01-01')


"""
df = pd.read_csv('Data/testfile_for_detrend.csv', delimiter=';')
Tests.detrend_df(df)
"""

"""df = pd.read_csv('Data/testfile_for_detrend.csv', delimiter=';')
df = df.loc[df['Station'] == 'NSK1']
df = df['X']

detrended = pd.DataFrame(signal.detrend(df.values, type == 'linear'))"""

