import numpy as np
import scipy.stats
from numpy.linalg import pinv
import pandas as pd
from pandas import DataFrame
from colorama import init, Fore
from scipy.stats import ttest_1samp, shapiro, chi2, f_oneway, chisquare
from scipy import signal
from Synthetic_data import SyntheticData
import matplotlib.pyplot as plt

"""
Это устаревшая версия. Новая версия находится в файле tests.py
"""


class Tests(DataFrame):
    def __init__(self):
        super().__init__()

    def congruency_test(df, method="line_based", calculation="all_dates",
                        start_date=None, end_date=None, threshold=0.05):
        """ Функция геометрического теста конгруэнтности геодезической сети на начальную (start_date)
        и i-ую (end_date) эпохи. Конгруэнтность проверяется при помощи T-теста (ttest) и теста Хи-квадрат (chi2) """

        ttest_rejected_dates = []
        chi2_rejected_dates = []  # список дат с отвергнутой гипотезой
        chi2_ai_rejected_dates = []
        f_rejected_dates = []

        def all_dates():
            """ вычисление для всех дат в файле """
            calculations_count = 0  # счетчики для измерений
            # Получение списка уникальных дат в DataFrame
            dates = df['Date'].unique()
            # Сортировка дат
            dates = sorted(dates)
            for i in range(len(dates) - 1):
                start_date = dates[i]
                end_date = dates[i + 1]
                print(f"Calculating for dates: {start_date} and {end_date}")
                # Вызов функции congruency_test для каждой пары дат
                specific_date(df, start_date=start_date, end_date=end_date)
                calculations_count += 1
            return calculations_count

        def specific_date(df=df, start_date=start_date, end_date=end_date):
            """ вычисление для конкретной пары дат """
            if start_date is None:
                start_date = df['Date'].min()  # Если парам. start_date не задан, берем начальную дату документа
            if end_date is None:
                end_date = df['Date'].max()  # Если парам. end_date не задан, берем конечную дату документа

            # Отбор уникальных станций в DataFrame
            stations = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'Station'].unique()

            def baselines(date):
                """ функция для вычисления базовых линий в сети на заданную эпоху """
                lines = []  # Список для хранения базовых линий
                computed_pairs = set()  # Множество для хранения уже вычисленных пар пунктов
                for station in stations:  # Цикл для определения БЛ
                    station_df = df.loc[(df['Date'] == date) & (df['Station'] == station)]  # координаты на дату
                    for other_station in stations:  # проверяем, чтобы алгоритм не сравнивал один пункт сам с собой
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
                values_i = baselines(end_date)  # и i-ую эпоху
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

            # T-тест
            pvalue = ttest_1samp(a=raz_list, popmean=0, nan_policy='omit')[1]

            # Chi2 test
            sigma_0 = 0.0075              # СКО шума
            # std_dev = np.std(np.array(raz_list))
            # sigma_0 = std_dev
            print('std dev of data:', sigma_0)
            d = np.array(raz_list)  # Конвертирование в массив Numpy
            Qdd = np.eye(d.shape[0])  # единичная матрица формы d
            '''
            статистика теста К
            d.transpose()  - транспонируем матрицу d
            dot(pinv(Qdd)) - скалярное произведение транспонированного массива d с псевдообратным значением Qdd
            dot(d)         - вычисляет скалярное произведение результата с исходным массивом d.
            sigma_0 ** 2   - Все выражение делится на данное значение, которое предст. собой СКО шума.
            '''
            K = d.transpose().dot(pinv(Qdd)).dot(d) / (sigma_0 ** 2)
            test_value = chi2.ppf(df=d.shape[0], q=threshold)

            # Calculate the expected values (assuming equal probabilities)
            expected = np.ones_like(raz_list) * 0.00001
            # Calculate the test statistic (K)
            K_ai = np.sum(((np.array(raz_list) - expected) ** 2) / expected)
            # Calculate the critical value
            alpha = 0.05  # significance level
            deg_of_freedom = len(raz_list) - 1  # degrees of freedom
            critical_value = chi2.ppf(q=1 - alpha, df=deg_of_freedom)

            """
            это статистика через scipy.chisquare - пока не разобрался, не используем

            exp = np.ones_like(raz_list)
            statistics_2 = chisquare(f_obs=raz_list, f_exp=exp)
            print(statistics_2)
            """

            # Shapiro test
            print('Shapiro:', shapiro(raz_list))

            print("T-test: ", end="")
            if pvalue <= threshold:
                ttest_rejected_dates.append((start_date, end_date, pvalue))
                print(Fore.RED + f"Нулевая гипотеза отвергается, pvalue = {round(pvalue, 3)}")
            else:
                print(Fore.GREEN + f"Нулевая гипотеза не отвергается, pvalue = {round(pvalue, 3)}")
            print(Fore.RESET, end="")

            print("chi2-test: ", end="")
            if K > test_value:
                chi2_rejected_dates.append((start_date, end_date, test_value, K))
                print(Fore.RED + f"Нулевая гипотеза отвергается, testvalue = {round(test_value, 3)}, K = {round(K, 3)}")
            else:
                print(Fore.GREEN + f"Нулевая гипотеза не отвергается,"
                                   f" testvalue = {round(test_value, 3)}, K = {round(K, 3)}")
            print(Fore.RESET, end="")

            print("chi2-test by AI: ", end="")
            if K_ai > critical_value:
                chi2_ai_rejected_dates.append((start_date, end_date, critical_value, K_ai))
                print(
                    Fore.RED + f"Нулевая гипотеза отвергается, testvalue = {round(critical_value, 3)}, K = {round(K_ai, 3)}")
            else:
                print(Fore.GREEN + f"Нулевая гипотеза не отвергается,"
                                   f" testvalue = {round(critical_value, 3)}, K = {round(K_ai, 3)}")
            print(Fore.RESET)

        if calculation == "all_dates":
            calc = all_dates()

            """# Вывод списка дат, в которых гипотеза была отвергнута
            if rejected_dates:
                print("Гипотеза отвергнута для следующих пар дат:")
                for date_pair in rejected_dates:
                    print(f"{date_pair[0]} and {date_pair[1]}, "
                          f"testvalue: {round(date_pair[2], 3)}, K: {round(date_pair[3], 3)}")
            else:
                print("Гипотеза не отвергнута ни для одной пары дат.")"""

            print(f"Всего выполнено тестов: {calc}.")
            print(f"F-test, не отвергнуто: {calc - len(f_rejected_dates)}, "
                  f"отвергнуто: {len(f_rejected_dates)}")
            print(f"Хи-квадрат, не отвергнуто: {calc - len(chi2_rejected_dates)}, "
                  f"отвергнуто: {len(chi2_rejected_dates)}")
            print(f"Хи-квадрат AI, не отвергнуто: {calc - len(chi2_ai_rejected_dates)}, "
                  f"отвергнуто: {len(chi2_ai_rejected_dates)}")
            print(f"Т-тест, не отвергнуто: {calc - len(ttest_rejected_dates)}, "
                  f"отвергнуто: {len(ttest_rejected_dates)}")

            """# Поиск даты, в которой гипотеза была отвергнута в паре с предыдущей и последующей датами
            anomaly_dates = []
            # Получение списка уникальных дат в DataFrame
            dates = df['Date'].unique()
            # Сортировка дат
            dates = sorted(dates)
            for i in range(1, len(dates) - 1):
                prev_date = dates[i - 1]
                curr_date = dates[i]
                next_date = dates[i + 1]
                if curr_date in rejected_dates and prev_date in rejected_dates and next_date in rejected_dates:
                    anomaly_dates.append(curr_date)

            if anomaly_dates:
                print("Аномалия обнаружена в следующих датах:")
                for date in anomaly_dates:
                    print(date)
            else:
                print("Аномалий не обнаружено.")"""

        if calculation == "specific_date":
            specific_date(df=df, start_date=start_date, end_date=end_date)

    def detrend_df(df):
        """ функция выполняет детренд файла DataFrame. Увы, пока не работает( """
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


df = pd.read_csv('Data/testfile_20points_2with_impulse_1impulse(3cm)_2022_01_16.csv', delimiter=';')
Tests.congruency_test(df=df, method='line_based', calculation='all_dates')
