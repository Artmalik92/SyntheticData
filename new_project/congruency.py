import numpy as np
import pandas as pd
from pandas import DataFrame
from colorama import init, Fore
from scipy.stats import ttest_1samp, shapiro


class Tests(DataFrame):
    def __init__(self):
        super().__init__()

    def congruency_test(df, start_date=None, end_date=None, threshold=0.05):
        """ Функция геометрического теста конгруэнтности геодезической сети на начальную (start_date)
        и i-ую (end_date) эпохи. Точность вычисляется при помощи T-теста (ttest) """
        if start_date is None:
            start_date = df['Date'].min()  # Если парам. start_date не задан, программа задает начальную дату документа
        if end_date is None:
            end_date = df['Date'].max()  # Если парам. end_date не задан, программа задает конечную дату документа

        # Отбор уникальных станций в DataFrame
        stations = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'Station'].unique()

        def baselines(date):
            """ Вспомогательная функция для вычисления базовых линий в сети на заданную эпоху """
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

        baselines_0 = baselines(start_date)  # Словари для базовых линий на начальную
        baselines_i = baselines(end_date)    # и i-ую эпоху
        print(baselines_0)
        print(baselines_i)

        raz_list = []  # Список разностей БЛ на 0 и i эпохи

        # Заполнение списка разностями
        for line1 in baselines_0:
            for line2 in baselines_i:
                if line1[0] == line2[0] and line1[1] == line2[1]:
                    raz_list.append(line1[2] - line2[2])
        print(raz_list)

        pvalue = ttest_1samp(a=raz_list, popmean=0, nan_policy='omit')[1]  # T-тест

        if pvalue <= threshold:
            print(Fore.RED + f"Нулевая гипотеза отвергается, pvalue = {round(pvalue, 6)}")
        else:
            print(Fore.GREEN + f"Нулевая гипотеза не отвергается, pvalue = {round(pvalue, 6)}")
        print(Fore.RESET)


df = pd.read_csv('Data/testfile_big_impulse.csv', delimiter=';')
Tests.congruency_test(df=df, start_date='2022-12-25', end_date='2023-01-01')

