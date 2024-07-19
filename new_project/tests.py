import numpy as np
from numpy.linalg import pinv
import pandas as pd
from pandas import DataFrame
from colorama import init, Fore
from scipy.stats import ttest_1samp, shapiro, chi2, f_oneway, chisquare
from scipy import signal


class Tests:
    def __init__(self, df: DataFrame):
        """
        Инициализация класса CongruencyTest.

        Параметры:
            df (DataFrame): DataFrame с временным рядом
        """
        self.df = df
        self.dates = df['Date'].unique()  # Store the unique dates

    def congruency_test(self, df, method: str,
                        calculation: str = "all_dates",
                        start_date: str = None,
                        end_date: str = None,
                        threshold: float = 0.05,
                        print_log: bool = True):
        """
        Функция геометрического теста конгруэнтности геодезической сети на начальную (start_date)
        и i-ую (end_date) эпохи. Конгруэнтность проверяется при помощи T-теста (ttest) и теста Хи-квадрат (chi2)

        Параметры:
            df (DataFrame):               DataFrame с данными для теста конгруэнтности.
            method (str):                 Метод теста конгруэнтности (line_based или coordinate_based).
            calculation (str, optional):  Тип расчета (все даты или конкретная дата).
            start_date (str, optional):   Начальная дата для расчета.
            end_date (str, optional):     Конечная дата для расчета.
            threshold (float, optional):  Пороговое значение для теста
        """

        ttest_rejected_dates = []
        chi2_rejected_dates = []

        if calculation == "all_dates":
            self._run_all_dates(df, method, threshold, ttest_rejected_dates, chi2_rejected_dates)
        elif calculation == "specific_date":
            self._run_specific_date(df, method, start_date, end_date, threshold, ttest_rejected_dates,
                                    chi2_rejected_dates)

        if print_log:
            self._print_results(ttest_rejected_dates, chi2_rejected_dates)

        return ttest_rejected_dates, chi2_rejected_dates

    def _run_all_dates(self, df, method, threshold, ttest_rejected_dates, chi2_rejected_dates):
        """
        Вычисление для всех дат в файле.

        Параметры:
            df (DataFrame):                   DataFrame с временным рядом.
            method (str):                     Метод теста конгруэнтности (line_based или coordinate_based).
            threshold (float):                Пороговое значение для теста.
            ttest_rejected_dates (list):      Список дат, для которых нулевая гипотеза отвергнута по T-тесту.
            chi2_rejected_dates (list):       Список дат, для которых нулевая гипотеза отвергнута по тесту Хи-квадрат.
        """
        # Получение списка уникальных дат в DataFrame
        dates = df['Date'].unique()
        # Сортировка дат
        dates = sorted(dates)
        for i in range(len(dates) - 1):
            start_date = dates[i]
            end_date = dates[i + 1]
            print(f"Calculating for dates: {start_date} and {end_date}")
            # Вызов функции congruency_test для каждой пары дат
            self._run_specific_date(df, method, start_date, end_date, threshold, ttest_rejected_dates,
                                    chi2_rejected_dates)

    def _run_specific_date(self, df, method, start_date, end_date, threshold, ttest_rejected_dates, chi2_rejected_dates):
        """
        Вычисление для конкретной пары дат.

        Параметры:
            df (DataFrame):                DataFrame с временным рядом.
            method (str):                  Метод теста конгруэнтности (line_based или coordinate_based).
            start_date (str):              Начальная дата для расчета.
            end_date (str):                Конечная дата для расчета.
            threshold (float):             Пороговое значение для теста.
            ttest_rejected_dates (list):   Список дат, для которых нулевая гипотеза отвергнута по T-тесту.
            chi2_rejected_dates (list):    Список дат, для которых нулевая гипотеза отвергнута по тесту Хи-квадрат.
        """
        stations = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'Station'].unique()
        raz_list = self._calculate_raz_list(df, method, start_date, end_date, stations)
        std_dev = np.std(np.array(raz_list))
        sigma_0 = 0.01 #0.0075 0.005
        #  sigma_0 = std_dev
        print(f"std dev of data: {sigma_0}")
        print('Shapiro:', shapiro(raz_list))

        '''ttest_result, pvalue = self._perform_ttest(raz_list, threshold)
        print("T-test: ", end="")
        if ttest_result:
            ttest_rejected_dates.append((start_date, end_date, pvalue))
            print(Fore.RED + f"Нулевая гипотеза отвергается, pvalue = {round(pvalue, 3)}")
        else:
            print(Fore.GREEN + f"Нулевая гипотеза не отвергается, pvalue = {round(pvalue, 3)}")
        print(Fore.RESET, end="")'''

        chi2_result, K, test_value = self._perform_chi2_test(raz_list, sigma_0, threshold)
        print("Chi-2: ", end="")
        if chi2_result:
            chi2_rejected_dates.append((start_date, end_date))
            print(Fore.RED + f"Нулевая гипотеза отвергается, testvalue = {round(test_value, 3)}, K = {round(K, 3)}")
        else:
            print(Fore.GREEN + f"Нулевая гипотеза не отвергается,"
                               f" testvalue = {round(test_value, 3)}, K = {round(K, 3)}")
        print(Fore.RESET)

    def _calculate_raz_list(self, df, method, start_date, end_date, stations):
        """
        Вычисление списка разностей между эпохами для теста конгруэнтности.

        Параметры:
            df (DataFrame):      DataFrame с временным рядом.
            method (str):        Метод теста конгруэнтности (line_based или coordinate_based).
            start_date (str):    Начальная дата для расчета.
            end_date (str):      Конечная дата для расчета.
            stations (list):     Список станций для расчета.

        Возвращает:
            list: Список разностей между эпохами.
        """
        raz_list = []
        if method == 'line_based':
            ''' метод, основанный на вычислении разностей базовых линий на разные эпохи '''
            values_0 = self._calculate_baselines(df, start_date, stations)   # Словарь координат на начальную эпоху
            values_i = self._calculate_baselines(df, end_date, stations)     # Словарь координат на i-ую эпоху
            # Заполнение списка разностями
            for line1 in values_0:
                for line2 in values_i:
                    if line1[0] == line2[0] and line1[1] == line2[1]:
                        raz_list.append(line1[2] - line2[2])
        elif method == 'coordinate_based':
            ''' метод, основанный на вычислении разностей координат пунктов на разные эпохи '''
            values_0 = self._calculate_coordinates(df, start_date, stations)  # Словарь координат на начальную эпоху
            values_i = self._calculate_coordinates(df, end_date, stations)    # Словарь координат на i-ую эпоху
            # Заполнение списка разностями
            for station, coord_0 in values_0.items():
                if station in values_i:
                    coord_i = values_i[station]
                    raz = np.subtract(coord_i, coord_0)
                    raz_list.extend(raz)
        return raz_list

    def _calculate_baselines(self, df, date, stations):
        """
        Вычисление базовых линий в сети на заданную эпоху.

        Параметры:
            df (DataFrame):     DataFrame с временным рядом.
            date (str):         Дата для расчета.
            stations (list):    Список станций.

        Возвращает:
            list: Список базовых линий в сети на заданную эпоху.
        """
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

    def _calculate_coordinates(self, df, date, stations):
        """
        Вычисление координат станций на заданную эпоху.

        Параметры:
            df (DataFrame):    DataFrame с временным рядом.
            date (str):        Дата для расчета.
            stations (list):   Список станций.

        Возвращает:
            dict: Словарь координат станций на заданную эпоху.
        """
        coords = {}  # Словарь для хранения координат станций
        for station in stations:
            station_df = df.loc[(df['Date'] == date) & (df['Station'] == station)]
            coords[station] = (station_df['X'].values[0], station_df['Y'].values[0], station_df['Z'].values[0])
        return coords

    def _perform_ttest(self, raz_list, threshold):
        """
        Выполнение T-теста.

        Параметры:
            raz_list (list):    Список разностей для T-теста.
            threshold (float):  Пороговое значение для T-теста.

        Возвращает:
            bool:   Результат T-теста (гипотеза отвергнута / нет).
            float:  p-value T-теста.
        """
        pvalue = ttest_1samp(a=raz_list, popmean=0, nan_policy='omit')[1]
        if pvalue <= threshold:
            return True, pvalue
        else:
            return False, pvalue

    def _perform_chi2_test(self, raz_list, sigma_0, threshold):
        """
        Выполнение теста Хи-квадрат.

        Параметры:
            raz_list (list):    Список разностей для теста Хи-квадрат.
            sigma_0 (float):    Сигма-ноль для теста Хи-квадрат.
            threshold (float):  Пороговое значение для теста Хи-квадрат.

        Возвращает:
            bool:    Результат теста Хи-квадрат (гипотеза отвергнута / нет).
            float:   K-значение теста Хи-квадрат.
            float:   testvalue теста Хи-квадрат.

        Переменные:
            d.transpose()  - транспонируем матрицу d
            dot(pinv(Qdd)) - скалярное произведение транспонированного массива d с псевдообратным значением Qdd
            dot(d)         - вычисляет скалярное произведение результата с исходным массивом d.
            sigma_0 ** 2   - Все выражение делится на данное значение, которое предст. собой СКО шума.
        """
        d = np.array(raz_list)
        Qdd = np.eye(d.shape[0])
        K = d.transpose().dot(pinv(Qdd)).dot(d) / (sigma_0 ** 2)
        test_value = chi2.ppf(df=d.shape[0], q=threshold)
        if K > test_value:
            return True, K, test_value
        else:
            return False, K, test_value

    def find_offset_points(self, df, method):
        offset_points = []
        stations = df['Station'].unique()
        for station in stations:
            temp_df = df[df['Station'] != station]
            ttest_rejected_dates, chi2_rejected_dates = self.congruency_test(df=temp_df, method=method,
                                                                             calculation="all_dates", print_log=True)
            if not (ttest_rejected_dates or chi2_rejected_dates):
                offset_points.append(station)
        return offset_points

    def find_offset_points_2(self, df, method):
        offset_points = []
        ttest_rejected_dates, chi2_rejected_dates = self.congruency_test(df, method, calculation="all_dates")
        rejected_dates = ttest_rejected_dates + chi2_rejected_dates
        print(len(rejected_dates))
        print('rej dates: ', rejected_dates)

        for start_date, end_date in rejected_dates:
            temp_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            for station in temp_df['Station'].unique():
                print(f'calculating for station {station}')
                temp_temp_df = temp_df[temp_df['Station'] != station]
                ttest_rejected, chi2_rejected = self.congruency_test(temp_temp_df, method, calculation="specific_date",
                                                                     start_date=start_date, end_date=end_date,
                                                                     print_log=False)
                if not (ttest_rejected or chi2_rejected):
                    offset_points.append((start_date, end_date, station))

        return offset_points

    def _print_results(self, ttest_rejected_dates, chi2_rejected_dates):
        """
        Печать результатов теста конгруэнтности.

        Параметры:
            ttest_rejected_dates (list):   Список дат, для которых нулевая гипотеза отвергнута по T-тесту.
            chi2_rejected_dates (list):    Список дат, для которых нулевая гипотеза отвергнута по тесту Хи-квадрат.
        """
        print("----------------------------")
        print(f"Всего выполнено тестов: {len(self.dates)}.")
        print(f"Хи-квадрат, не отвергнуто: {len(self.dates) - len(chi2_rejected_dates)}, "
              f"отвергнуто: {len(chi2_rejected_dates)} ({len(chi2_rejected_dates) / len(self.dates) * 100:.2f}%)")
        print(f"Т-тест, не отвергнуто: {len(self.dates) - len(ttest_rejected_dates)}, "
              f"отвергнуто: {len(ttest_rejected_dates)} ({len(ttest_rejected_dates) / len(self.dates) * 100:.2f}%)")

        print(f"Chi2 rejected dates: {chi2_rejected_dates}")


def main():

    df = pd.read_csv('Data/testfile_20points_3with_impulse_1impulse(3cm)_2022_05_08.csv', delimiter=';')

    test = Tests(df)
    #test.congruency_test(df=df, method='coordinate_based')
    offset_points = test.find_offset_points_2(df=df, method='coordinate_based')
    print("Candidate points with offsets:", offset_points)


if __name__ == "__main__":
    main()
