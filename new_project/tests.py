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
        Initializes a SyntheticData object.

        Args:
            df (DataFrame): DataFrame containing time series
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
        Performs a congruence test on the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).
            calculation (str, optional): The type of calculation to perform (all_dates or specific_date). Defaults to "all_dates".
            start_date (str, optional): The start date for the calculation. Defaults to None.
            end_date (str, optional): The end date for the calculation. Defaults to None.
            threshold (float, optional): The threshold value for the test. Defaults to 0.05.
            print_log (bool, optional): Whether to print the log. Defaults to True.

        Returns:
            tuple: A tuple containing the rejected dates for the T-test and Chi2 test.
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
        Runs the congruence test for all dates in the DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).
            threshold (float): The threshold value for the test.
            ttest_rejected_dates (list): A list to store the rejected dates for the T-test.
            chi2_rejected_dates (list): A list to store the rejected dates for the Chi2 test.
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
        Runs the congruence test for a specific date range.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).
            start_date (str): The start date for the calculation.
            end_date (str): The end date for the calculation.
            threshold (float): The threshold value for the test.
            ttest_rejected_dates (list): A list to store the rejected dates for the T-test.
            chi2_rejected_dates (list): A list to store the rejected dates for the Chi2 test.
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
        Calculates the list of differences for the congruence test.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).
            start_date (str): The start date for the calculation.
            end_date (str): The end date for the calculation.
            stations (list): A list of stations.

        Returns:
            list: A list of differences for the congruence test.
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
        Calculates the baselines for the given date and stations.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            date (str): The date for the calculation.
            stations (list): A list of stations.

        Returns:
            list: A list of baselines for the given date and stations.
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
        Calculates the coordinates for the given date and stations.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            date (str): The date for the calculation.
            stations (list): A list of stations.

        Returns:
            dict: A dictionary of coordinates for the given date and stations.
        """
        coords = {}  # Словарь для хранения координат станций
        for station in stations:
            station_df = df.loc[(df['Date'] == date) & (df['Station'] == station)]
            coords[station] = (station_df['X'].values[0], station_df['Y'].values[0], station_df['Z'].values[0])
        return coords

    def _perform_ttest(self, raz_list, threshold):
        """
        Performs a T-test on the given list of differences.

        Args:
            raz_list (list): The list of differences.
            threshold (float): The threshold value for the test.

        Returns:
            tuple: A tuple containing the result of the T-test and the p-value.
        """
        pvalue = ttest_1samp(a=raz_list, popmean=0, nan_policy='omit')[1]
        if pvalue <= threshold:
            return True, pvalue
        else:
            return False, pvalue

    def _perform_chi2_test(self, raz_list, sigma_0, threshold):
        """
        Performs a Chi2 test on the given list of differences.

        Args:
            raz_list (list): The list of differences.
            sigma_0 (float): The sigma-0 value for the test.
            threshold (float): The threshold value for the test.

        Returns:
            tuple: A tuple containing the result of the Chi2 test, K-value, and test value.
        """
        """
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
        """
        Finds the offset points for the given DataFrame and method.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).

        Returns:
            list: A list of offset points.
        """
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
        """
        Finds the offset points for the given DataFrame and method.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).

        Returns:
            list: A list of offset points.
        """
        offset_points = []
        ttest_rejected_dates, chi2_rejected_dates = self.congruency_test(df, method, calculation="all_dates")
        rejected_dates = ttest_rejected_dates + chi2_rejected_dates

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
        Prints the results of the congruence test.

        Args:
            ttest_rejected_dates (list): A list of rejected dates for the T-test.
            chi2_rejected_dates (list): A list of rejected dates for the Chi2 test.
        """
        print("----------------------------")
        print(f"Всего выполнено тестов: {len(self.dates)}.")
        print(f"Хи-квадрат, не отвергнуто: {len(self.dates) - len(chi2_rejected_dates)}, "
              f"отвергнуто: {len(chi2_rejected_dates)} ({len(chi2_rejected_dates) / len(self.dates) * 100:.2f}%)")
        print(f"Т-тест, не отвергнуто: {len(self.dates) - len(ttest_rejected_dates)}, "
              f"отвергнуто: {len(ttest_rejected_dates)} ({len(ttest_rejected_dates) / len(self.dates) * 100:.2f}%)")

        print(f"Chi2 rejected dates: {chi2_rejected_dates}")


def main():
    """
    The main function.
    """
    df = pd.read_csv('Data/YGGR_2022_06_19_10points_impulse0.05.csv', delimiter=';')

    test = Tests(df)
    #test.congruency_test(df=df, method='coordinate_based')
    offset_points = test.find_offset_points_2(df=df, method='coordinate_based')
    print("Candidate points with offsets:", offset_points)


if __name__ == "__main__":
    main()
