import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import ttest_1samp, shapiro, chi2
from scipy.signal import medfilt
from scipy.linalg import block_diag
from jinja2 import Template
from io import StringIO, BytesIO
import logging
from tkinter import filedialog
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# Установка логгера
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# обработчик StringIO для записи логов
string_io_handler = logging.StreamHandler(StringIO())
string_io_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
string_io_handler.setFormatter(formatter)
logger.addHandler(string_io_handler)


class Tests:
    """
    A class for performing congruence tests on time series data.

    Args:
        df (DataFrame): The input DataFrame containing time series data.
    """
    def __init__(self, df: DataFrame):
        """
        Initializes a Tests object.

        Args:
            df (DataFrame): DataFrame containing time series
        """
        self.df = df
        self.dates = df['Date'].unique()  # датафрейм с уникальными датами

    def congruency_test(self,
                        df: DataFrame,
                        Qv: DataFrame,
                        sigma_0: DataFrame,
                        Qdd_status: str = '0',
                        m_coef: float = 1,
                        calculation: str = "all_dates",
                        start_date: str = None,
                        end_date: str = None,
                        threshold: float = 0.95) -> tuple:
        """
        Performs a congruence test on the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            Qv (DataFrame): The DataFrame containing variance matrix of the residuals.
            sigma_0 (DataFrame): The DataFrame containing sigma values for the coordinates.
            Qdd_status (str, optional): Set Qdd-matrix status. 1 - identity matrix, 2 - covariance matrix.
            m_coef (float, optional): Scale-factor value for the Chi-test statistic. Defaults to 1.
            calculation (str, optional): Test mode (all_dates or specific_date). Defaults to "all_dates".
            start_date (str, optional): The start date for the calculation. Defaults to None.
            end_date (str, optional): The end date for the calculation. Defaults to None.
            threshold (float, optional): The threshold value for the test. Defaults to 0.95.

        Returns:
            tuple: A tuple containing the rejected dates for the T-test and Chi2 test.
        """
        ttest_rejected_dates = []
        chi2_rejected_dates = []

        if calculation == "all_dates":
            self._run_all_dates(df, threshold, sigma_0, ttest_rejected_dates, chi2_rejected_dates, Qv=Qv,
                                Qdd_status=Qdd_status, m_coef=m_coef)
        elif calculation == "specific_date":
            self._run_specific_date(df, start_date, end_date, threshold, sigma_0, ttest_rejected_dates,
                                    chi2_rejected_dates, Qv=Qv, Qdd_status=Qdd_status, m_coef=m_coef)

        return ttest_rejected_dates, chi2_rejected_dates

    def _run_all_dates(self,
                       df: DataFrame,
                       threshold: float,
                       sigma_0: DataFrame,
                       ttest_rejected_dates: list,
                       chi2_rejected_dates: list,
                       Qv: DataFrame,
                       Qdd_status: str,
                       m_coef: float) -> None:
        """
        Runs the congruence test for all dates in the DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            threshold (float): The threshold value for the test. Defaults to 0.95.
            sigma_0 (DataFrame): The DataFrame containing sigma values for the coordinates.
            ttest_rejected_dates (list): A list to store the rejected dates for the T-test.
            chi2_rejected_dates (list): A list to store the rejected dates for the Chi2 test.
            Qv (DataFrame): The DataFrame containing variance matrix of the residuals.
            Qdd_status (str, optional): Set Qdd-matrix status. 1 - identity matrix, 2 - covariance matrix.
            m_coef (float, optional): Scale-factor value for the Chi-test statistic. Defaults to 1.
        """
        # Получение датафрейма со списком уникальных дат
        dates = df['Date']
        # Сортировка дат
        dates = sorted(dates)
        for i in range(len(dates) - 1):
            start_date = dates[i]
            end_date = dates[i + 1]
            logger.info(f"Calculating for dates: {start_date} and {end_date}")
            # Вызов функции congruency_test для каждой пары дат
            self._run_specific_date(df, start_date, end_date, threshold,
                                    sigma_0, ttest_rejected_dates, chi2_rejected_dates, Qv=Qv,
                                    Qdd_status=Qdd_status, m_coef=m_coef)

    def _run_specific_date(self,
                           df: DataFrame,
                           start_date: str,
                           end_date: str,
                           threshold: float,
                           sigma_0: DataFrame,
                           ttest_rejected_dates: list,
                           chi2_rejected_dates: list,
                           Qv: DataFrame,
                           Qdd_status: str,
                           m_coef: float) -> None:
        """
        Runs the congruence test for a specific date range.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            start_date (str): The start date for the calculation.
            end_date (str): The end date for the calculation.
            threshold (float): The threshold value for the test. Defaults to 0.95.
            sigma_0 (DataFrame): The DataFrame containing sigma values for the coordinates.
            ttest_rejected_dates (list): A list to store the rejected dates for the T-test.
            chi2_rejected_dates (list): A list to store the rejected dates for the Chi2 test.
            Qv (DataFrame): The DataFrame containing variance matrix of the residuals.
            Qdd_status (str, optional): Set Qdd-matrix status. 1 - identity matrix, 2 - covariance matrix.
            m_coef (float, optional): Scale-factor value for the Chi-test statistic. Defaults to 1.
        """
        # Вычисляем вектор разностей между двумя датами
        raz_list = self._calculate_raz_list(df, start_date, end_date)

        # Находим матрицы Qv для двух дат
        Qv_0 = Qv[Qv['Date'] == start_date]
        Qv_i = Qv[Qv['Date'] == end_date]

        # Создание списков для хранения матриц
        matrices_0 = []
        matrices_i = []

        # цикл сборки матриц для первой эпохи
        for column in Qv_0.columns:
            if column != 'Date':
                matrices_0.extend(Qv_0[column].tolist())

        # цикл сборки матриц для второй эпохи
        for column in Qv_i.columns:
            if column != 'Date':
                matrices_i.extend(Qv_i[column].tolist())

        # Из собранных матриц формируем блочно-диагональные
        Qv_0_block = block_diag(*matrices_0)
        Qv_i_block = block_diag(*matrices_i)

        # Находим сумму матриц
        Qv_sum = Qv_0_block + Qv_i_block
        Qv_sum = np.array(Qv_sum.tolist())

        # Находим сумму значений sigma
        mu_0 = sigma_0[sigma_0['Date'] == start_date]
        mu_i = sigma_0[sigma_0['Date'] == end_date]
        mu_sum = (mu_0['Sum_of_Squares'].reset_index(drop=True) + mu_i['Sum_of_Squares'].reset_index(drop=True).sum())
        sigma_0 = float(mu_sum.iloc[0])

        # Записываем в логи значение статистики теста Шапиро-Уилка
        logger.info('Shapiro: %s', shapiro(raz_list))

        # Тест Хи-квадрат
        chi2_result, K, test_value = self._perform_chi2_test(raz_list, sigma_0, threshold, Qv=Qv_sum,
                                                             Qdd_status=Qdd_status, m_coef=m_coef)
        if chi2_result:
            chi2_rejected_dates.append((start_date, end_date))
            logger.info("Chi-2: " + f"<span style='color:red'>Null hypothesis rejected, "
                                    f"testvalue = {round(test_value, 3)}, K = {round(K, 3)}</span>")
        else:
            logger.info("Chi-2: " + f"Null hypothesis not rejected, testvalue = {round(test_value, 3)},"
                        f" K = {round(K, 3)}")
        logger.info("")

    def _calculate_raz_list(self,
                            df: DataFrame,
                            start_date: str,
                            end_date: str) -> list:
        """
        Calculates the list of differences for the congruence test.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            start_date (str): The start date for the calculation.
            end_date (str): The end date for the calculation.

        Returns:
            list: A list of differences for the congruence test.
        """
        # Конвертация столбцов с координатами в числовой формат
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)
        df = df.loc[:, df.columns.str.startswith(('x_', 'y_', 'z_')) | (df.columns == 'Date')]

        row_0 = df[df['Date'] == start_date]
        row_i = df[df['Date'] == end_date]

        # Проверка наличия данных в столбцах
        if row_0.empty or row_i.empty:
            return None

        # вычисление разностей столбцов
        result = row_0.iloc[0, 1:] - row_i.iloc[0, 1:]

        # возвращение результата в формате списка
        return result.tolist()

    def _perform_ttest(self,
                       raz_list: list,
                       threshold: float) -> tuple:
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

    def _perform_chi2_test(self,
                           raz_list: list,
                           sigma_0: float,
                           threshold: float,
                           Qv: np.ndarray,
                           Qdd_status: str,
                           m_coef: float) -> tuple:
        """
        Performs a Chi2 test on the given list of differences.

        Args:
            raz_list (list): The list of differences.
            sigma_0 (float): The sigma-0 value for the test.
            threshold (float): The threshold value for the test.
            Qv (np.ndarray): The variance matrix of the residuals.
            Qdd_status (str, optional): Set Qdd-matrix status. 1 - identity matrix, 2 - covariance matrix.
            m_coef (float, optional): Scale-factor value for the Chi-test statistic. Defaults to 1.

        Returns:
            tuple: A tuple containing the result of the Chi2 test, K-value, and test value.
        """
        # вектор разностей между эпохами
        d = np.array(raz_list)

        # Формирование матрицы Qdd согласно заданному статусу
        if Qdd_status == '1':
            Qdd = np.eye(Qv.shape[0])  # Единичная матрица Qdd
        elif Qdd_status == '0':
            Qdd = Qv  # Ковариационная матрица Qdd

        # Вычисление статистики
        K = (d.dot(np.linalg.inv(Qdd)).dot(d.transpose()) / sigma_0) * m_coef

        # Вычисление тестового значения статистики
        test_value = chi2.ppf(df=((d.shape[0]) / 3) * 6, q=threshold)

        # Проверка гипотезы
        if K > test_value:
            return True, K, test_value
        else:
            return False, K, test_value

    def find_offset_points(self,
                           df: DataFrame,
                           sigma_0: DataFrame,
                           m_coef: float,
                           Qdd_status: str,
                           Qv: DataFrame,
                           max_drop: int = 1) -> tuple:
        """
        Finds the offset points for the given DataFrame and method.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            sigma_0 (DataFrame): The DataFrame containing sigma values for the coordinates.
            m_coef (float, optional): Scale-factor value for the Chi-test statistic. Defaults to 1.
            Qdd_status (str, optional): Set Qdd-matrix status. 1 - identity matrix, 2 - covariance matrix.
            Qv (DataFrame): The variance matrix of the residuals.
            max_drop (int, optional): The maximum number of stations to drop. Defaults to 1.

        Returns:
            list: A list of offset points.
        """
        # Список для деформированных станций
        offset_points = []

        # сумма всех значений sigma возведенных в квадрат
        sum_of_squares = sigma_0.drop('Date', axis=1).apply(lambda row: sum(row ** 2), axis=1)

        # создаем датафрейм с ними
        mu_mean_df = sigma_0[['Date']].copy()
        mu_mean_df['Sum_of_Squares'] = sum_of_squares

        # находим даты с деформациями при помощи геометрического теста
        ttest_rejected_dates, chi2_rejected_dates = self.congruency_test(df=df,
                                                                         calculation="all_dates", sigma_0=mu_mean_df,
                                                                         Qv=Qv, Qdd_status=Qdd_status, m_coef=m_coef)
        rejected_dates = ttest_rejected_dates + chi2_rejected_dates

        logger.info('<h2>Finding the offset points:</h2>')

        # Получаем список с названиями станций и сбрасываем повторяющиеся значения
        station_names = list(set(df.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

        # Цикл перебора для нахождения деформированных станций
        for start_date, end_date in rejected_dates:
            # Создаем датафрейм, который содержит временной интервал с деформацией
            date_range_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            # Цикл для сброса разного кол-ва станций (начиная с одной, заканчивая max_drop)
            for drop_count in range(1, min(max_drop + 1, len(station_names) + 1)):
                # Находим все возможные комбинации станций при помощи itertools и создаем цикл
                for station_combination in itertools.combinations(station_names, drop_count):
                    logger.info(f'Calculating for stations {station_combination} and dates {start_date} to {end_date}')
                    # формируем датафреймы, исключая подозреваемую станцию/станции
                    non_station_df = self.drop_station_columns(date_range_df, station_combination)
                    non_station_qv = self.drop_station_columns(Qv, station_combination)
                    non_station_sigma = self.drop_station_columns(sigma_0, station_combination)

                    # заново находим сумму значений sigma
                    sum_of_squares = non_station_sigma.drop('Date', axis=1).apply(lambda row: sum(row ** 2), axis=1)
                    # создаем датафрейм с ними
                    mu_mean_df = non_station_sigma[['Date']].copy()
                    mu_mean_df['Sum_of_Squares'] = sum_of_squares

                    # Исключив подозреваемые станции, проводим геометрический тест
                    ttest_rejected, chi2_rejected = self.congruency_test(df=non_station_df,
                                                                         calculation="specific_date",
                                                                         start_date=start_date, end_date=end_date,
                                                                         sigma_0=mu_mean_df, Qv=non_station_qv,
                                                                         Qdd_status=Qdd_status, m_coef=m_coef)

                    # Если деформация исчезла, записываем данные об исключенных станциях и завершаем цикл
                    if not (ttest_rejected or chi2_rejected):
                        station_cols = [col for col in date_range_df.columns if
                                        any(station in col for station in station_combination)]
                        start_station_df = date_range_df[station_cols].loc[date_range_df['Date'] == start_date]
                        end_station_df = date_range_df[station_cols].loc[date_range_df['Date'] == end_date]
                        offset_size = np.sqrt(np.sum((start_station_df.values - end_station_df.values) ** 2))
                        for station in station_combination:
                            offset_points.append((start_date, end_date, station, offset_size))
                        break
                # Если искомые станция/станции найдены, прекращаем перебор
                if offset_points and offset_points[-1][0] == start_date and offset_points[-1][1] == end_date:
                    break

        return offset_points, rejected_dates

    def drop_station_columns(self,
                             df: DataFrame,
                             stations: list) -> DataFrame:
        """
        Drops columns corresponding to specific stations.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            stations (list): A list of station names to drop.

        Returns:
            DataFrame: The DataFrame with the specified station columns dropped.
        """
        # удаляем колонны, соответствующие указанной станции
        station_cols = [col for col in df.columns if any(station in col for station in stations)]
        return df.drop(station_cols, axis=1)

    def geometric_chi_test_calc(self,
                                time_series_frag: np.ndarray,
                                sigma: np.ndarray,
                                sigma_0, # to be removed
                                covariances,
                                Q_status) -> tuple:
        """
        A method that is used in geometric_chi_test_statictics to perform geometric chi test.

        Args:
            time_series_frag (np.ndarray): The time series fragment.
            sigma (np.ndarray): The sigma values.
            sigma_0 (float): The sigma_0 value. To be removed.

        Returns:
            tuple: A tuple containing the x_LS_first, x_LS, Qv, and mu values.
        """
        # Фрагмент временного ряда с одной станцией
        time_series_frag = pd.DataFrame(time_series_frag)
        time_series_frag.reset_index(drop=True, inplace=True)
        time_series_frag.columns = [0, 1, 2]

        # скп для этой станции
        sigma = pd.DataFrame(sigma)
        sigma.reset_index(drop=True, inplace=True)
        sigma.columns = [0, 1, 2]

        # ковариации для этой станции
        covariances = pd.DataFrame(covariances)
        covariances.reset_index(drop=True, inplace=True)
        covariances.columns = [0, 1, 2]

        # маска для NaN значений
        # mask = ~np.isnan(time_series_frag)

        L = np.zeros((time_series_frag.shape[0] * 3))

        # задаем массив эпох
        t = np.arange(0, time_series_frag.shape[0], 1)

        # Заполняем массив L
        for i in range(time_series_frag.shape[0]):
            for j in range(3):
                L[3 * i] = time_series_frag[0][i]
                L[3 * i + 1] = time_series_frag[1][i]
                L[3 * i + 2] = time_series_frag[2][i]

        # цикл формирования матрицы коэффициентов
        for m in range(time_series_frag.shape[0]):
            ti = t[m]
            if m == 0:
                A = np.hstack((np.identity(3) * ti, np.identity(3)))
            else:
                Aux = np.hstack((np.identity(3) * ti, np.identity(3)))
                A = np.vstack((A, Aux))

        # создание матрицы Q
        Q_size = time_series_frag.shape[0] * 3
        Q = np.zeros((Q_size, Q_size))

        # заполнение матрицы Q
        if Q_status == '0':
            for i in range(time_series_frag.shape[0]):
                row_start = i * 3
                sde = sigma[0][i]
                sdn = sigma[1][i]
                sdu = sigma[2][i]
                sden = covariances[0][i]
                sdnu = covariances[1][i]
                sdue = covariances[2][i]

                Q[row_start:row_start + 3, row_start:row_start + 3] = np.array([
                    [sde**2, sden**2, sdue**2],
                    [sden**2, sdn**2, sdnu**2],
                    [sdue**2, sdnu**2, sdu**2]])
        # если матрица единичная
        elif Q_status == '1':
            for i in range(time_series_frag.shape[0]):
                row_start = i * 3
                Q[row_start:row_start + 3, row_start:row_start + 3] = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

        P = Q/(0.02**2)

        N = A.transpose().dot(np.linalg.inv(P)).dot(A)

        X = np.linalg.inv(N).dot(A.transpose().dot(np.linalg.inv(P)).dot(L))  # вектор параметров кинематической модели

        x_LS = np.array([X[0] * t[-1] + X[3], X[1] * t[-1] + X[4], X[2] * t[-1] + X[5]])

        x_LS_first = X[0] + X[1] * t[0]

        # вычисляем вектор невязок
        V = A.dot(X) - L

        # СКП единицы веса
        mu = np.sqrt(np.sum(V.transpose().dot(np.linalg.inv(P)).dot(V)) / (V.shape[0] - 6))

        # Qx = np.linalg.inv(N) * mu ** 2
        Qx = np.linalg.inv(N)
        C = np.array([[t[-1], 0, 0, 1, 0, 0], [0, t[-1], 0, 0, 1, 0], [0, 0, t[-1], 0, 0, 1]])
        Qv = C.dot(Qx).dot(C.transpose())

        return x_LS_first, x_LS, Qv, mu, Qx


    def chi_test_calc_without_wls(self,
                                time_series_frag: np.ndarray,
                                sigma: np.ndarray,
                                sigma_0,
                                covariances,
                                mu,
                                Q_status) -> tuple:
        """
        A method that is used in geometric_chi_test_statictics to perform geometric chi test.

        Args:
            time_series_frag (np.ndarray): The time series fragment.
            sigma (np.ndarray): The sigma values.
            sigma_0 (float): The sigma_0 value.

        Returns:
            tuple: A tuple containing the x_LS_first, x_LS, Qv, and mu values.
        """
        # Фрагмент временного ряда с одной станцией
        time_series_frag = pd.DataFrame(time_series_frag)
        time_series_frag.reset_index(drop=True, inplace=True)
        time_series_frag.columns = [0, 1, 2]

        # скп для этой станции
        sigma = pd.DataFrame(sigma)
        sigma.reset_index(drop=True, inplace=True)
        sigma.columns = [0, 1, 2]

        # ковариации для этой станции
        covariances = pd.DataFrame(covariances)
        covariances.reset_index(drop=True, inplace=True)
        covariances.columns = [0, 1, 2]

        # скп единицы веса для этой станции
        mu_values = pd.DataFrame(mu)
        mu_values.reset_index(drop=True, inplace=True)
        mu_values.columns = [0]

        # заполнение матрицы Q
        if Q_status == '0':
            sde = sigma[0][0]
            sdn = sigma[1][0]
            sdu = sigma[2][0]
            sden = covariances[0][0]
            sdnu = covariances[1][0]
            sdue = covariances[2][0]

            Q = np.array([
                [sde**2, sden**2, sdue**2],
                [sden**2, sdn**2, sdnu**2],
                [sdue**2, sdnu**2, sdu**2]])
        # если матрица единичная
        elif Q_status == '1':
            Q = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

        P = Q/((0.005*3)**2)

        return P


    def geometric_chi_test_statictics(self, station,
                                      time_series_df: DataFrame,
                                      window_size: str,
                                      sigma_0,
                                      Q_status) -> tuple:
        """
        Performs a geometric Chi test on the given time series DataFrame.

        Args:
            station():
            time_series_df (DataFrame): The time series DataFrame.
            window_size (str): The size of time intervals for wls interpolation.
            sigma_0 (float): The sigma_0 value.
            Q_status (str)

        Returns:
            tuple: A tuple containing the X_WLS, Qv_WLS, test_statistic, wls_df, and Qv_df values.
        """
        X_WLS = []
        Qv_WLS = []
        wls_times = []
        initial_values_X = []
        initial_values_Qv = []
        initial_values_mu = []
        mu_list = []

        time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
        time_series_df.set_index('Date', inplace=True)

        # Вычисление начального значения
        start_time = time_series_df.index[0]
        end_time = start_time + pd.Timedelta(window_size)
        fragment = time_series_df[(time_series_df.index >= start_time) & (time_series_df.index < end_time)]

        coord_cols = [col for col in fragment.columns if
                      col.startswith('x_') or col.startswith('y_') or col.startswith('z_')]
        sigma_cols = [f'sde_{station}' if col.startswith('x_') else f'sdn_{station}' if col.startswith(
            'y_') else f'sdu_{station}' for col in coord_cols]
        covariance_cols = [f'sden_{station}' if col.startswith('x_') else f'sdnu_{station}' if col.startswith(
            'y_') else f'sdue_{station}' for col in coord_cols]

        x_LS_first, _, Qv, mu_first, Qx = self.geometric_chi_test_calc(time_series_frag=fragment[coord_cols],
                                                                   sigma=fragment[sigma_cols],
                                                                   covariances=fragment[covariance_cols],
                                                                   sigma_0=sigma_0, Q_status=Q_status)
        initial_values_X.append(x_LS_first)
        initial_values_Qv.append(Qx)
        initial_values_mu.append(mu_first)

        '''X_WLS.append(x_LS_first)
        Qv_WLS.append(Qx)
        mu_list.append(mu_first)
        wls_times.append(start_time)'''

        i = 0

        while end_time <= time_series_df.index[-1]:
            wls_times.append(end_time)
            # извлечение фрагмента
            fragment = time_series_df[(time_series_df.index >= start_time) & (time_series_df.index < end_time)]

            # Apply the least squares code to the fragment
            x_LS_first, x_LS, Qv, mu, Qx = self.geometric_chi_test_calc(time_series_frag=fragment[coord_cols],
                                                                   sigma=fragment[sigma_cols],
                                                                   covariances=fragment[covariance_cols],
                                                                   sigma_0=sigma_0, Q_status=Q_status)

            X_WLS.append(x_LS)
            Qv_WLS.append(Qv)
            mu_list.append(mu)

            # Двигаемся к следующему фрагменту
            start_time = end_time
            end_time = start_time + pd.Timedelta(window_size)
            i += 1

        # конвертация списков в формат numpy
        X_WLS, mu_list = np.array(X_WLS), np.array(mu_list)

        time_series_df.reset_index(inplace=True)

        wls_df = pd.DataFrame(columns=['Date', f'x_{station}', f'y_{station}', f'z_{station}'])
        Qv_df = pd.DataFrame({'Date': pd.to_datetime(wls_times), f'Qv_{station}': Qv_WLS})
        mu_df = pd.DataFrame({'Date': pd.to_datetime(wls_times), f'MU_{station}': mu_list})

        # добавляем колонку с датами в датафреймы
        wls_df['Date'] = pd.to_datetime(wls_times)

        wls_df.iloc[:, 1:] = X_WLS
        #Qv_df.iloc[:, 1:] = Qv_WLS

        return X_WLS, Qv_WLS, wls_df, Qv_df, mu_df

    def geometric_without_wls(self, station,
                                      time_series_df: DataFrame,
                                      window_size: str,
                                      sigma_0,
                                      Q_status) -> tuple:

        X_WLS = []
        Qv_WLS = []
        wls_times = []
        initial_values_X = []
        initial_values_Qv = []
        initial_values_mu = []
        mu_list = []

        time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])

        # Получение датафрейма со списком уникальных дат
        dates = time_series_df['Date']
        # Сортировка дат
        dates = sorted(dates)

        time_series_df.set_index('Date', inplace=True)

        # Вычисление начального значения
        start_date = dates[0]
        end_date = dates[1]
        fragment = time_series_df.loc[time_series_df.index.isin([start_date, end_date])]

        coord_cols = [col for col in fragment.columns if
                      col.startswith('x_') or col.startswith('y_') or col.startswith('z_')]
        sigma_cols = [f'sde_{station}' if col.startswith('x_') else f'sdn_{station}' if col.startswith(
            'y_') else f'sdu_{station}' for col in coord_cols]
        covariance_cols = [f'sden_{station}' if col.startswith('x_') else f'sdnu_{station}' if col.startswith(
            'y_') else f'sdue_{station}' for col in coord_cols]
        mu_col = f'sigma0_{station}'

        '''x_LS_first, _, Qv, mu_first, Qx = self.geometric_chi_test_calc(time_series_frag=fragment[coord_cols],
                                                                       sigma=fragment[sigma_cols],
                                                                       covariances=fragment[covariance_cols],
                                                                       sigma_0=sigma_0, Q_status=Q_status)
        initial_values_X.append(x_LS_first)
        initial_values_Qv.append(Qx)
        initial_values_mu.append(mu_first)'''

        for i in range(len(dates)): # -1
            start_date = dates[i]
            # end_date = dates[i + 1]

            wls_times.append(start_date) # end_date
            # извлечение фрагмента
            fragment = time_series_df.loc[time_series_df.index.isin([start_date])]

            # Apply the least squares code to the fragment
            Q = self.chi_test_calc_without_wls(time_series_frag=fragment[coord_cols],
                                                                        sigma=fragment[sigma_cols],
                                                                        covariances=fragment[covariance_cols],
                                                                        mu=fragment[mu_col],
                                                                        sigma_0=sigma_0, Q_status=Q_status)

            #X_WLS.append(x_LS)
            Qv_WLS.append(Q)
            #mu_list.append(mu)

        # конвертация списков в формат numpy
        X_WLS, mu_list = np.array(X_WLS), np.array(mu_list)

        time_series_df.reset_index(inplace=True)

        wls_df = time_series_df[['Date', f'x_{station}', f'y_{station}', f'z_{station}']]
        Qv_df = pd.DataFrame({'Date': pd.to_datetime(wls_times), f'Qv_{station}': Qv_WLS})
        mu_df = time_series_df[['Date', f'sigma0_{station}']].rename(columns={f'sigma0_{station}': f'MU_{station}'})

        return X_WLS, Qv_WLS, wls_df, Qv_df, mu_df


    def interpolate_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Interpolates missing values in the DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.

        Returns:
            DataFrame: The DataFrame with interpolated missing values.
        """
        df_interpolated = df.interpolate(method='linear', limit_direction='both')
        return df_interpolated

    def filter_data(self, df: DataFrame, kernel_size: int) -> DataFrame:
        """
        Filters the data using a median filter.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            kernel_size (int): The size of the median filter kernel.

        Returns:
            DataFrame: The filtered DataFrame.
        """
        df_filtered = df.copy()
        for col in df_filtered.columns:
            if col.startswith(('x_', 'y_', 'z_')):
                df_filtered[col] = medfilt(df_filtered[col], kernel_size=kernel_size)
        return df_filtered

    def perform_wls(self, df: DataFrame, window_size: str, sigma_0: float, Q_status: str, wls_status: str) -> tuple:
        """
        Performs a weighted least squares (WLS) analysis on the given DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            window_size (str): The size of time intervals for WLS interpolation.
            sigma_0 (float): The sigma_0 value used in the WLS calculations.
            Q_status (str): Set Q-matrix status. 1 - identity matrix, 2 - covariance matrix.
            wls_status (str): A flag indicating whether to perform WLS ('1') or not ('0').

        Returns:
            tuple: A tuple containing the following DataFrames:
                - wls (DataFrame): The DataFrame containing the results of the WLS analysis.
                - raw (DataFrame): The DataFrame containing the raw data for all stations.
                - filtered (DataFrame): The DataFrame containing the filtered data for all stations.
                - Qv (DataFrame): The DataFrame containing Qv values for all stations.
                - mu_mean_df (DataFrame): A DataFrame containing the mean values of MU for each date.
                - MU (DataFrame): The DataFrame containing MU values for all stations.
        """
        station_names = list(set(df.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

        raw_dfs = []
        filtered_dfs = []
        wls_dfs = []
        Qv_dfs = []
        mu_dfs = []

        for station in station_names:
            # Получаем колонны в соответствии с текущей станцией
            station_cols = ['Date'] + [col for col in df.columns if station in col]

            # извлекаем колонны
            station_df_raw = df[station_cols]

            # интерполируем пропуски в данных
            station_df = self.interpolate_missing_values(station_df_raw)

            # медианный фильтр
            station_df_filtered = self.filter_data(station_df, kernel_size=11)

            if wls_status == '1':
                # Perform the geometric chi test
                X_WLS, Qv_WLS, wls_df, Qv_df, mu_station_df = self.geometric_chi_test_statictics(
                    station=station,
                    time_series_df=station_df_filtered,
                    window_size=window_size,
                    sigma_0=sigma_0, Q_status=Q_status)
            if wls_status == '0':
                # Perform the geometric chi test without least squares interpolation
                X_WLS, Qv_WLS, wls_df, Qv_df, mu_station_df = self.geometric_without_wls(
                    station=station,
                    time_series_df=station_df_filtered,
                    window_size=window_size,
                    sigma_0=sigma_0, Q_status=Q_status)

            # Append the wls_df to the list
            raw_dfs.append(station_df_raw)
            filtered_dfs.append(station_df_filtered)
            wls_dfs.append(wls_df)
            Qv_dfs.append(Qv_df)
            mu_dfs.append((mu_station_df))

        # Объединение данных всех станций в единые списки
        raw = pd.concat(raw_dfs, axis=1)
        filtered = pd.concat(filtered_dfs, axis=1)
        wls = pd.concat(wls_dfs, axis=1)
        Qv = pd.concat(Qv_dfs, axis=1)
        MU = pd.concat(mu_dfs, axis=1)

        # вычисляем средние значения MU каждой строки
        mean_values = MU.drop('Date', axis=1).mean(axis=1)
        # Создаем новый датафрейм с этими значениями и соответствующими датами
        mu_mean_df = MU[['Date']].copy()
        mu_mean_df['mu_mean'] = mean_values

        # Убираем дупликаты
        raw = raw.loc[:, ~raw.columns.duplicated()]
        filtered = filtered.loc[:, ~filtered.columns.duplicated()]
        wls = wls.loc[:, ~wls.columns.duplicated()]
        Qv = Qv.loc[:, ~Qv.columns.duplicated()]
        mu_mean_df = mu_mean_df.loc[:, ~mu_mean_df.columns.duplicated()]
        MU = MU.loc[:, ~MU.columns.duplicated()]

        return wls, raw, filtered, Qv, mu_mean_df, MU

    def create_html_report(self, offset_points, rejected_dates, wls, raw, filtered, window_size,
                           file_path, file_name, Q_status, Qdd_status, m_coef):
        """
        Generates an HTML report containing offset analysis and visualizations.

        Args:
            offset_points (list): A list of tuples containing offset data, where each tuple consists of
                                  (start_date, end_date, station, offset_size).
            rejected_dates (list): A list of tuples containing rejected date ranges, where each tuple consists of
                                   (start_date, end_date).
            wls (DataFrame): A DataFrame containing weighted least squares data.
            raw (DataFrame): A DataFrame containing raw data.
            filtered (DataFrame): A DataFrame containing filtered data.
            window_size (str): The size of time intervals for wls interpolation.
            file_path (str): The file path for saving the report.
            file_name (str): The name of the report file.
            Q_status (str): Set Q-matrix status. 1 - identity matrix, 2 - covariance matrix.
            Qdd_status (str): Set Qdd-matrix status. 1 - identity matrix, 2 - covariance matrix.
            m_coef (float): Scale-factor value for the Chi-test statistic. Defaults to 1.

        Returns:
            None: The function saves the report to the specified output path and does not return a value.
        """
        # сохранение результатов в таблицу для HTML отчета
        offsets_table = pd.DataFrame(offset_points, columns=['Start_date', 'End_date', 'Station', 'Offset size'])
        offsets_html_table = offsets_table.to_html(index=False)

        # Извлечение названий станций из названий колонок
        stations = list(set(wls.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

        # создание папки с результатами
        result_directory = f'Data/test-Q({Q_status})-Qdd({Qdd_status})-m_coef({m_coef})-{file_name}'
        os.makedirs(result_directory, exist_ok=True)

        # сохранение смещений в виде csv таблицы
        offsets_table.to_csv(f'{result_directory}/Offsets-table-{file_name}.csv', sep=';', index=False)

        raw.to_csv(f'{result_directory}/Raw-data-{file_name}.csv', sep=';', index=False)

        # получение логов
        string_io_handler.flush()
        log_contents = string_io_handler.stream.getvalue()

        ## группировка смещений по станциям
        station_offsets = {station: [] for station in stations}  # Initialize with all stations
        for start_date, end_date, station, offset_size in offset_points:
            if station in station_offsets:
                station_offsets[station].append((start_date, end_date))


        # словарь для создания HTML отчета
        report_data = {
            'file_name': file_path,
            'total_tests': (len(wls['Date']) - 1),
            'stations_length': len(stations),
            'stations_names': stations,
            'window_size': window_size,
            'offset_points': offsets_html_table,
            'offset_plots': '',
            'triangulation_map': '',
            'log_contents': log_contents}

        # Создаем графики в HTML отчете для каждой станции

        # for station, offsets in station_offsets.items():
        for station in stations:
            offsets = station_offsets[station]  # Get offsets for the current station

            station_df_wls = wls[
                [col for col in wls.columns if
                 station in col and not col.startswith(f"sd{station}"[0:2]) or col == 'Date']]
            station_df_raw = raw[
                [col for col in raw.columns if
                 station in col and not col.startswith(f"sd{station}"[0:2]) or col == 'Date']]
            station_df_filtered = filtered[
                [col for col in filtered.columns if
                 station in col and not col.startswith(f"sd{station}"[0:2]) or col == 'Date']]

            x_values_raw = station_df_raw[f'x_{station}']
            y_values_raw = station_df_raw[f'y_{station}']
            z_values_raw = station_df_raw[f'z_{station}']

            x_values_fil = station_df_filtered[f'x_{station}']
            y_values_fil = station_df_filtered[f'y_{station}']
            z_values_fil = station_df_filtered[f'z_{station}']

            x_values = station_df_wls[f'x_{station}']
            y_values = station_df_wls[f'y_{station}']
            z_values = station_df_wls[f'z_{station}']

            fig = make_subplots(rows=3, cols=1, vertical_spacing=0.02)

            # график сырых координат
            fig.add_trace(go.Scatter(x=station_df_raw['Date'], y=x_values_raw, mode='lines', name='Raw data',
                                     line=dict(color='lightgray'), legendgroup='Raw data', showlegend=True), row=1,
                          col=1)
            fig.add_trace(go.Scatter(x=station_df_raw['Date'], y=y_values_raw, mode='lines', name='Raw data',
                                     line=dict(color='lightgray'), legendgroup='Raw data', showlegend=False), row=2,
                          col=1)
            fig.add_trace(go.Scatter(x=station_df_raw['Date'], y=z_values_raw, mode='lines', name='Raw data',
                                     line=dict(color='lightgray'), legendgroup='Raw data', showlegend=False), row=3,
                          col=1)

            # график координат с фильтром
            fig.add_trace(go.Scatter(x=station_df_filtered['Date'], y=x_values_fil, mode='lines', name='Filtered data',
                                     line=dict(color='blue'), yaxis='y2', legendgroup='Filtered data', showlegend=True),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=station_df_filtered['Date'], y=y_values_fil, mode='lines', name='Filtered data',
                                     line=dict(color='blue'), yaxis='y2', legendgroup='Filtered data',
                                     showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=station_df_filtered['Date'], y=z_values_fil, mode='lines', name='Filtered data',
                                     line=dict(color='blue'), yaxis='y2', legendgroup='Filtered data',
                                     showlegend=False),
                          row=3, col=1)

            # график координат с МНК
            fig.add_trace(go.Scatter(x=station_df_wls['Date'], y=x_values, mode='lines', name='WLS Estimate',
                                     line=dict(color='red'), yaxis='y3', legendgroup='WLS Estimate', showlegend=True),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=station_df_wls['Date'], y=y_values, mode='lines', name='WLS Estimate',
                                     line=dict(color='red'), yaxis='y3', legendgroup='WLS Estimate', showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=station_df_wls['Date'], y=z_values, mode='lines', name='WLS Estimate',
                                     line=dict(color='red'), yaxis='y3', legendgroup='WLS Estimate', showlegend=False),
                          row=3, col=1)

            # подсвечиваем даты-кандидаты на графике
            for start_date, end_date in rejected_dates:
                fig.add_vrect(x0=start_date, x1=end_date,
                              fillcolor='red', opacity=0.2,
                              layer='below', line_width=0)

            # подсвечиваем смещения на графике
            for start_date, end_date in offsets:
                fig.add_vrect(x0=start_date, x1=end_date,
                              fillcolor='green', opacity=0.5,
                              layer='above', line_width=0)

            for i in range(3):
                fig.update_xaxes(showgrid=False, row=i + 1, col=1)
                fig.update_yaxes(showgrid=False, row=i + 1, col=1)
                fig.update_yaxes(showgrid=False, row=i + 1, col=1, secondary_y=True)
                fig.update_yaxes(showgrid=False, row=i + 1, col=1, secondary_y=True, tertiary=True)
                if i < 2:
                    fig.update_xaxes(tickvals=[], row=i + 1, col=1)
                else:
                    fig.update_xaxes(tickformat='%H:%M:%S', row=i + 1, col=1)

            fig.update_layout(height=600, width=1200,  # корректируем размеры графика
                              title_text=f'Offsets found in {station} station: ',
                              margin=dict(l=10, r=10, t=50, b=10))

            # Конвертация графиков в HTML
            html_img = pio.to_html(fig, include_plotlyjs=True, full_html=False)

            # добавляем графики в HTML отчет
            report_data['offset_plots'] += html_img + "<br>"

        # Сохранение отчета в файл
        self.save_html_report(report_data=report_data,
                              output_path=f'{result_directory}/CongruencyTest-report-{file_name}' + '.html')

        print('File saved in ' + f'{result_directory}/CongruencyTest-report-{file_name}' + '.html')
        # удаляем обработчик StringIO
        logger.removeHandler(string_io_handler)

    def save_html_report(self, report_data: dict, output_path: str) -> str:
        """
        Saves an HTML report to the specified output path.

        Args:
            report_data (dict): A dictionary containing the report data.
            output_path (str): The output path for the report.

        Returns:
            str: The output path for the report.
        """

        html_template = """
        <html>
        <head>
        <title>Congruency Test Report</title>
        <style>
            img {
                width: 50%;
            }
        </style>
        </head>
        <body>
            <h1>Congruency Test Report</h1>
            <p><strong>Input file:</strong> {{ file_name }}</p>
            <p><strong>Total Tests:</strong> {{ total_tests }}</p>
            <p><strong>Total stations processed:</strong> {{ stations_length }}</p>
            <p><strong>Stations names:</strong> {{ stations_names }}</p>
            <p><strong>WLS window size:</strong> {{ window_size }}</p>
            <p><strong>Offset Points:</strong></p>
            {{ offset_points }}
            <h2>Points with offsets:</h2>
            {{ offset_plots }}
            <h2>Detailed Log:</h2>
            <pre>{{ log_contents }}</pre>
        </body>
        </html>
        """
        template = Template(html_template)
        html_content = template.render(**report_data)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        return output_path

    def extract_offset_windows(self, df, rejected_dates):
        """
        This function is used in window_iteration() and going to be removed.
        """
        offset_windows = []

        for start_date, end_date in rejected_dates:
            window = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            offset_windows.append(window)

        return offset_windows

    def window_iteration(self, df):
        """
        This function is going to be removed
        """
        window_sizes = ['1h', '10min', '1min']
        df['Date'] = pd.to_datetime(df['Date'])
        offset_windows = [df]

        # Iterate over the offset windows with decreasing window sizes
        rejected_dates_list = []
        for window_size in window_sizes:
            new_offset_windows = []
            for offset_window in offset_windows:
                print(f"Processing window of size {window_size}")
                # Run the wls inside each offsetted dataframe with the current window size
                wls, raw, filtered, Qv, mu_mean_df, MU = self.perform_wls(offset_window, window_size, 0.005)

                # Get the rejected dates and extract intervals
                ttest_rejected_dates, chi2_rejected_dates = self.congruency_test(df=wls, calculation="all_dates",
                                                                                 sigma_0=mu_mean_df,
                                                                                 Qv=Qv, Qdd_status=Qdd_status)
                rejected_dates = ttest_rejected_dates + chi2_rejected_dates
                if rejected_dates:
                    print(f"Rejected dates: {rejected_dates}")
                    new_offset_windows.extend(self.extract_offset_windows(df, rejected_dates))

                    # Store the rejected dates for the current window size
                    rejected_dates_list.append(rejected_dates)

            offset_windows = new_offset_windows

        # Output the final 1-second intervals and rejected dates
        return offset_windows, rejected_dates_list[-1]


def select_file() -> tuple:
    """
    Selects an input file using a file dialog.

    Returns:
        str: The selected file path.
    """
    file_path = filedialog.askopenfilename(title="Select input file",
                                           filetypes=[("CSV files", "*.csv")])
    file_name = os.path.basename(file_path)
    return file_path, file_name


def main() -> None:
    """
    The main function.
    """

    # чтение входного файла
    file_path, file_name = select_file()
    df = pd.read_csv(file_path, delimiter=';')

    # инициализация объекта Tests
    test = Tests(df)

    # WLS статус
    wls_status = str(input('Нужна ли линейная регрессия \nда - 1\nнет - 0\n '))

    # размер окна
    window_size = str(input('Размер окна: '))

    # обработка координат при помощи МНК
    Q_status = str(input('Матрица Q (в лин регрессии)\nединичная - 1\nс ковариациями - 0\n'))
    Qdd_status = str(input('Матрица Qdd (в хи-тесте)\nединичная - 1\nс ковариациями - 0\n'))
    m_coef = float(input('Масштабный коэффициент: '))

    # Выполнение линейной регрессии
    wls, raw, filtered, Qv, mu_mean_df, MU = test.perform_wls(df=df, window_size=window_size, sigma_0=0.05,
                                                              Q_status=Q_status, wls_status=wls_status)

    # перевод дат в корректный формат
    wls['Date'] = wls['Date'].dt.to_pydatetime()

    # Геометрический тест + поиск станций со смещением
    offset_points, rejected_dates = test.find_offset_points(df=wls, sigma_0=MU, Qv=Qv, max_drop=2, Qdd_status=Qdd_status, m_coef=m_coef)

    # Создание отчета
    test.create_html_report(offset_points=offset_points, rejected_dates=rejected_dates, wls=wls, raw=raw,
                                      filtered=filtered, window_size=window_size, file_path=file_path,
                                      file_name=file_name, Q_status=Q_status, Qdd_status=Qdd_status, m_coef=m_coef)


if __name__ == "__main__":
    main()

