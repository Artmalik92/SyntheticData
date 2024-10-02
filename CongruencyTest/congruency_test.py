import numpy as np
from numpy.linalg import pinv
import pandas as pd
from pandas import DataFrame
from scipy.stats import ttest_1samp, shapiro, chi2
from scipy.optimize import minimize
from scipy.signal import medfilt
from jinja2 import Template
from io import StringIO, BytesIO
import logging
import matplotlib.pyplot as plt
from tkinter import filedialog
import base64
import contextily as ctx
import pyproj
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
    def __init__(self, df: DataFrame):
        """
        Initializes a SyntheticData object.

        Args:
            df (DataFrame): DataFrame containing time series
        """
        self.df = df
        self.dates = df['Date'].unique()  # датафрейм с уникальными датами

    def congruency_test(self, df, Qv,
                        calculation: str = "all_dates",
                        start_date: str = None,
                        end_date: str = None,
                        threshold: float = 0.05,
                        sigma_0: float = 0.005):
        """
        Performs a congruence test on the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            calculation (str, optional): The type of calculation to perform (all_dates or specific_date). Defaults to "all_dates".
            start_date (str, optional): The start date for the calculation. Defaults to None.
            end_date (str, optional): The end date for the calculation. Defaults to None.
            threshold (float, optional): The threshold value for the test. Defaults to 0.05.
            sigma_0 (float, optional): The sigma_0 value for the Chi2 test. Defaults to 0.005.

        Returns:
            tuple: A tuple containing the rejected dates for the T-test and Chi2 test.
        """

        ttest_rejected_dates = []
        chi2_rejected_dates = []

        if calculation == "all_dates":
            self._run_all_dates(df, threshold, sigma_0, ttest_rejected_dates, chi2_rejected_dates, Qv=Qv)
        elif calculation == "specific_date":
            self._run_specific_date(df, start_date, end_date, threshold, sigma_0, ttest_rejected_dates,
                                    chi2_rejected_dates, Qv=Qv)

        return ttest_rejected_dates, chi2_rejected_dates

    def _run_all_dates(self, df, threshold, sigma_0, ttest_rejected_dates, chi2_rejected_dates, Qv):
        """
        Runs the congruence test for all dates in the DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            threshold (float): The threshold value for the test.
            ttest_rejected_dates (list): A list to store the rejected dates for the T-test.
            chi2_rejected_dates (list): A list to store the rejected dates for the Chi2 test.
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
                                    sigma_0, ttest_rejected_dates, chi2_rejected_dates, Qv=Qv)

    def _run_specific_date(self, df, start_date, end_date, threshold, sigma_0,
                           ttest_rejected_dates, chi2_rejected_dates, Qv):
        """
        Runs the congruence test for a specific date range.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            start_date (str): The start date for the calculation.
            end_date (str): The end date for the calculation.
            threshold (float): The threshold value for the test.
            ttest_rejected_dates (list): A list to store the rejected dates for the T-test.
            chi2_rejected_dates (list): A list to store the rejected dates for the Chi2 test.
        """

        raz_list = self._calculate_raz_list(df, start_date, end_date)

        Qv.iloc[:, 1:] = Qv.iloc[:, 1:].apply(pd.to_numeric)

        Qv_0 = Qv[Qv['Date'] == start_date]
        Qv_i = Qv[Qv['Date'] == end_date]

        Qv_sum = Qv_0.iloc[0, 1:] + Qv_i.iloc[0, 1:]
        Qv_sum = np.array(Qv_sum.tolist())

        logger.info('Shapiro: %s', shapiro(raz_list))

        chi2_result, K, test_value = self._perform_chi2_test(raz_list, sigma_0, threshold, Qv=Qv_sum)
        if chi2_result:
            chi2_rejected_dates.append((start_date, end_date))
            logger.info("Chi-2: " + f"<span style='color:red'>Null hypothesis rejected, "
                                    f"testvalue = {round(test_value, 3)}, K = {round(K, 3)}</span>")
        else:
            logger.info("Chi-2: " + f"Null hypothesis not rejected, testvalue = {round(test_value, 3)},"
                        f" K = {round(K, 3)}")
        logger.info("")

    def _calculate_raz_list(self, df, start_date, end_date):
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

        row_0 = df[df['Date'] == start_date]
        row_i = df[df['Date'] == end_date]

        # Check if both rows exist
        if row_0.empty or row_i.empty:
            return None

        # Subtract the values of the corresponding columns
        result = row_0.iloc[0, 1:] - row_i.iloc[0, 1:]

        # Return the result as a list
        return result.tolist()

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
            if not station_df.empty:
                coords[station] = (station_df['X'].values[0], station_df['Y'].values[0], station_df['Z'].values[0])
            else:
                #raise ValueError("The list of coordinates is empty")
                pass
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

    def _perform_chi2_test(self, raz_list, sigma_0, threshold, Qv):
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

        #Qdd = np.diag(Qv)
        Qdd = np.eye(d.shape[0])
        K = d.transpose().dot(pinv(Qdd)).dot(d) / (sigma_0 ** 2)
        test_value = chi2.ppf(df=d.shape[0], q=threshold)

        if K > test_value:
            return True, K, test_value
        else:
            return False, K, test_value

    def find_offset_points(self, df, method, sigma_0, Qv, max_drop=1):
        """
        Finds the offset points for the given DataFrame and method.

        Args:
            df (DataFrame): The input DataFrame containing time series data.
            method (str): The method to use for the congruence test (line_based or coordinate_based).
            sigma_0 (float, optional): The sigma_0 value for the Chi2 test. Defaults to 0.005.

        Returns:
            list: A list of offset points.
        """
        offset_points = []
        ttest_rejected_dates, chi2_rejected_dates = self.congruency_test(df=df, method=method,
                                                                         calculation="all_dates", sigma_0=sigma_0, Qv=Qv)
        rejected_dates = ttest_rejected_dates + chi2_rejected_dates

        logger.info('<h2>Finding the offset points:</h2>')

        # Получаем список с названиями станций и дропаем повторяющиеся значения
        station_names = list(set(df.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

        """for start_date, end_date in rejected_dates:
            temp_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            for station in station_names:
                logger.info(f'calculating for station {station}')
                logger.info(f'dates  {start_date} and {end_date}')
                # Get the columns corresponding to the current station
                station_cols = [col for col in temp_df.columns if station in col]
                # Create a temporary DataFrame with all columns except the current station
                temp_temp_df = temp_df.drop(station_cols, axis=1)
                ttest_rejected, chi2_rejected = self.congruency_test(df=temp_temp_df, method=method, calculation="specific_date",
                                                                     start_date=start_date, end_date=end_date,
                                                                     sigma_0=sigma_0,
                                                                     print_log=False, Qv=Qv)
                if not (ttest_rejected or chi2_rejected):
                    offset_points.append((start_date, end_date, station))"""
        for start_date, end_date in rejected_dates:
            date_range_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            for drop_count in range(1, min(max_drop + 1, len(station_names) + 1)):
                for station_combination in itertools.combinations(station_names, drop_count):
                    logger.info(f'Calculating for stations {station_combination} and dates {start_date} to {end_date}')
                    non_station_df = self.drop_station_columns(date_range_df, station_combination)
                    ttest_rejected, chi2_rejected = self.congruency_test(df=non_station_df, method=method,
                                                                         calculation="specific_date",
                                                                         start_date=start_date, end_date=end_date,
                                                                         sigma_0=sigma_0, print_log=False, Qv=Qv)
                    if not (ttest_rejected or chi2_rejected):
                        station_cols = [col for col in date_range_df.columns if
                                        any(station in col for station in station_combination)]
                        start_station_df = date_range_df[station_cols].loc[date_range_df['Date'] == start_date]
                        end_station_df = date_range_df[station_cols].loc[date_range_df['Date'] == end_date]
                        offset_size = np.sqrt(np.sum((start_station_df.values - end_station_df.values) ** 2))
                        for station in station_combination:
                            offset_points.append((start_date, end_date, station, offset_size))
                        break
                if offset_points and offset_points[-1][0] == start_date and offset_points[-1][1] == end_date:
                    break

        return offset_points

    def drop_station_columns(self, df, stations):
        # Drop columns corresponding to specific stations
        station_cols = [col for col in df.columns if any(station in col for station in stations)]
        return df.drop(station_cols, axis=1)

    def geometric_chi_test_calc(self, time_series_frag, sigma, sigma_0):
        # Mask NaN values
        mask = ~np.isnan(time_series_frag)
        L = time_series_frag[mask]

        sigma = np.ones(L.size) * 0.01
        # задаем массив эпох
        t = np.arange(0, L.size, 1)
        A = np.column_stack((np.ones(L.size), t))
        P = np.diag(1 / (sigma * sigma_0))

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

    def geometric_chi_test_statictics(self, time_series_df,
                                      window_size,
                                      sigma_0):
        X_WLS = []
        Qv_WLS = []
        wls_times = []
        initial_values_X = []
        initial_values_Qv = []

        # sigma = np.ones(int(time_series_df.shape[0] / window)) * 0.01

        time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
        time_series_df.set_index('Date', inplace=True)

        # Calculate the initial values
        start_time = time_series_df.index[0]
        end_time = start_time + pd.Timedelta(window_size)
        fragment = time_series_df[(time_series_df.index >= start_time) & (time_series_df.index < end_time)]
        for col in time_series_df.columns:
            x_LS_first, _, Qx, _ = self.geometric_chi_test_calc(time_series_frag=fragment[col],
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
                x_LS_first, x_LS, Qx, mu = self.geometric_chi_test_calc(time_series_frag=fragment[col],
                                                                   sigma=np.ones(fragment.shape[0]) * 0.01,
                                                                   sigma_0=sigma_0)

                x_LS_values.append(x_LS)
                Qx_values.append(Qx)

            X_WLS.append(x_LS_values)
            Qv_WLS.append(Qx_values)

            # Move to the next fragment
            start_time = end_time
            end_time = start_time + pd.Timedelta(window_size)
            i += 1

        # Convert the lists to numpy arrays
        X_WLS = np.array(X_WLS)
        Qv_WLS = np.array(Qv_WLS)

        time_series_df.reset_index(inplace=True)
        wls_df = pd.DataFrame(columns=time_series_df.columns)
        Qv_df = pd.DataFrame(columns=time_series_df.columns)

        # Assign wls_times to the first column of wls_df
        wls_df['Date'] = wls_times
        Qv_df['Date'] = wls_times

        # Assign the values of X_WLS to the remaining columns of wls_df
        wls_df.iloc[:, 1:] = X_WLS
        Qv_df.iloc[:, 1:] = Qv_WLS

        wls_df['Date'] = pd.to_datetime(wls_df['Date'])
        Qv_df['Date'] = pd.to_datetime(Qv_df['Date'])

        # Calculate the test statistic
        test_statistic = np.zeros((X_WLS.shape[0] - 1))
        for l in range(X_WLS.shape[0] - 1):
            Qv = Qv_WLS[l] + Qv_WLS[l + 1]
            d = X_WLS[l] - X_WLS[l + 1]
            Qdd = np.diag(Qv)
            test_statistic[l] = d.transpose().dot(Qdd).dot(d) / (sigma_0 ** 2)

        return X_WLS, Qv_WLS, test_statistic, wls_df, Qv_df

    def interpolate_missing_values(self, df):
        """
        Interpolate missing values in the dataframe.

        Parameters:
        df (pandas.DataFrame): The input dataframe.

        Returns:
        pandas.DataFrame: The dataframe with interpolated missing values.
        """
        df_interpolated = df.interpolate(method='linear', limit_direction='both')
        return df_interpolated

    def filter_data(self, df: pd.DataFrame, kernel_size: int) -> pd.DataFrame:
        """
        Filter the data using a median filter.

        Parameters:
        df (pandas.DataFrame): The input dataframe.
        kernel_size (int): The size of the median filter kernel.

        Returns:
        pandas.DataFrame: The filtered dataframe.
        """
        df_filtered = df.copy()
        df_filtered.iloc[:, 1:] = df_filtered.iloc[:, 1:].apply(lambda x: medfilt(x, kernel_size=kernel_size))
        return df_filtered

    def perform_wls(self, df, window_size, sigma_0):
        station_names = list(set(df.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

        raw_dfs = []
        filtered_dfs = []
        wls_dfs = []
        Qv_dfs = []

        for station in station_names:
            # Get the columns corresponding to the current station
            station_cols = ['Date'] + [col for col in df.columns if station in col]

            # Extract the columns for the current station
            station_df = df[station_cols]

            # Interpolate missing values
            station_df = self.interpolate_missing_values(station_df)

            # Filter the data
            station_df_filtered = self.filter_data(station_df, kernel_size=11)

            # Perform the geometric chi test
            X_WLS, Qv_WLS, test_statistic, wls_df, Qv_df = self.geometric_chi_test_statictics(time_series_df=station_df_filtered,
                                                                                  window_size=window_size,
                                                                                  sigma_0=sigma_0)
            # Append the wls_df to the list
            raw_dfs.append(station_df)
            filtered_dfs.append(station_df_filtered)
            wls_dfs.append(wls_df)
            Qv_dfs.append(Qv_df)

        # Concatenate the wls_dfs along the columns (axis=1)
        raw = pd.concat(raw_dfs, axis=1)
        filtered = pd.concat(filtered_dfs, axis=1)
        wls = pd.concat(wls_dfs, axis=1)
        Qv = pd.concat(Qv_dfs, axis=1)

        # Remove duplicate 'Date' columns
        raw = raw.loc[:, ~raw.columns.duplicated()]
        filtered = filtered.loc[:, ~filtered.columns.duplicated()]
        wls = wls.loc[:, ~wls.columns.duplicated()]
        Qv = Qv.loc[:, ~Qv.columns.duplicated()]

        return wls, raw, filtered, Qv
        # return pd.DataFrame(wls), pd.DataFrame(raw), pd.DataFrame(filtered)





    def save_html_report(self, report_data, output_path):
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


def select_file():
    file_path = filedialog.askopenfilename(title="Select input file",
                                           filetypes=[("CSV files", "*.csv")])
    return file_path


def main():
    """
    The main function.
    """

    file_path = select_file()
    df = pd.read_csv(file_path, delimiter=';')

    test = Tests(df)

    window_size = '1min'
    wls, raw, filtered, Qv = test.perform_wls(df, window_size, 0.015)
    wls['Date'] = wls['Date'].dt.to_pydatetime()

    # Extract station names from column names
    stations = list(set(wls.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

    offset_points = test.find_offset_points(df=wls, method='coordinate_based', sigma_0=0.005, Qv=Qv, max_drop=2)

    offsets_table = pd.DataFrame(offset_points, columns=['Start_date', 'End_date', 'Station', 'Offset size'])
    offsets_html_table = offsets_table.to_html(index=False)
    offsets_table.to_csv('Data/moments.csv', sep=';', index=False)

    # Get the log contents
    string_io_handler.flush()
    log_contents = string_io_handler.stream.getvalue()

    # Group offsets by station
    station_offsets = {}
    for start_date, end_date, station, offset_size in offset_points:
        if station not in station_offsets:
            station_offsets[station] = []
        station_offsets[station].append((start_date, end_date))

    report_data = {
        'file_name': file_path,
        'total_tests': (len(wls['Date'])-1),
        'stations_length': len(stations),
        'stations_names': stations,
        'window_size': window_size,
        'offset_points': offsets_html_table,
        'offset_plots': '',
        'triangulation_map': '',
        'log_contents': log_contents}

    # Create plots for each station with multiple offsets
    for station, offsets in station_offsets.items():
        station_df_wls = wls[[col for col in df.columns if station in col or col == 'Date']]
        station_df_raw = raw[[col for col in df.columns if station in col or col == 'Date']]
        station_df_filtered = filtered[[col for col in df.columns if station in col or col == 'Date']]

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

        # Add Raw data plot on primary y-axis
        fig.add_trace(go.Scatter(x=station_df_raw['Date'], y=x_values_raw, mode='lines', name='Raw data',
                                 line=dict(color='lightgray'), legendgroup='Raw data', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=station_df_raw['Date'], y=y_values_raw, mode='lines', name='Raw data',
                                 line=dict(color='lightgray'), legendgroup='Raw data', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=station_df_raw['Date'], y=z_values_raw, mode='lines', name='Raw data',
                                 line=dict(color='lightgray'), legendgroup='Raw data', showlegend=False), row=3, col=1)

        # Add Filtered data plot on secondary y-axis (twinx axis)
        fig.add_trace(go.Scatter(x=station_df_filtered['Date'], y=x_values_fil, mode='lines', name='Filtered data',
                                 line=dict(color='blue'), yaxis='y2', legendgroup='Filtered data', showlegend=True),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=station_df_filtered['Date'], y=y_values_fil, mode='lines', name='Filtered data',
                                 line=dict(color='blue'), yaxis='y2', legendgroup='Filtered data', showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=station_df_filtered['Date'], y=z_values_fil, mode='lines', name='Filtered data',
                                 line=dict(color='blue'), yaxis='y2', legendgroup='Filtered data', showlegend=False),
                      row=3, col=1)

        # Add WLS Estimate plot on tertiary y-axis (twinx axis)
        fig.add_trace(go.Scatter(x=station_df_wls['Date'], y=x_values, mode='lines', name='WLS Estimate',
                                 line=dict(color='red'), yaxis='y3', legendgroup='WLS Estimate', showlegend=True),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=station_df_wls['Date'], y=y_values, mode='lines', name='WLS Estimate',
                                 line=dict(color='red'), yaxis='y3', legendgroup='WLS Estimate', showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=station_df_wls['Date'], y=z_values, mode='lines', name='WLS Estimate',
                                 line=dict(color='red'), yaxis='y3', legendgroup='WLS Estimate', showlegend=False),
                      row=3, col=1)

        # Highlight all offset periods for the station
        for start_date, end_date in offsets:
            fig.add_vrect(x0=start_date, x1=end_date,
                          fillcolor='red', opacity=0.5,
                          layer='below', line_width=0)

        for i in range(3):
            fig.update_xaxes(showgrid=False, row=i + 1, col=1)
            fig.update_yaxes(showgrid=False, row=i + 1, col=1)
            fig.update_yaxes(showgrid=False, row=i + 1, col=1, secondary_y=True)
            fig.update_yaxes(showgrid=False, row=i + 1, col=1, secondary_y=True, tertiary=True)
            if i < 2:  # Hide time labels on the first two subplots
                fig.update_xaxes(tickvals=[], row=i + 1, col=1)
            else:  # Show time labels on the last subplot
                fig.update_xaxes(tickformat='%H:%M:%S', row=i + 1, col=1)

        fig.update_layout(height=600, width=1200,  # Adjust the figure size
                          title_text=f'Offsets found in {station} station: ',
                          margin=dict(l=10, r=10, t=50, b=10))

        # Convert the figure to HTML (with Plotly JavaScript library embedded)
        html_img = pio.to_html(fig, include_plotlyjs=True, full_html=False)

        # Add the HTML image to your report
        report_data['offset_plots'] += html_img + "<br>"

    test.save_html_report(report_data=report_data, output_path='congruency_test_report'+'.html')

    # Remove the StringIO handler
    logger.removeHandler(string_io_handler)


if __name__ == "__main__":
    main()

