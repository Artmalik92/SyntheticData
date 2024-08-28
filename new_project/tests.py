from Synthetic_data import SyntheticData
import numpy as np
from numpy.linalg import pinv
import pandas as pd
from pandas import DataFrame
from scipy.stats import ttest_1samp, shapiro, chi2, f_oneway, chisquare
from scipy.optimize import minimize
from jinja2 import Template
from io import StringIO, BytesIO
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import tkinter as tk
from tkinter import filedialog
import base64
from scipy import signal
from dateutil.parser import parse
import contextily as ctx
import pyproj


# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a StringIO handler
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
        self.dates = df['Date'].unique()  # Store the unique dates

    def congruency_test(self, df, method: str,
                        calculation: str = "all_dates",
                        start_date: str = None,
                        end_date: str = None,
                        threshold: float = 0.05,
                        sigma_0: float = 0.005,
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
            sigma_0 (float, optional): The sigma_0 value for the Chi2 test. Defaults to 0.005.
            print_log (bool, optional): Whether to print the log. Defaults to True.

        Returns:
            tuple: A tuple containing the rejected dates for the T-test and Chi2 test.
        """

        ttest_rejected_dates = []
        chi2_rejected_dates = []

        if calculation == "all_dates":
            self._run_all_dates(df, method, threshold, sigma_0, ttest_rejected_dates, chi2_rejected_dates)
        elif calculation == "specific_date":
            self._run_specific_date(df, method, start_date, end_date, threshold, sigma_0, ttest_rejected_dates,
                                    chi2_rejected_dates)

        if print_log:
            self._print_results(ttest_rejected_dates, chi2_rejected_dates)

        return ttest_rejected_dates, chi2_rejected_dates

    def _run_all_dates(self, df, method, threshold, sigma_0, ttest_rejected_dates, chi2_rejected_dates):
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
            logger.info(f"Calculating for dates: {start_date} and {end_date}")
            # Вызов функции congruency_test для каждой пары дат
            self._run_specific_date(df, method, start_date, end_date, threshold,
                                    sigma_0, ttest_rejected_dates, chi2_rejected_dates)

    def _run_specific_date(self, df, method, start_date, end_date, threshold, sigma_0, ttest_rejected_dates, chi2_rejected_dates):
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
        #std_dev = np.std(np.array(raz_list))
        #sigma_0 = 0.005 #0.0075 0.005
        #  sigma_0 = std_dev
        #print(f"std dev of data: {std_dev}")
        logger.info('Shapiro: %s', shapiro(raz_list))

        '''ttest_result, pvalue = self._perform_ttest(raz_list, threshold)
        print("T-test: ", end="")
        if ttest_result:
            ttest_rejected_dates.append((start_date, end_date, pvalue))
            print(Fore.RED + f"Нулевая гипотеза отвергается, pvalue = {round(pvalue, 3)}")
        else:
            print(Fore.GREEN + f"Нулевая гипотеза не отвергается, pvalue = {round(pvalue, 3)}")
        print(Fore.RESET, end="")'''

        chi2_result, K, test_value = self._perform_chi2_test(raz_list, sigma_0, threshold)
        if chi2_result:
            chi2_rejected_dates.append((start_date, end_date))
            logger.info("Chi-2: " + f"<span style='color:red'>Null hypothesis rejected, "
                                    f"testvalue = {round(test_value, 3)}, K = {round(K, 3)}</span>")
        else:
            logger.info("Chi-2: " + f"Null hypothesis not rejected, testvalue = {round(test_value, 3)},"
                        f" K = {round(K, 3)}")
        logger.info("")

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

    def find_offset_points(self, df, method, sigma_0):
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
                                                                         calculation="all_dates", sigma_0=sigma_0)
        rejected_dates = ttest_rejected_dates + chi2_rejected_dates

        logger.info('<h2>Finding the offset points:</h2>')

        for start_date, end_date in rejected_dates:
            temp_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            for station in temp_df['Station'].unique():
                logger.info(f'calculating for station {station}')
                temp_temp_df = temp_df[temp_df['Station'] != station]
                ttest_rejected, chi2_rejected = self.congruency_test(temp_temp_df, method, calculation="specific_date",
                                                                     start_date=start_date, end_date=end_date,
                                                                     sigma_0=sigma_0,
                                                                     print_log=False)
                if not (ttest_rejected or chi2_rejected):
                    offset_points.append((start_date, end_date, station))

        return offset_points

    def auto_sigma(self, df, method, sigma_range=(0.005, 0.01), step_size=0.001, threshold=0.05):
        best_sigma_0 = None
        best_diff = float('inf')

        sigma_values = np.arange(sigma_range[0], sigma_range[1] + step_size, step_size)

        for sigma_0 in sigma_values:
            logger.info(f"<h3>Testing sigma_0 = {sigma_0}</h3>")
            _, chi2_rejected_dates = self.congruency_test(df, method, sigma_0=sigma_0, threshold=threshold,
                                                          print_log=False)
            if chi2_rejected_dates:
                start_date, end_date = chi2_rejected_dates[0]
                stations = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'Station'].unique()
                raz_list = self._calculate_raz_list(df, method, start_date, end_date, stations)
                _, K, test_value = self._perform_chi2_test(raz_list, sigma_0, threshold)

                diff = abs(K - test_value)
                if diff < best_diff:
                    best_diff = diff
                    best_sigma_0 = sigma_0
                logger.info(f"<h3>Current sigma_0 = {sigma_0}, diff = {diff}, best_sigma_0 = {best_sigma_0}</h3>")

        if best_sigma_0 is None:
            print("No rejected dates found")
            return None

        logger.info(f"<h3>Optimal sigma_0 found: {best_sigma_0}</h3>")
        return best_sigma_0

    def _objective(self, sigma_0, d, threshold):
        """
        Objective function to minimize the absolute difference between K and the test value.

        Args:
            sigma_0 (float): The sigma_0 value being tested.
            d (np.array): Array of differences.
            threshold (float): The threshold value for the test.

        Returns:
            float: The absolute difference between K and the test value.
        """
        Qdd = np.eye(d.shape[0])
        K = d.transpose().dot(pinv(Qdd)).dot(d) / (sigma_0 ** 2)
        test_value = chi2.ppf(df=d.shape[0], q=threshold)
        return abs(K - test_value)

    def _perform_chi2_test_ai(self, raz_list, threshold):
        """
        Performs a Chi2 test on the given list of differences, automatically optimizing sigma_0.

        Args:
            raz_list (list): The list of differences.
            threshold (float): The threshold value for the test.

        Returns:
            tuple: A tuple containing the result of the Chi2 test, K-value, test value, and optimal sigma_0.
        """
        d = np.array(raz_list)

        # Initial guess for sigma_0
        initial_sigma_0 = 1.0

        # Define bounds to ensure sigma_0 is positive
        bounds = [(1e-6, None)]

        # Use minimize to find the best sigma_0 with bounds
        result = minimize(self._objective, initial_sigma_0, args=(d, threshold), bounds=bounds, method='L-BFGS-B')

        # Optimal sigma_0
        optimal_sigma_0 = result.x[0]

        # Calculate K and test_value with optimal sigma_0
        Qdd = np.eye(d.shape[0])
        K = d.transpose().dot(pinv(Qdd)).dot(d) / (optimal_sigma_0 ** 2)
        test_value = chi2.ppf(df=d.shape[0], q=threshold)

        if K > test_value:
            return True, K, test_value, optimal_sigma_0
        else:
            return False, K, test_value, optimal_sigma_0


    def _print_results(self, ttest_rejected_dates, chi2_rejected_dates):
        """
        Prints the results of the congruence test.

        Args:
            ttest_rejected_dates (list): A list of rejected dates for the T-test.
            chi2_rejected_dates (list): A list of rejected dates for the Chi2 test.
        """
        logger.info("----------------------------")
        logger.info(f"Всего выполнено тестов: {len(self.dates)}.")
        logger.info(f"Хи-квадрат, не отвергнуто: {len(self.dates) - len(chi2_rejected_dates)}, "
              f"отвергнуто: {len(chi2_rejected_dates)} ({len(chi2_rejected_dates) / len(self.dates) * 100:.2f}%)")
        logger.info(f"Т-тест, не отвергнуто: {len(self.dates) - len(ttest_rejected_dates)}, "
              f"отвергнуто: {len(ttest_rejected_dates)} ({len(ttest_rejected_dates) / len(self.dates) * 100:.2f}%)")

        logger.info(f"Chi2 rejected dates: {chi2_rejected_dates}")

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
            <p><strong>Total Tests:</strong> {{ total_tests }}</p>
            <p><strong>Chi2 Rejected Dates:</strong> {{ chi2_rejected_dates }}</p>
            <p><strong>T-Test Rejected Dates:</strong> {{ ttest_rejected_dates }}</p>
            <p><strong>Optimal Sigma:</strong> {{ best_sigma_0 }}</p>
            <p><strong>Offset Points:</strong> {{ offset_points }}</p>
            <h2>Stations Map:</h2>
            {{ triangulation_map }}
            <h2>Points with offsets:</h2>
            {{ offset_plots }}
            <h2>Detailed Log:</h2>
            <pre>{{ log_contents }}</pre>
        </body>
        </html>
        """
        template = Template(html_template)
        html_content = template.render(**report_data)

        with open(output_path, 'w') as file:
            file.write(html_content)
        return output_path


def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select input file",
                                           filetypes=[("CSV files", "*.csv")],
                                           initialdir="E:/docs_for_univer/Diplom_project/diplom/new_project/Data")
    return file_path


def main():
    """
    The main function.
    """

    file_path = select_file()
    df = pd.read_csv(file_path, delimiter=';')

    #df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce').dt.round('s')
    #df['Date'] = df['Date'].apply(parse).dt.normalize()
    #df['Date'].apply(parse).dt.floor('s')

    test = Tests(df)

    #best_sigma = test.auto_sigma(df, method='coordinate_based', sigma_range=(0.005, 0.01))


    stations = df['Station'].unique()
    raz_list = test._calculate_raz_list(df, method='coordinate_based', start_date=min(df['Date']),
                                        end_date=max(df['Date']), stations=stations)
    _, _, _, best_sigma = test._perform_chi2_test_ai(raz_list, threshold=0.05)



    offset_points = test.find_offset_points(df=df, method='coordinate_based', sigma_0=0.005)

    ttest_rejected_dates, chi2_rejected_dates = test.congruency_test(df, method='coordinate_based')

    # Get the log contents
    string_io_handler.flush()
    log_contents = string_io_handler.stream.getvalue()

    # Group offsets by station
    station_offsets = {}
    for start_date, end_date, station in offset_points:
        if station not in station_offsets:
            station_offsets[station] = []
        station_offsets[station].append((start_date, end_date))

    report_data = {
        'total_tests': len(test.dates),
        'chi2_rejected_dates': [f"{start_date} to {end_date}" for start_date, end_date in chi2_rejected_dates],
        'ttest_rejected_dates': [f"{start_date} to {end_date}" for start_date, end_date in ttest_rejected_dates],
        'best_sigma_0': best_sigma,
        'offset_points': offset_points,
        'offset_plots': '',
        'triangulation_map': '',
        'log_contents': log_contents}

    #df_last_date = df[df['Date'] == df['Date'].max()]
    #df_last_date = SyntheticData.my_ecef2geodetic(df_last_date)

    try:
        # Find the epoch with the most stations
        station_counts = df['Date'].value_counts()
        most_stations_epoch = station_counts.index[0]

        # Filter the data for the most stations epoch
        df_most_stations = df[df['Date'] == most_stations_epoch]
        df_last_date = df_most_stations

        # Define the ENU and BLH CRS objects
        enu_crs = pyproj.CRS.from_epsg(4978)
        # Define the Web Mercator CRS object
        webmercator_crs = pyproj.CRS.from_epsg(3857)

        # Define the projection systems
        blh_proj = pyproj.Proj(proj='longlat', ellps='WGS84', datum='WGS84')
        webmercator_proj = pyproj.Proj(init='epsg:3857')

        """# Convert BLH coordinates to Web Mercator
        df_last_date['x_webmercator'], df_last_date['y_webmercator'] = pyproj.transform(blh_proj, webmercator_proj,
                                                                                        df_last_date['L'].values,
                                                                                        df_last_date['B'].values)
        """
        # Transform ENU coordinates to Web Mercator
        df_last_date['x_webmercator'], df_last_date['y_webmercator'], _ = pyproj.transform(
            enu_crs, webmercator_crs, df_last_date['X'].values, df_last_date['Y'].values, df_last_date['Z'].values)

        # Create the triangulation plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df_last_date['x_webmercator'], df_last_date['y_webmercator'])

        # Add a real earth map as the background
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Plot station names
        for index, row in df_last_date.iterrows():
            ax.annotate(row['Station'], xy=(row['x_webmercator'], row['y_webmercator']), xytext=(0, 10),
                        textcoords='offset points', ha='center', va='bottom')

        # Highlight stations with offsets
        offset_stations = [station for station, offsets in station_offsets.items() if offsets]
        for station in offset_stations:
            station_df = df_last_date[df_last_date['Station'] == station]
            ax.scatter(station_df['x_webmercator'], station_df['y_webmercator'], c='red', marker='*', s=100)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        triangulation_map = base64.b64encode(buf.getvalue()).decode('utf-8')
        report_data['triangulation_map'] = f"<img src='data:image/png;base64,{triangulation_map}'><br>"
        plt.close(fig)
    except Exception as e:
        print(e)

    # Create plots for each station with multiple offsets
    for station, offsets in station_offsets.items():
        station_df = df[df['Station'] == station]
        dates = station_df['Date']
        x_values = station_df['X']
        y_values = station_df['Y']
        z_values = station_df['Z']

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        ax[0].plot(dates, x_values, label='X')
        ax[1].plot(dates, y_values, label='Y')
        ax[2].plot(dates, z_values, label='Z')

        # Highlight all offset periods for the station
        for start_date, end_date in offsets:
            for a in ax:
                a.axvspan(start_date, end_date, color='red', alpha=0.5)
                # Add label for offset point
                a.annotate(f"{start_date} - {end_date}", xy=(start_date, 0), xytext=(0, 10),
                           textcoords='offset points', ha='center', va='bottom', rotation=45)

        for a in ax:
            a.legend()

        plt.gcf().autofmt_xdate()
        plt.suptitle(f"Station {station}")

        # Set the plot width to 100%
        fig.tight_layout()
        fig.set_size_inches((12, 6))  # Set the figure size

        # Save the plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Encode the image data as base64
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Add the image to the HTML report
        report_data['offset_plots'] += f"<img src='data:image/png;base64,{img_data}'><br>"

        plt.close()

    test.save_html_report(report_data=report_data, output_path='Data/congruency_test_report_2024_08_16(concatenated)'+'.html')

    # Remove the StringIO handler
    logger.removeHandler(string_io_handler)


if __name__ == "__main__":
    main()
