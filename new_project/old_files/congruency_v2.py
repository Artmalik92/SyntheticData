import itertools
import numpy as np
import pandas as pd
from colorama import init, Fore
from scipy.stats import ttest_1samp, shapiro


def congruency_test(df, epoch_0):
    threshold = 1e-3  # пороговое значение RMSD для определения геометрической целостности

    for date in epoch_0:
        start_date, end_date = date[0], date[1]
        stations = df.loc[(df['Date'] >= start_date) & (
                    df['Date'] <= end_date), 'Station'].unique()  # список всех станций в интервале
        line_lists = {}  # словарь списков базовых линий для каждой станции на каждую дату

        for station in stations:
            station_df = df.loc[(df['Date'] >= start_date) &
                                (df['Date'] <= end_date) & (df['Station'] == station)]
            lines = []

            for other_station in stations:
                if other_station == station:
                    continue

                other_station_df = df.loc[(df['Date'] >= start_date) &
                                          (df['Date'] <= end_date) & (df['Station'] == other_station)]
                dist = np.sqrt((station_df['X'].values - other_station_df['X'].values) ** 2
                               + (station_df['Y'].values - other_station_df['Y'].values) ** 2
                               + (station_df['Z'].values - other_station_df['Z'].values) ** 2)
                if len(dist) > 0:
                    line = [station, other_station, dist.mean()]
                    lines.append(line)

            line_lists[station] = lines

        for interval2 in epoch_0:
            if date == interval2:
                continue

            start_date2, end_date2 = interval2[0], interval2[1]
            line_lists2 = {}  # словарь списков базовых линий для каждой станции на второй интервал

            for station in stations:
                station_df = df.loc[
                    (df['Date'] >= start_date2) & (df['Date'] <= end_date2) & (df['Station'] == station)]
                lines = []

                for other_station in stations:
                    if other_station == station:
                        continue

                    other_station_df = df.loc[
                        (df['Date'] >= start_date2) & (df['Date'] <= end_date2) & (df['Station'] == other_station)]

                    dist = np.sqrt((station_df['X'].values - other_station_df['X'].values) ** 2
                                   + (station_df['Y'].values - other_station_df['Y'].values) ** 2
                                   + (station_df['Z'].values - other_station_df['Z'].values) ** 2)
                    if len(dist) > 0:
                        line = [station, other_station, dist.mean()]
                        lines.append(line)

                line_lists2[station] = lines

            raz_list = []

            for station in stations:
                for line1, line2 in itertools.product(line_lists[station], line_lists2[station]):
                    if line1[0] == line2[0] and line1[1] == line2[1]:
                        raz_list.append(line1[2] - line2[2])

    #print(shapiro(pd.DataFrame(raz_list)))  # Тест Шапиро-Уилка

    pvalue = ttest_1samp(a=raz_list, popmean=0)[1]

    if pvalue <= 0.05:
        print(Fore.RED + f"Гипотеза отвергается, pvalue = {round(pvalue, 6)}")
    else:
        print(Fore.GREEN + f"Гипотеза не отвергается, pvalue = {round(pvalue, 6)}")


df = pd.read_csv('Data/testfile_3points_noises_anomalies.csv', delimiter=';')
congruency_test(df,  [('2022-01-02', '2022-01-02'), ('2023-01-01', '2023-01-01')])


'''for station in stations:
    for line1, line2 in itertools.product(line_lists[station], line_lists2[station]):
        if line1[0] == line2[0] and line1[1] == line2[1]:
            rmsd_list.append((line1[2] - line2[2]) ** 2)

print(rmsd_list)
if len(rmsd_list) > 0:
    rmsd = np.sqrt(np.mean(rmsd_list))
else:
    rmsd = 0

if len(line_lists[station]) > 0 and len(line_lists2[station]) > 0:
    if rmsd > threshold:
        print(Fore.RED + f"Тест не пройден на даты "
                         f"{start_date} - {end_date} и {start_date2} - {end_date2}. СКО = {round(rmsd, 6)}")
    else:
        print(Fore.GREEN + f"Тест пройден на даты "
                           f"{start_date} - {end_date} и {start_date2} - {end_date2}. СКО = {round(rmsd, 6)}")
else:
    print(f"На интервале {start_date} - {end_date} или {start_date2} - {end_date2} нет данных.")
'''