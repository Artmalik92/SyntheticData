import pandas as pd
import numpy as np
from scipy.stats import t
import math


def rmsd(values):
    if len(values) == 0:
        return 0.0
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    return math.sqrt(sum(squared_diffs) / len(values))


def congruency_test(df):
    # Получаем список всех пунктов
    points = df['Station'].unique()

    # Сортируем данные по дате измерений
    df = df.sort_values(by='Date')

    # Создаем новый DataFrame для хранения результатов
    result = []

    # Проходимся по всем парам пунктов и вычисляем длину базовых линий на начальную и конечную эпоху
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Выбираем данные для двух пунктов
            data1 = df[df['Station'] == points[i]]
            data2 = df[df['Station'] == points[j]]

            # Вычисляем длину базовой линии на начальную и конечную эпоху
            start_length = np.sqrt(
                (data1.iloc[0]['X'] - data2.iloc[0]['X']) ** 2 + (data1.iloc[0]['Y'] - data2.iloc[0]['Y']) ** 2 + (
                        data1.iloc[0]['Z'] - data2.iloc[0]['Z']) ** 2)
            end_length = np.sqrt(
                (data1.iloc[-1]['X'] - data2.iloc[-1]['X']) ** 2 + (data1.iloc[-1]['Y'] - data2.iloc[-1]['Y']) ** 2 + (
                        data1.iloc[-1]['Z'] - data2.iloc[-1]['Z']) ** 2)

            # Вычисляем разницу длин
            length_diff = end_length - start_length

            # Вычисляем p-value теста Стьюдента
            n = len(data1)
            mean_diff = np.mean(data1[['X', 'Y', 'Z']].values - data2[['X', 'Y', 'Z']].values, axis=0)
            std_diff = np.std(data1[['X', 'Y', 'Z']].values - data2[['X', 'Y', 'Z']].values, axis=0, ddof=1)
            t_value = mean_diff / (std_diff / math.sqrt(n))
            p_value = 2 * t.sf(np.abs(t_value), n - 1)

            # Добавляем результаты в DataFrame
            result.append({'point1': points[i], 'point2': points[j],
                           'start_length': start_length, 'end_length': end_length,
                           'length_diff': length_diff, 'p_value': p_value})

    result = pd.DataFrame(result)
    # Вычисляем RMSD для всех разностей длин
    if result is not None:
        rmsd_value = rmsd(result['length_diff'])
        print('RMSD value:', rmsd_value)
    else:
        print('Result is None')

    # Проверяем, является ли вся сеть конгруэнтной
    is_congruent = (result['p_value'] > 0.05)

    # Выводим результаты
    print(f'RMSD: {rmsd_value}')
    print(f'Is congruent: {is_congruent}')

    # Если в сети обнаруживаются аномальные сдвиги, даем информацию о том, где и когда они происходят
    if not is_congruent:
        # Выбираем все пары пунктов, у которых p-value меньше 0.05
        anomalies = result[result['p_value'] < 0.05]

        # Группируем результаты по пунктам
        grouped_anomalies = anomalies.groupby(['point1', 'point2'])

        # Выводим информацию о месте и времени аномалий
        for group, df in grouped_anomalies:
            print(f"Anomaly between {group[0]} and {group[1]} at {df.iloc[0]['Date']}")


df = pd.read_csv('Data/test_file_xyz.csv', delimiter=';')
congruency_test(df)
