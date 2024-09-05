import pandas as pd
from moncenterlib.gnss.gnss_time_series import parse_pos_file
from moncenterlib.tools import get_files_from_dir
import json
import datetime


def resample(data: list,
             station: str,
             resample_interval: str = None):
    """
    Resamples the input data for a given station.

    Args:
        data (list): Input data to be resampled.
        station (str): Name of the station.
        resample_interval (str, optional): Resampling interval. Defaults to None.

    Returns:
        pd.DataFrame: Resampled data.
    """

    df = pd.DataFrame(data)
    df = df.iloc[:, :4]  # Вытаскиваем из .pos файла первые 4 колонки (дата и координаты)

    df[0] = pd.to_datetime(df[0])  # Ставим правильный формат даты
    df[0] = df[0].dt.round('s')  # убираем миллисекунды

    # Поскольку координаты спарсились в формате string, переводим их в числа
    df[1], df[2], df[3] = pd.to_numeric(df[1]), pd.to_numeric(df[2]), pd.to_numeric(df[3])

    # Переименование колонок файла
    df = df.rename(columns={0: 'Date', 1: f'x_{station}', 2: f'y_{station}', 3: f'z_{station}'})

    # Ресемплинг файла с определенным интервалом даты (опционально)
    if resample_interval is not None:
        try:
            df = df.resample(resample_interval, on='Date', origin='epoch').first()
            df.reset_index(inplace=True)
        except Exception as e:
            print(e)

    # Определение порядка колонок
    df = df.loc[:, ['Date', f'x_{station}', f'y_{station}', f'z_{station}']]

    return df


def makefile(directory: str,
             point_names: list,
             resample_interval: str = None,
             start_date: str = None,
             end_date: str = None,
             zero_epoch_coords: dict = None,
             dropna: bool = True):
    """
    Creates a merged DataFrame from multiple POS files.

    Args:
        directory (str): Directory with pos-files.
        point_names (list): List of point names.
        resample_interval (str, optional): Resampling interval. Defaults to None.
        start_date (str, optional): Start date. Defaults to None.
        end_date (str, optional): End date. Defaults to None.
        zero_epoch_coords (dict, optional): Zero epoch coordinates. Defaults to None.
        dropna (bool, optional): Whether to drop rows with NaN values. Defaults to True.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """

    # Получаем пути к .pos файлам в указанной директории
    file_paths = {}
    for p in point_names:
        file_paths[p] = list(filter(lambda x: ".pos" in x, get_files_from_dir(f"{directory}/{p}", False)))

    dfs = []  # список для хранения .pos файлов, переведенных в формат DataFrame

    for station, files in file_paths.items():
        for file in files:
            # Парсинг .pos файла
            header, data = parse_pos_file(path2file=file)

            # Пропустить файл если он пуст
            if not data:
                print(f"File {file} is empty. Skipping...")
                continue

            # Ресемплинг файла + изменяем колонки
            resampled_data = resample(data, station, resample_interval)

            # Фильтрация файла на определенный временной период (опционально)
            if all([start_date is not None, end_date is not None]):
                resampled_data = resampled_data[(resampled_data['Date'] >= start_date)
                                                & (resampled_data['Date'] <= end_date)]

            # Добавление датафрейма в общий список
            dfs.append(resampled_data)

    # Последовательное объединение всех полученных датафреймов по дате
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    # Выброс рядов с пробелами / NaN / None
    if dropna:
        merged_df = merged_df.dropna()

    # Добавление координат на нулевую эпоху, если они указаны пользователем
    if zero_epoch_coords is not None:
        zero_epoch_data = {'Date': ['1900-01-01 00:00:00']}
        for station in point_names:  # перебираем список point_names
            if station in zero_epoch_coords:  # проверяем, есть ли станция в списке point_names
                # создаем колонки с координатами
                zero_epoch_data[f'x_{station}'] = [zero_epoch_coords[station][0]]
                zero_epoch_data[f'y_{station}'] = [zero_epoch_coords[station][1]]
                zero_epoch_data[f'z_{station}'] = [zero_epoch_coords[station][2]]
        zero_epoch_df = pd.DataFrame(zero_epoch_data)
        merged_df = pd.concat([zero_epoch_df, merged_df], ignore_index=True)  # Объединяем с основным файлом

    return merged_df


zero_epoch_coordinates = json.load(open('2024-08-29/first_epoch.json'))

merged_data = makefile(point_names=["SNSK00RUS", "SNSK01RUS", "SNSK02RUS", "SNSK03RUS"],
                       zero_epoch_coords=None,
                       dropna=False,
                       directory='2024-08-29',
                       resample_interval=None)

merged_data.to_csv('Data/merged_2024-08-29_with_na_nozeroepoch.csv', sep=';', index=False)

print('Done')



