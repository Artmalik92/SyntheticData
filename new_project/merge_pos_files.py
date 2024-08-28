import pandas as pd
from moncenterlib.gnss.gnss_time_series import parse_pos_file
from moncenterlib.tools import get_files_from_dir
import json
import datetime


def resample(data: list,
             station: str,
             resample_interval: str = None):

    df = pd.DataFrame(data)
    df = df.iloc[:, :4]

    df[0] = pd.to_datetime(df[0])

    df[1], df[2], df[3] = pd.to_numeric(df[1]), pd.to_numeric(df[2]), pd.to_numeric(df[3])

    if resample_interval is not None:
        df = df.resample(resample_interval, on='Date').mean()
        df.reset_index(inplace=True)

    # Rename
    df = df.rename(columns={0: 'Date', 1: f'x_{station}', 2: f'y_{station}', 3: f'z_{station}'})

    # Reorder the columns
    df = df.loc[:, ['Date', f'x_{station}', f'y_{station}', f'z_{station}']]

    return df


def makefile(point_names: list,
             resample_interval: str = None,
             start_date: str = None,
             end_date: str = None,
             zero_epoch_coords: dict = None):

    file_paths = {}

    for p in point_names:
        file_paths[p] = list(filter(lambda x: ".pos" in x, get_files_from_dir(f"2024_08_16/{p}", False)))

    # list to store the DataFrames
    dfs = []

    '''# Process zero epoch coordinates
    if zero_epoch_coords is not None:
        zero_epoch_df = pd.DataFrame.from_dict(zero_epoch_coords, orient='index', columns=['X', 'Y', 'Z'])
        zero_epoch_df.reset_index(inplace=True)
        zero_epoch_df = zero_epoch_df.rename(columns={'index': 'Station'})
        zero_epoch_df['Date'] = '00.00.0000 00:00:00'
        zero_epoch_df = zero_epoch_df.loc[:, ['Date', 'Station', 'X', 'Y', 'Z']]
        zero_epoch_df['Date'] = pd.to_datetime(zero_epoch_df['Date'])
        dfs.append(zero_epoch_df)'''

    for station, files in file_paths.items():
        for file in files:
            # Read the data from the file
            header, data = parse_pos_file(path2file=file)

            # Skip the file if it's empty
            if not data:
                print(f"File {file} is empty. Skipping...")
                continue

            # Resample the data
            resampled_data = resample(data, station, resample_interval)

            # Filter the data for the specified period
            if all([start_date is not None, end_date is not None]):
                resampled_data = resampled_data[(resampled_data['Date'] >= start_date)
                                                & (resampled_data['Date'] <= end_date)]

            # Append the DataFrame to the list
            dfs.append(resampled_data)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='inner')

    return merged_df


points = ["SNSK00RUS", "SNSK01RUS", "SNSK02RUS", "SNSK03RUS"]

zero_epoch_coordinates = json.load(open('2024_08_16/first_epoch.json'))

merged_data = makefile(point_names=points) #, zero_epoch_coords=zero_epoch_coordinates)

# Save the merged DataFrame to a CSV file
merged_data.to_csv('Data/merged_2024-08-16(feature).csv', sep=';', index=False)

print('Done')



