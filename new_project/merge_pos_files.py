import pandas as pd
from moncenterlib.gnss.gnss_time_series import parse_pos_file
from moncenterlib.tools import get_files_from_dir


def resample(data: list,
             station: str,
             resample_interval: str = None):

    df = pd.DataFrame(data)
    df = df.iloc[:, :4]
    df = df.rename(columns={0: 'Date', 1: 'X', 2: 'Y', 3: 'Z'})

    df['Date'] = pd.to_datetime(df['Date'])

    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    df['Z'] = pd.to_numeric(df['Z'])

    if resample_interval is not None:
        df = df.resample(resample_interval, on='Date').mean()
        df.reset_index(inplace=True)

    # Add the station name as a column
    df['Station'] = station
    # Reorder the columns
    df = df.loc[:, ['Date', 'Station', 'X', 'Y', 'Z']]

    return df


def makefile(point_names: list,
             resample_interval: str = None,
             start_date: str = None,
             end_date: str = None):

    file_paths = {}

    for p in point_names:
        file_paths[p] = list(filter(lambda x: ".pos" in x, get_files_from_dir(f"Data_pos/{p}", False)))

    # list to store the DataFrames
    dfs = []

    for station, files in file_paths.items():
        for file in files:
            # Read the data from the file
            header, data = parse_pos_file(path2file=file)

            # Resample the data
            resampled_data = resample(data, station, resample_interval)

            # Filter the data for the specified period
            if all([start_date is not None, end_date is not None]):
                resampled_data = resampled_data[(resampled_data['Date'] >= start_date)
                                                & (resampled_data['Date'] <= end_date)]

            # Append the DataFrame to the list
            dfs.append(resampled_data)

    # Concatenate the list of DataFrames into a single DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df


points = ["AMDR", "ARKH", "AST3", "BARE", "BELG",
          "BORO", "CHIT", "CNG1", "DKSN", "ANDR"]

merged_data = makefile(point_names=points,
                       start_date='2020-01-09 23:00:00',
                       end_date='2020-01-10 01:00:00')

# Save the merged DataFrame to a CSV file
merged_data.to_csv('Data/merged_data_30sec_dates_2020_01_09_and_2020_01_10.csv', sep=';', index=False)

print('Done')



