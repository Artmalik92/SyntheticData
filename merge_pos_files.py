"""
GNSS Position File Merger

This module provides functionality to merge and process GNSS position files (.pos).
It handles resampling, filtering, and merging of multiple position files from different stations.
The output is a pandas DataFrame table in the .csv format.
The core module is designed to work only with this file format.
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from functools import reduce
from moncenterlib.gnss.gnss_time_series import parse_pos_file
from moncenterlib.tools import get_files_from_dir
import json
from datetime import datetime
from pathlib import Path
from config.logger_config import get_logger

logger = get_logger('log')  # getting logger instance (set up in process_pos.py)

# Constants
TIMESTAMP_COL = 0         # time stamps
COORD_COLS = [1, 2, 3]    # x, y, z coordinates
Q_STATUS_COL = 4          # Quality status
SIGMA_COLS = [6, 7, 8]    # sigma e, n, u
COVAR_COLS = [9, 10, 11]  # covariance pairs
SIGMA0_COL = 14           # sigma_0

# All columns to extract from pos file
POS_COLUMNS = [TIMESTAMP_COL] + COORD_COLS + [Q_STATUS_COL] + SIGMA_COLS + COVAR_COLS + [SIGMA0_COL]

# Fixed solution indicator
FIXED_SOLUTION_VALUE = 1

# Default zero epoch date
ZERO_EPOCH_DATE = '1900-01-01 00:00:00'


def resample(data: List[List],
             station: str,
             fixed_solution_only: Optional[bool] = False,
             resample_interval: Optional[str] = None) -> pd.DataFrame:
    """
    Resamples and processes GNSS position data for a given station.

    Column mapping from .pos file:
    [0] - timestamp (GPS Time)
    [1, 2, 3] - coordinates (x, y, z)
    [4] - Q-status (1: fixed solution, 2: float solution)
    [6, 7, 8] - sigma values (e, n, u)
    [9, 10, 11] - covariance pairs (en, nu, ue)
    [14] - sigma_0 (This value is not present in the original .pos files and was calculated manually)

    Args:
        data (list): Input data from .pos file as a list of lists
        station (str): Name of the station
        fixed_solution_only (bool, optional): If True, only fixed solutions (Q=1) are kept. Defaults to False.
        resample_interval (str, optional): Time interval for resampling (e.g., '1H' for hourly). Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with processed and resampled position data
    """
    logger.debug(f"Starting resampling for station {station}")
    try:
        # Validate input data structure
        if not data or not all(isinstance(row, (list, tuple)) for row in data):
            raise ValueError("Invalid data format: expected list of lists/tuples")

        # Pre-allocate columns
        df = pd.DataFrame(data, columns=range(len(data[0]))).iloc[:, POS_COLUMNS]

        # Convert timestamp to datetime and round to seconds
        df[0] = pd.to_datetime(df[0]).dt.round('s')

        # Convert string values to numeric
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename column names to Congruence-Test format
        column_mapping = {
            0: 'Date',
            1: f'x_{station}',
            2: f'y_{station}',
            3: f'z_{station}',
            4: f'Q_{station}',
            6: f'sde_{station}',
            7: f'sdn_{station}',
            8: f'sdu_{station}',
            9: f'sden_{station}',
            10: f'sdnu_{station}',
            11: f'sdue_{station}',
            14: f'sigma0_{station}'
        }
        df = df.rename(columns=column_mapping)

        # Resample data if interval is specified
        if resample_interval:
            try:
                df = df.resample(resample_interval, on='Date', origin='epoch').first()
                df.reset_index(inplace=True)
            except Exception as e:
                logger.error(f"Resampling failed for station {station}: {str(e)}")
                raise

        # Filter fixed solutions if requested
        if fixed_solution_only:
            df = df[df[f'Q_{station}'] == FIXED_SOLUTION_VALUE]

        # Drop Q-status column and reorder columns
        df = df.drop(columns=[f'Q_{station}'])
        columns_order = ['Date',
                         f'x_{station}', f'y_{station}', f'z_{station}',
                         f'sde_{station}', f'sdn_{station}', f'sdu_{station}',
                         f'sden_{station}', f'sdnu_{station}', f'sdue_{station}',
                         f'sigma0_{station}']

        return df[columns_order]

    except Exception as e:
        logger.error(f"Error processing data for station {station}: {str(e)}")
        raise


def process_file(file_path: str,
                 station: str,
                 fixed_solution_only: bool,
                 resample_interval: Optional[str],
                 start_date: Optional[str],
                 end_date: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Process a single POS file

    Args:
        file_path: Path to the .pos file
        station: Station name
        fixed_solution_only: Whether to keep only fixed solutions
        resample_interval: Time interval for resampling
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        pd.DataFrame: Processed DataFrame or None if processing fails
    """
    try:
        # Parse .pos using moncenterlib
        _, data = parse_pos_file(path2file=file_path)

        if not data:
            logger.warning(f"File {file_path} is empty. Skipping...")
            return None

        # changing columns and resampling if requested
        df = resample(data=data,
                      station=station,
                      fixed_solution_only=fixed_solution_only,
                      resample_interval=resample_interval)

        # Filtering for a specific time period (optional)
        if start_date and end_date:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        return df

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {str(e)}")
        return None


def makefile(directory: str,
             point_names: List[str],
             resample_interval: Optional[str] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             zero_epoch_coords: Optional[Dict] = None,
             dropna: Optional[bool] = False,
             fixed_solution_only: Optional[bool] = False) -> pd.DataFrame:
    """
    Creates a merged DataFrame from multiple POS files

    Args:
        directory (str): Directory containing .pos files
        point_names (list[str]): List of station names
        resample_interval (str, optional): Time interval for resampling
        start_date (str, optional): Start date for filtering
        end_date (str, optional): End date for filtering
        zero_epoch_coords (dict, optional): Dictionary of zero epoch coordinates
        dropna (bool, optional): Whether to drop rows with NaN values
        fixed_solution_only: Whether to keep only fixed solutions

    Returns:
        pd.DataFrame: Merged DataFrame containing data from all stations
    """
    # Validate input parameters
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Invalid directory path")
    if not isinstance(point_names, list) or not point_names:
        raise ValueError("point_names must be a non-empty list")

    directory = directory.rstrip('/')
    logger.info(f"Processing {len(point_names)} stations in directory: {directory}")
    try:
        # Get file paths for each station
        file_paths = {}
        for station in point_names:
            station_dir = f"{directory}/{station}"
            if not Path(station_dir).exists():
                logger.warning(f"Directory not found for station {station}: {station_dir}")
                continue

            pos_files = [f for f in get_files_from_dir(station_dir, False)
                         if f.endswith('.pos')]
            if not pos_files:
                logger.warning(f"No .pos files found for station {station}")
                continue

            file_paths[station] = pos_files
            logger.info(f"Found {len(pos_files)} .pos file(s) for station {station}")

        # Process all files and collect resulting DataFrames
        dfs = []
        for station, files in file_paths.items():
            for file in files:
                df = process_file(
                    file_path=file,
                    station=station,
                    fixed_solution_only=fixed_solution_only,
                    resample_interval=resample_interval,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None:
                    dfs.append(df)

        if not dfs:
            logger.error("No valid data found in any of the input files")
            return pd.DataFrame()

        # Merge all DataFrames
        # merged_df = reduce(
        #     lambda left, right: pd.merge(left, right, on='Date', how='outer'),
        #     dfs
        # )
        # Another way to merge:
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

        # Drop rows with NaN values if requested
        if dropna:
            merged_df = merged_df.dropna()

        # Add zero epoch coordinates if provided
        if zero_epoch_coords:
            zero_epoch_data = {'Date': [ZERO_EPOCH_DATE]}
            for station in point_names:
                if station in zero_epoch_coords:
                    coords = zero_epoch_coords[station]
                    zero_epoch_data.update({
                        f'x_{station}': [coords[0]],
                        f'y_{station}': [coords[1]],
                        f'z_{station}': [coords[2]]
                    })
            zero_epoch_df = pd.DataFrame(zero_epoch_data)
            merged_df = pd.concat([zero_epoch_df, merged_df], ignore_index=True)

        logger.info("Successfully merged .pos files")
        return merged_df

    except Exception as e:
        logger.error(f"Error in makefile: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        # Example usage
        logger.info("Starting GNSS position file processing")

        point_names = ["SNSK00RUS", "SNSK01RUS", "SNSK02RUS", "SNSK03RUS", "BUZZ"]

        merged_data = makefile(
            point_names=point_names,
            zero_epoch_coords=None,
            dropna=False,
            directory='2024-08-30',
            resample_interval=None,
            fixed_solution_only=False
        )

        # Ensure output directory exists
        output_dir = Path('Data/input_files')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the merged data
        output_file = output_dir / '2024-08-30.csv'
        merged_data.to_csv(output_file, sep=';', index=False)

        logger.info(f'Successfully saved merged data to {output_file}')

        # Log some statistics about the merged data
        logger.info(f'Merged data shape: {merged_data.shape}')
        logger.info(f'Date range: {merged_data["Date"].min()} to {merged_data["Date"].max()}')

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
