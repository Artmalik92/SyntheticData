"""
Tools for processing data
"""

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.signal import medfilt
from typing import List, Tuple, Optional, Union, Dict


def calculate_differences(df: pd.DataFrame,
                          start_date: pd.Timestamp,
                          end_date: pd.Timestamp) -> Optional[List[float]]:
    """
    Calculates coordinate differences between two dates.
    """
    coord_cols = df.columns[df.columns.str.match(r'[xyz]_.*')].tolist()
    row_0 = df[df['Date'] == start_date]
    row_i = df[df['Date'] == end_date]

    if row_0.empty or row_i.empty:
        return None

    return (row_0[coord_cols].iloc[0] - row_i[coord_cols].iloc[0]).tolist()


def extract_Qv_matrices(Qv: pd.DataFrame,
                        start_date: pd.Timestamp,
                        end_date: pd.Timestamp) -> Optional[np.ndarray]:
    """
    Extracts and combines Qv matrices for two dates.
    """
    Qv_0 = Qv[Qv['Date'] == start_date]
    Qv_i = Qv[Qv['Date'] == end_date]

    if Qv_0.empty or Qv_i.empty:
        return None

    matrices_0 = []
    matrices_i = []

    for col in Qv_0.columns:
        if col != 'Date':
            matrices_0.extend(Qv_0[col].tolist())
            matrices_i.extend(Qv_i[col].tolist())

    Qv_0_block = block_diag(*matrices_0)
    Qv_i_block = block_diag(*matrices_i)

    return Qv_0_block + Qv_i_block


def drop_station_columns(df: pd.DataFrame, stations: List[str]) -> pd.DataFrame:
    """
    Removes columns corresponding to specified stations from DataFrame.
    """
    station_cols = [col for col in df.columns if any(station in col for station in stations)]
    return df.drop(station_cols, axis=1)


def filter_data(df: pd.DataFrame, kernel_size: int) -> pd.DataFrame:
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


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates missing values in the DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing time series data.

    Returns:
        DataFrame: The DataFrame with interpolated missing values.
    """
    df_interpolated = df.interpolate(method='linear', limit_direction='both')
    return df_interpolated