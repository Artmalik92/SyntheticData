"""
Statistical Tests Module
Contains core statistical test functions for GNSS time series analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, chi2
from typing import List, Tuple
from config.logger_config import get_logger
from .tools import (
    calculate_differences,
    extract_Qv_matrices,
    drop_station_columns
)

logger = get_logger('log')


def perform_ttest(raz_list: List[float],
                  threshold: float = 0.95) -> Tuple[bool, float]:
    """
    Performs a T-test on the given list of differences.

    Args:
        raz_list: List of differences to test
        threshold: Significance threshold (default: 0.95)

    Returns:
        tuple: (rejected: bool, p_value: float)
    """
    pvalue = ttest_1samp(a=raz_list, popmean=0, nan_policy='omit')[1]
    return pvalue <= threshold, pvalue


def perform_chi2_test(raz_list: List[float],
                      sigma_0: float,
                      Qv: np.ndarray,
                      threshold: float = 0.95,
                      Qdd_status: str = '0',
                      m_coef: float = 1.0) -> Tuple[bool, float, float]:
    """
    Performs a Chi-square test on the given list of differences.

    Args:
        raz_list: List of differences
        sigma_0: The sigma-0 value for the test
        Qv: The variance matrix of the residuals
        threshold: Significance threshold (default: 0.95)
        Qdd_status: Set Qdd-matrix status ('1' for identity, '0' for covariance)
        m_coef: Scale factor for the Chi-test statistic

    Returns:
        tuple: (rejected: bool, K_value: float, test_value: float)
    """
    # Convert input to numpy array
    d = np.array(raz_list)

    # Determine Qdd matrix based on status
    if Qdd_status == '1':
        Qdd = np.eye(Qv.shape[0])
    else:
        Qdd = Qv

    # Calculate test statistic
    K = (d.dot(np.linalg.inv(Qdd)).dot(d.transpose()) / sigma_0) * m_coef

    # Calculate critical value
    test_value = chi2.ppf(df=((d.shape[0]) / 3) * 6, q=threshold)

    return K > test_value, K, test_value


def geometric_chi_test_calc(time_series_frag: np.ndarray,
                            sigma: np.ndarray,
                            covariances: np.ndarray,
                            Q_status: str = '0') -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Performs geometric chi-test calculations on time series data.

    Args:
        time_series_frag: Time series fragment data
        sigma: Sigma values for the coordinates
        covariances: Covariance values
        Q_status: Q-matrix status ('1' for identity, '0' for covariance)

    Returns:
        tuple: (x_LS_first, x_LS, Qv, mu, Qx)
    """
    # Convert inputs to DataFrames with consistent structure
    time_series_df = pd.DataFrame(time_series_frag, columns=[0, 1, 2])
    sigma_df = pd.DataFrame(sigma, columns=[0, 1, 2])
    covariances_df = pd.DataFrame(covariances, columns=[0, 1, 2])

    # Initialize arrays
    L = np.zeros((time_series_df.shape[0] * 3))
    t = np.arange(0, time_series_df.shape[0], 1)

    # Fill L array with coordinates
    for i in range(time_series_df.shape[0]):
        L[3 * i:3 * i + 3] = time_series_df.iloc[i].values
    """ Old version
    for i in range(time_series_df.shape[0]):
        for j in range(3):
            L[3 * i] = time_series_df[0][i]
            L[3 * i + 1] = time_series_df[1][i]
            L[3 * i + 2] = time_series_df[2][i]"""

    # # Create coefficient matrix A
    A = np.vstack([np.hstack((np.identity(3) * ti, np.identity(3)))
                   for ti in t])
    """ Old version
    for m in range(time_series_df.shape[0]):
        ti = t[m]
        if m == 0:
            A = np.hstack((np.identity(3) * ti, np.identity(3)))
        else:
            Aux = np.hstack((np.identity(3) * ti, np.identity(3)))
            A = np.vstack((A, Aux))"""

    # Create Q matrix
    Q_size = time_series_df.shape[0] * 3
    Q = np.zeros((Q_size, Q_size))

    # Fill Q matrix based on status
    for i in range(time_series_df.shape[0]):
        row_start = i * 3
        if Q_status == '0':
            sde, sdn, sdu = sigma_df.iloc[i]
            sden, sdnu, sdue = covariances_df.iloc[i]
            Q[row_start:row_start + 3, row_start:row_start + 3] = np.array([
                [sde**2, sden**2, sdue**2],
                [sden**2, sdn**2, sdnu**2],
                [sdue**2, sdnu**2, sdu**2]
            ])
        else:  # Q_status == '1'
            Q[row_start:row_start + 3, row_start:row_start + 3] = np.eye(3)

    # Calculate weight matrix and normal equation matrix
    P = Q/((0.005*3)**2)  # 0.02 is the accuracy of the device (or (0.005*3) for our receiver)
    N = A.T @ np.linalg.inv(P) @ A

    # Parameters vector
    X = np.linalg.inv(N) @ (A.T @ np.linalg.inv(P) @ L)

    # Calculate final coordinates
    x_LS = np.array([
        X[0] * t[-1] + X[3],
        X[1] * t[-1] + X[4],
        X[2] * t[-1] + X[5]
    ])
    # x_LS_first = np.array([X[0] + X[1] * t[0], X[2]])
    x_LS_first = X[0] + X[1] * t[0]

    # Calculate residuals and standard deviation
    V = A @ X - L
    mu = np.sqrt(np.sum(V.T @ np.linalg.inv(P) @ V) / (V.shape[0] - 6))

    # Calculate covariance matrices
    Qx = np.linalg.inv(N)
    C = np.array([
        [t[-1], 0, 0, 1, 0, 0],
        [0, t[-1], 0, 0, 1, 0],
        [0, 0, t[-1], 0, 0, 1]
    ])
    Qv = C @ Qx @ C.T

    return x_LS_first, x_LS, Qv, mu, Qx


def find_offsets(df: pd.DataFrame,
                 sigma_0: pd.DataFrame,
                 Qv: pd.DataFrame,
                 max_drop: int = 1,
                 Qdd_status: str = '0',
                 m_coef: float = 1.0,
                 threshold: float = 0.95) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Finds offset points in the time series data.

    The function implements a statistical test to detect significant changes
    in coordinate time series between consecutive epochs. It uses both the
    coordinate differences and their associated uncertainties (through Qv matrices)
    to perform rigorous statistical testing.

    Args:
        df: Input DataFrame with time series data
        sigma_0: DataFrame with sigma values
        Qv: Variance matrix of residuals
        max_drop: Maximum number of stations to drop
        Qdd_status: Qdd-matrix status
        m_coef: Scale factor for Chi-test
        threshold: Significance threshold

    Returns:
        tuple: (offset_points, rejected_dates)
    """
    # Get unique dates and station names
    dates = sorted(df['Date'].unique())
    station_names = list(set(df.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

    offset_points = []
    rejected_dates = []

    # Calculate reference variance for each epoch using sum of squares
    sigma_cols = [col for col in sigma_0.columns if col.startswith('sigma0_')]
    if not sigma_cols:
        logger.error("No sigma0 columns found in input data")
        return [], []

    # Calculate sum of squares for each row (matching legacy computation)
    sum_of_squares = sigma_0[sigma_cols].apply(lambda row: sum(x ** 2 for x in row), axis=1)
    logger.debug(f"Computed sum of squares range: {sum_of_squares.min():.6f} to {sum_of_squares.max():.6f}")

    mu_mean_df = pd.DataFrame({'Date': sigma_0['Date'], 'Sum_of_Squares': sum_of_squares})

    # Process each consecutive pair of dates
    for i in range(len(dates) - 1):
        start_date, end_date = dates[i], dates[i + 1]

        logger.info(f'Calculate Chi-2 for dates: {start_date} to {end_date}')

        # Calculate differences between dates
        raz_list = calculate_differences(df, start_date, end_date)
        if raz_list is None:
            continue

        # Get Qv matrices for both dates
        Qv_matrices = extract_Qv_matrices(Qv, start_date, end_date)
        if Qv_matrices is None:
            continue

        # Get sigma0 for the epoch with enhanced validation
        sigma0_values = mu_mean_df[mu_mean_df['Date'] == start_date]['Sum_of_Squares']
        if sigma0_values.empty:
            logger.warning(f"No sigma0 value found for date {start_date}")
            continue

        current_sigma0 = float(sigma0_values.iloc[0])

        # Validate sigma0 value
        min_sigma0 = 1e-6
        if current_sigma0 < min_sigma0:
            logger.warning(
                f"Unusually small sigma0 ({current_sigma0:.2e}) found for date {start_date}. Using minimum value.")
            current_sigma0 = min_sigma0

        # Perform Chi-square test with validated sigma0
        chi2_result = perform_chi2_test(
            raz_list=raz_list,
            sigma_0=current_sigma0,
            Qv=Qv_matrices,
            threshold=threshold,
            Qdd_status=Qdd_status,
            m_coef=m_coef
        )

        if chi2_result[0]:  # If test rejected null hypothesis
            rejected_dates.append((start_date, end_date))

            logger.info(f"<span style='color:red'>Null hypothesis rejected, "
                        f"testvalue = {round(chi2_result[2], 3)}, K = {round(chi2_result[1], 3)}</span>")

            # Try dropping stations to find which ones caused the offset
            offset_found = locate_offset_stations(
                df, station_names, start_date, end_date,
                max_drop, Qv, sigma_0, Qdd_status, m_coef
            )

            if offset_found:
                offset_points.extend(offset_found)
        else:
            logger.info(f"Null hypothesis not rejected, "
                        f"testvalue = {round(chi2_result[2], 3)}, K = {round(chi2_result[1], 3)}")
    return offset_points, rejected_dates


def locate_offset_stations(df: pd.DataFrame,
                           station_names: List[str],
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp,
                           max_drop: int,
                           Qv: pd.DataFrame,
                           sigma_0: pd.DataFrame,
                           Qdd_status: str,
                           m_coef: float) -> List[Tuple]:
    """
    Identifies stations that may have caused an offset.
    """
    from itertools import combinations

    offset_points = []
    date_range_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    logger.info(f'Attempting to define offset stations for dates {start_date} to {end_date}')

    for drop_count in range(1, min(max_drop + 1, len(station_names) + 1)):
        for station_combination in combinations(station_names, drop_count):
            logger.info(f'Testing stations {station_combination}')

            # Drop selected stations and recalculate
            non_station_df = drop_station_columns(date_range_df, station_combination)
            non_station_qv = drop_station_columns(Qv, station_combination)
            non_station_sigma = drop_station_columns(sigma_0, station_combination)

            # Recalculate reference variance using sum of squares
            sigma_cols = [col for col in non_station_sigma.columns if col.startswith('sigma0_')]
            if not sigma_cols:
                logger.warning(f"No sigma0 columns found after dropping stations {station_combination}")
                continue

            # Use same sum of squares computation as main function
            sum_squares = non_station_sigma[sigma_cols].apply(lambda row: sum(x ** 2 for x in row), axis=1)
            mu_df = pd.DataFrame({'Date': non_station_sigma['Date'], 'Sum_of_Squares': sum_squares})

            # Calculate differences and perform test
            raz_list = calculate_differences(non_station_df, start_date, end_date)
            if raz_list is None:
                continue

            Qv_matrices = extract_Qv_matrices(non_station_qv, start_date, end_date)
            if Qv_matrices is None:
                continue

            # Get sigma0 for the epoch with validation
            sigma0_values = mu_df[mu_df['Date'] == start_date]['Sum_of_Squares']
            if sigma0_values.empty:
                logger.warning(f"No sigma0 value found for date {start_date} in offset station analysis")
                continue

            current_sigma0 = float(sigma0_values.iloc[0])
            # if current_sigma0 < min_sigma0:
            #     logger.warning(f"Small sigma0 ({current_sigma0:.2e}) in offset analysis for date {start_date}")
            #     current_sigma0 = min_sigma0

            chi2_result = perform_chi2_test(
                raz_list=raz_list,
                sigma_0=current_sigma0,
                Qv=Qv_matrices,
                Qdd_status=Qdd_status,
                m_coef=m_coef
            )

            if not chi2_result[0]:  # If test passes after dropping stations

                logger.info(f"Null hypothesis not rejected, "
                            f"testvalue = {round(chi2_result[2], 3)}, K = {round(chi2_result[1], 3)}")

                logger.info(f"<span style='color:green'>Stations {station_combination} exhibit an offset "
                            f"for dates {start_date} to {end_date}</span>")

                # Calculate offset size
                station_cols = [col for col in date_range_df.columns
                                if any(station in col for station in station_combination)]
                start_vals = date_range_df[station_cols].loc[date_range_df['Date'] == start_date]
                end_vals = date_range_df[station_cols].loc[date_range_df['Date'] == end_date]
                offset_size = np.sqrt(np.sum((start_vals.values - end_vals.values) ** 2))

                for station in station_combination:
                    offset_points.append((start_date, end_date, station, offset_size))
                return offset_points
            else:
                logger.info(f"<span style='color:red'>Null hypothesis rejected, "
                            f"testvalue = {round(chi2_result[2], 3)}, K = {round(chi2_result[1], 3)}</span>")
    if not offset_points:
        logger.info(f"<span style='color:red; font-weight: bold'>failed to locate the offset stations "
                    f"for dates {start_date} to {end_date}</span>")
    return offset_points



