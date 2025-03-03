"""
Main script for Congruence Test.
See configuration file in config/settings.yaml before using this script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from merge_pos_files import makefile
from core.tests import (
    geometric_chi_test_calc,
    find_offsets,
    perform_chi2_test
)
from core.report_generator import ReportGenerator
from core.tools import (
    interpolate_missing_values,
    filter_data
)
from config.config_loader import load_config, ConfigurationError
from config.logger_config import setup_logger

# Initialize logger
logger, logger_capture = setup_logger('log', capture_string=True)


def process_gnss_data(config_path: str = "config/settings.yaml") -> None:
    """
    Process GNSS data according to configuration settings.

    Args:
        config_path: Path to the configuration file
    """
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        # Extract configuration values
        paths = config['paths']
        stations = config['stations']
        processing = config['processing']

        # Step 1: Merge POS files
        logger.info("Merging POS files...")
        merged_data = makefile(
            directory=paths['input_directory'],
            point_names=stations['point_names'],
            resample_interval=processing['merge']['resample_interval'],
            zero_epoch_coords=None,
            dropna=processing['merge']['dropna'],
            fixed_solution_only=processing['merge']['fixed_solution_only']
        )

        # Save merged data
        merged_path = Path(paths['merged_data'])
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(merged_path, sep=';', index=False)
        logger.info(f"Merged data saved to: {merged_path}")

        # Step 2: Run congruency test
        logger.info("Running congruency test...")

        # Since we're not using WLS, use the merged data directly
        raw = merged_data.copy()
        filtered = merged_data.copy()
        wls = merged_data.copy()

        # Get station names
        station_list = stations['point_names']

        # Process each station to get Qv matrices and MU values
        Qv_data = {'Date': wls['Date']}
        MU_data = {'Date': wls['Date']}

        for station in station_list:
            # Get station-specific columns
            coord_cols = [col for col in wls.columns if col.startswith(('x_', 'y_', 'z_')) and station in col]
            sigma_cols = [f'sde_{station}', f'sdn_{station}', f'sdu_{station}']
            covar_cols = [f'sden_{station}', f'sdnu_{station}', f'sdue_{station}']
            mu_col = f'sigma0_{station}'

            # Calculate geometric parameters for the station
            _, _, Qv, mu, _ = geometric_chi_test_calc(
                time_series_frag=wls[coord_cols].values,
                sigma=wls[sigma_cols].values,
                covariances=wls[covar_cols].values,
                mu=wls[mu_col].values,
                Q_status=processing['congruency']['Q_status']
            )

            Qv_data[f'Qv_{station}'] = [Qv.tolist()] * len(wls)
            MU_data[f'sigma0_{station}'] = [mu] * len(wls)

        # Median filter
        if processing['congruency']['med_filter']:
            wls = filter_data(wls, kernel_size=processing['congruency']['filter_kernel'])
            filtered = wls.copy()

        # interpolate missing values
        if processing['congruency']['interpolate_missing']:
            wls = interpolate_missing_values(wls)

        # Convert to DataFrames
        Qv_df = pd.DataFrame(Qv_data)
        MU_df = pd.DataFrame(MU_data)

        # Find offset points
        logger.info("Analyzing time series for offset points...")
        offset_points, rejected_dates = find_offsets(
            df=wls,
            sigma_0=MU_df,
            Qv=Qv_df,
            max_drop=processing['congruency']['max_drop'],
            Qdd_status=processing['congruency']['Qdd_status'],
            m_coef=processing['congruency']['m_coef']
        )

        if not rejected_dates:
            logger.info("No rejected date ranges found in the congruency test")
        else:
            logger.info(f"Found {len(rejected_dates)} rejected date ranges")

        if not offset_points:
            logger.warning("No offsets were detected in the time series")
        else:
            logger.info(f"Found {len(offset_points)} offsets")

        # Generate report
        report_data = {
            'file_name': str(merged_path),
            'total_tests': len(wls['Date']) - 1,
            'stations_length': len(station_list),
            'stations_names': station_list,
            'window_size': ('N/A (No WLS)' if not processing['congruency']['use_wls']
                            else processing['congruency']['window_size'])
        }

        # Get captured logs for report
        log_contents = logger_capture.stream.getvalue() if logger_capture else ""

        # Initialize report generator
        report_generator = ReportGenerator(output_dir=paths['reports'])

        # Generate report with logs
        report_path = report_generator.generate_report(
            report_data=report_data,
            wls_df=wls,
            raw_df=raw,
            filtered_df=filtered,
            offset_points=offset_points,
            rejected_dates=rejected_dates,
            log_contents=log_contents
        )

        logger.info("Processing completed successfully")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing GNSS data: {str(e)}")
        raise


if __name__ == "__main__":
    process_gnss_data()
