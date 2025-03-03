"""
Core functionality package initialization.
"""
import sys
from pathlib import Path

# Add project root to sys.path for consistent imports
root_path = Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from .tests import (
    perform_ttest,
    perform_chi2_test,
    geometric_chi_test_calc,
    find_offsets,
    calculate_differences,
    extract_Qv_matrices,
    locate_offset_stations,
    drop_station_columns
)

from .tools import (
    calculate_differences,
    extract_Qv_matrices,
    drop_station_columns,
    filter_data,
    interpolate_missing_values
)

from .report_generator import ReportGenerator

__all__ = [
    'perform_ttest',
    'perform_chi2_test',
    'geometric_chi_test_calc',
    'find_offsets',
    'calculate_differences',
    'extract_Qv_matrices',
    'locate_offset_stations',
    'drop_station_columns',
    'ReportGenerator',
    'calculate_differences',
    'extract_Qv_matrices',
    'drop_station_columns',
    'filter_data',
    'interpolate_missing_values'
]
