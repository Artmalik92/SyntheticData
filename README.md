# Congruence-Test

A Python-based tool for processing GNSS position files and performing congruence testing on GNSS networks.

## Project Structure

```
├── config/
│   ├── logger_config.py     # Logging configuration
│   ├── config_loader.py     # Configuration loading and validation
│   └── settings.yaml        # Main configuration file
├── core/
│   ├── tests.py             # Core statistical test functions for GNSS time series analysis
│   └── report_generator.py  # HTML report generation with visualizations
├── merge_pos_files.py       # .pos file merging functionality
├── process_pos.py           # Main processing script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Features

- GNSS Position File Processing:
  - Merge multiple .pos files
  - Resample files by timestamps
  - Handle fixed/float solutions
  - Support for multiple GNSS stations

- Congruence Testing:
  - Calculate statistics using Chi-square test
  - Offset stations detection

- HTML Reports:
  - Interactive visualizations using Plotly
  - Detailed statistical summaries

## Requirements

- Python 3.7+
- Required packages (install via requirements.txt):
  ```
  pandas
  numpy
  scipy
  plotly
  jinja2
  pyyaml
  moncenterlib
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Artmalik92/Congruence-Test.git
   cd Congruence-Test
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The tool uses a YAML configuration file (`config/settings.yaml`) for all settings:

```yaml
paths:
  input_directory: "directory"     # Directory containing .pos files
  output_directory: "Data"         # Base output directory
  merged_data: "Data"              # Directory for merged .csv file
  reports: "Data/reports"          # Directory for reports

stations:
  point_names:                     # List of GNSS stations to process
    - "STATION1"
    - "STATION2"
    # Add more stations as needed

processing:
  merge:
    resample_interval: null        # Optional resampling interval
    dropna: false                  # Whether to drop NaN values
    fixed_solution_only: false     # Use only fixed solutions

  congruency:
    med_filter: true               # Whether to apply median filter
    filter_kernel: 11              # Kernel size for median filter
    use_wls: false                 # Whether to use Weighted Least Squares
    window_size: "1h"              # Window size for WLS (if used)
    Q_status: "0"                  # Matrix type (0=covariance, 1=identity)
    Qdd_status: "0"                # Matrix type (0=covariance, 1=identity)
    m_coef: 1.0                    # Scale coefficient for Chi-test
    max_drop: 2                    # Max stations to drop in offset detection
```

## Usage

1. Configure settings in `config/settings.yaml`
2. Run the processing script:
   ```bash
   python process_pos.py
   ```

The script will:
1. Load configuration from settings.yaml
2. Merge POS files from specified stations
3. Perform congruency testing with configured parameters
4. Generate an interactive HTML report

### Input Data Structure

Place your .pos files in a directory structure like:
```
your-data-directory/
├── STATION1/
│   └── file.pos
├── STATION2/
│   └── file.pos
...
```

### Output

The tool generates:
1. Merged data file (path specified in config)
2. HTML report (path specified in config) containing:
   - Summary statistics
   - Offset point analysis
   - Interactive visualizations
   - Processing logs

## Core Modules

### core/tests.py
Contains implementations of:
- Geometric chi-test calculations
- Offset point detection

### core/report_generator.py
Handles:
- HTML report generation
- Interactive Plotly visualizations
- Bootstrap-based design

### core/tools.py
Provides:
- Tools for data processing

### config/config_loader.py
Provides:
- Configuration loading and validation
- Path resolution
- Default configuration values


## Authors

Artem Malikov a.o.malikov@mail.ru
