"""
Report Generator Module
----------------------
Handles generation of HTML reports with modern styling and interactive visualizations.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template
from datetime import datetime
from pathlib import Path
from config.logger_config import get_logger

logger = get_logger('log')


class ReportGenerator:
    """
    Generates HTML reports with Bootstrap styling and Plotly visualizations.
    """

    BOOTSTRAP_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ title }}</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding-top: 2rem; }
            .plot-container { margin-bottom: 2rem; }
            .log-container { 
                max-height: 400px;
                overflow-y: auto;
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 4px;
            }
            .stats-card {
                margin-bottom: 1rem;
                box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
            }
            .nav-pills .nav-link.active {
                background-color: #0d6efd;
            }
            @media print {
                .log-container { max-height: none; }
                .no-print { display: none; }
            }
            .table th {
                text-align: left; /* Align headers to the left */
            }
            .table td {
                text-align: left; /* Align data cells to the left */
            }
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="row">
                <!-- Sidebar -->
                <div class="col-md-3 col-lg-2 d-md-block bg-light sidebar no-print">
                    <div class="position-sticky pt-3">
                        <nav class="nav flex-column nav-pills">
                            <a class="nav-link active" href="#summary">Summary</a>
                            <a class="nav-link" href="#offsets">Offset Points</a>
                            <a class="nav-link" href="#visualizations">Visualizations</a>
                            <a class="nav-link" href="#logs">Detailed Logs</a>
                        </nav>
                    </div>
                </div>

                <!-- Main content -->
                <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
                    <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
                        <h1>Congruency Test Report</h1>
                        <div class="btn-toolbar mb-2 mb-md-0 no-print">
                            <button onclick="window.print()" class="btn btn-sm btn-outline-secondary">
                                Print Report
                            </button>
                        </div>
                    </div>

                    <!-- Summary Section -->
                    <section id="summary">
                        <h2>Summary</h2>
                        <div class="row">
                            <div class="col-md-6 col-lg-4">
                                <div class="card stats-card">
                                    <div class="card-body">
                                        <h5 class="card-title">Input File</h5>
                                        <p class="card-text">{{ file_name }}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6 col-lg-4">
                                <div class="card stats-card">
                                    <div class="card-body">
                                        <h5 class="card-title">Total Tests</h5>
                                        <p class="card-text">{{ total_tests }}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6 col-lg-4">
                                <div class="card stats-card">
                                    <div class="card-body">
                                        <h5 class="card-title">Stations Processed</h5>
                                        <p class="card-text">{{ stations_length }}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-12">
                                <div class="card stats-card">
                                    <div class="card-body">
                                        <h5 class="card-title">Configuration</h5>
                                        <ul class="list-unstyled">
                                            <li><strong>Window Size:</strong> {{ window_size }}</li>
                                            <li><strong>Stations:</strong> {{ stations_names|join(', ') }}</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                    <!-- Offset Points Section -->
                    <section id="offsets" class="mt-5">
                        <div class="d-flex justify-content-between align-items-center">
                            <h2>Offset Points</h2>
                            {% if has_offsets %}
                            <button onclick="downloadOffsets()" class="btn btn-primary no-print">
                                Download Offsets CSV
                            </button>
                            {% endif %}
                        </div>
                        <div class="table-responsive">
                            {{ offset_points }}
                        </div>
                        {% if has_offsets %}
                        <div class="alert alert-info mt-3">
                            <h5>Offset Analysis Summary:</h5>
                            <ul>
                                <li>Total offset points detected: {{ offset_count }}</li>
                                <li>Affected stations: {{ affected_stations|join(', ') }}</li>
                                <li>Date range: {{ offset_date_range }}</li>
                            </ul>
                        </div>
                        {% endif %}
                    </section>

                    <script>
                    function downloadOffsets() {
                        const csvContent = `{{ offsets_csv|safe }}`;
                        const blob = new Blob([csvContent], { type: 'text/csv' });
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.setAttribute('href', url);
                        a.setAttribute('download', 'offset_points.csv');
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                    }
                    </script>

                    <!-- Visualizations Section -->
                    <section id="visualizations" class="mt-5">
                        <h2>Visualizations</h2>
                        <div class="plot-container">
                            {{ offset_plots }}
                        </div>
                    </section>

                    <!-- Logs Section -->
                    <section id="logs" class="mt-5">
                        <h2>Detailed Logs</h2>
                        <div class="log-container" style="max-height: 800px">
                            <pre>{{ log_contents }}</pre>
                        </div>
                    </section>
                </main>
            </div>
        </div>

        <!-- Bootstrap Bundle with Popper -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

    def __init__(self, output_dir: str):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory where reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_visualization(self,
                             station: str,
                             dates: pd.Series,
                             raw_data: Dict[str, pd.DataFrame],
                             filtered_data: Dict[str, pd.DataFrame],
                             wls_data: Dict[str, pd.DataFrame],
                             rejected_dates: List[Tuple],
                             offsets: List[Tuple]) -> go.Figure:
        """
        Creates a Plotly figure for a station's data visualization.
        """
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.02)

        # Add traces for each coordinate (x, y, z)
        for i, coord in enumerate(['x', 'y', 'z']):
            # Raw data
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=raw_data[f'{coord}_{station}'],
                    mode='lines',
                    name='Raw data' if i == 0 else None,
                    line=dict(color='lightgray'),
                    legendgroup='Raw data',
                    showlegend=i == 0
                ),
                row=i + 1, col=1
            )

            # Filtered data
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=filtered_data[f'{coord}_{station}'],
                    mode='lines',
                    name='Filtered data' if i == 0 else None,
                    line=dict(color='blue'),
                    legendgroup='Filtered data',
                    showlegend=i == 0
                ),
                row=i + 1, col=1
            )

            # WLS data
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=wls_data[f'{coord}_{station}'],
                    mode='lines',
                    name='WLS Estimate' if i == 0 else None,
                    line=dict(color='red'),
                    legendgroup='WLS Estimate',
                    showlegend=i == 0
                ),
                row=i + 1, col=1
            )

        # Add rejected dates highlighting
        for start_date, end_date in rejected_dates:
            for i in range(3):
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date,
                    fillcolor='red',
                    opacity=0.2,
                    layer='below',
                    line_width=0,
                    row=i + 1,
                    col=1
                )

        # Add offset highlighting
        station_offsets = [
            (start, end) for start, end, s, _ in offsets if s == station
        ]
        for start_date, end_date in station_offsets:
            for i in range(3):
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date,
                    fillcolor='green',
                    opacity=0.5,
                    layer='above',
                    line_width=0,
                    row=i + 1,
                    col=1
                )

        # Update layout
        fig.update_layout(
            height=600,
            width=1200,
            title_text=f'Offsets found in {station} station',
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True
        )

        # Update axes
        for i in range(3):
            fig.update_xaxes(showgrid=False, row=i + 1, col=1)
            fig.update_yaxes(showgrid=False, row=i + 1, col=1)
            if i < 2:
                fig.update_xaxes(tickvals=[], row=i + 1, col=1)
            else:
                fig.update_xaxes(tickformat='%H:%M:%S', row=i + 1, col=1)

        return fig

    def generate_report(self,
                        report_data: Dict[str, Any],
                        wls_df: pd.DataFrame,
                        raw_df: pd.DataFrame,
                        filtered_df: pd.DataFrame,
                        offset_points: List[Tuple],
                        rejected_dates: List[Tuple],
                        log_contents: str = "") -> str:
        """
        Generates an HTML report with visualizations and data analysis.

        Args:
            report_data: Dictionary containing report metadata
            wls_df: DataFrame with WLS estimates
            raw_df: DataFrame with raw data
            filtered_df: DataFrame with filtered data
            offset_points: List of detected offset points
            rejected_dates: List of rejected date ranges

        Returns:
            str: Path to the generated report
        """
        try:
            # Process offset points
            has_offsets = bool(offset_points)
            if has_offsets:
                offsets_df = pd.DataFrame(
                    offset_points,
                    columns=['Start_date', 'End_date', 'Station', 'Offset size']
                )

                # Create HTML table with bootstrap classes
                offsets_table = offsets_df.to_html(
                    classes='table table-striped table-hover',
                    index=False,
                    float_format=lambda x: '{:.6f}'.format(x) if isinstance(x, float) else x
                )

                # Prepare CSV content for download
                offsets_csv = offsets_df.to_csv(sep=';', index=False)

                # Calculate summary statistics
                offset_count = len(offset_points)
                affected_stations = sorted(set(offsets_df['Station']))
                offset_date_range = f"{min(offsets_df['Start_date'])} to {max(offsets_df['End_date'])}"
            else:
                offsets_table = '<div class="alert alert-info">No offset points were detected in the analysis.</div>'
                offsets_csv = ''
                offset_count = 0
                affected_stations = []
                offset_date_range = 'N/A'

            # Get unique station names
            stations = list(set(wls_df.columns[1:].str.extract('_(.*)').iloc[:, 0].tolist()))

            # Generate visualizations for each station
            plots_html = ""
            for station in stations:
                fig = self.create_visualization(
                    station=station,
                    dates=wls_df['Date'],
                    raw_data=raw_df,
                    filtered_data=filtered_df,
                    wls_data=wls_df,
                    rejected_dates=rejected_dates,
                    offsets=offset_points
                )
                plots_html += fig.to_html(full_html=False, include_plotlyjs=True)
                plots_html += "<br>"

            # Prepare template data with enhanced offset information
            template_data = {
                **report_data,
                'offset_points': offsets_table,
                'log_contents': log_contents if log_contents else "No logs captured during processing.",
                'offset_plots': plots_html,
                'title': 'Congruency Test Report',
                'has_offsets': has_offsets,
                'offsets_csv': offsets_csv,
                'offset_count': offset_count,
                'affected_stations': affected_stations,
                'offset_date_range': offset_date_range
            }

            # Generate HTML
            template = Template(self.BOOTSTRAP_TEMPLATE)
            html_content = template.render(**template_data)

            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'report_{timestamp}.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f'Report generated successfully: {output_path}')
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
