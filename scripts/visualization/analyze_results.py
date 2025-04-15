#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from mpl_toolkits.mplot3d import Axes3D
import argparse
import logging

# --- Custom Exceptions ---
class DataParsingError(Exception):
    """Custom exception for errors during data parsing."""
    pass

class PlottingError(Exception):
    """Custom exception for errors during plot generation."""
    pass
# --- End Custom Exceptions ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# --- Argument Parsing ---

def parse_args():
    """Parses command-line arguments for the analysis script.

    Uses argparse to define and parse arguments for the job directory,
    analysis mode, 3D plot skipping, and verbosity.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze NUMA benchmark results (Single or Multiple Sizes)')
    parser.add_argument('job_dir', type=Path, help='Directory containing benchmark results')
    parser.add_argument('--mode', choices=['single', 'multiple'], default=None,
                      help='Analysis mode: single or multiple memory sizes (default: auto-detect)')
    parser.add_argument('--no-3d', action='store_true',
                      help='Skip all 3D visualizations in multiple size mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose (DEBUG) logging')
    args = parser.parse_args()
    return args

# --- Data Parsing and Mode Detection ---

def detect_data_mode(csv_files, verbose=False):
    """Detects whether the input data represents single or multiple memory sizes.

    Examines the first valid CSV file found to determine the mode.
    Checks for the presence of a 'size (MB)' column and more than one row.
    Handles empty files and defaults to 'single' mode if detection fails.

    Args:
        csv_files (list[Path]): A list of Path objects for the CSV files to check.
        verbose (bool, optional): If True, enables debug logging during detection.
                                   Defaults to False.

    Returns:
        str: The detected mode ('single' or 'multiple').
    """
    if not csv_files:
        log.debug("No CSV files found, defaulting to single size mode.")
        return 'single'
    
    try:
        df = pd.read_csv(csv_files[0])
        if 'size (MB)' in df.columns and len(df) > 1:
            if verbose: log.debug(f"Detected multiple memory sizes in {csv_files[0].name}")
            return 'multiple'
        else:
            if verbose: log.debug(f"File {csv_files[0].name} appears to be single-size data.")
            return 'single'
    except pd.errors.EmptyDataError:
        log.warning(f"File {csv_files[0].name} is empty, cannot determine mode accurately. Defaulting to single.")
        return 'single'
    except Exception as e:
        log.warning(f"Could not accurately determine data mode from {csv_files[0].name}: {e}. Defaulting to single.")
        return 'single'

def parse_csv_files(job_dir, mode=None, verbose=False):
    """Parses all relevant CSV files in the job directory.

    Identifies CSV files starting with 'sequential_' or 'interleaved_',
    extracts the number of ranks from the filename, and parses latency data
    based on the specified or auto-detected analysis mode ('single' or 'multiple').

    Calculates basic statistics (mean, std, min, max) for each run (single mode)
    or for each memory size within each run (multiple mode).

    Args:
        job_dir (Path): The path to the directory containing the CSV result files.
        mode (str | None, optional): The analysis mode ('single' or 'multiple').
                                     If None, mode is auto-detected. Defaults to None.
        verbose (bool, optional): If True, enables debug logging during parsing.
                                   Defaults to False.

    Returns:
        tuple[list[dict], str]: A tuple containing:
            - A list of dictionaries, where each dictionary represents data from
              one CSV file, including ranks, type, source file, and latency data
              (either as a single 'latencies' array or nested in 'data_by_size').
            - The determined analysis mode string ('single' or 'multiple').

    Raises:
        DataParsingError: If the job directory is invalid, no valid CSV files
                          are found, or critical errors occur during parsing
                          (e.g., invalid format, missing columns, data conversion issues).
    """
    log.info(f"Starting to parse CSV files in {job_dir}")
    data = []
    
    # Validate job_dir earlier
    if not job_dir.exists() or not job_dir.is_dir():
        # Raise specific error instead of exiting directly
        raise DataParsingError(f"Error: Directory {job_dir} does not exist or is not a directory")
    
    csv_files = sorted([f for f in job_dir.glob("*.csv") if f.stem.startswith(('sequential_', 'interleaved_'))])
    log.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        # Raise specific error
        raise DataParsingError(f"Error: No valid CSV files (sequential_*.csv or interleaved_*.csv) found in {job_dir}")
    
    # Auto-detect mode if not specified by user
    if mode is None:
        mode = detect_data_mode(csv_files, verbose=verbose)
        log.info(f"Auto-detected analysis mode: {mode}")
    else:
        log.info(f"Using specified analysis mode: {mode}")
    
    # Parse each file according to the determined mode
    for csv_file in csv_files:
        log.debug(f"Processing file: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                log.warning(f"Skipping empty file: {csv_file.name}"); continue
            if len(df.columns) < 2: # Need at least rank 0 data
                # Raise specific error
                raise DataParsingError(f"Invalid CSV format in {csv_file.name}: expected >= 2 cols, found {len(df.columns)}")

            # Extract number of ranks from filename
            match = re.search(r'(\d+)ranks', csv_file.stem)
            if not match:
                # Raise specific error
                raise DataParsingError(f"Could not find number of ranks (e.g., '16ranks') in filename {csv_file.name}")
            ranks = int(match.group(1)) # Assume int conversion is safe after regex match
            log.debug(f"Found {ranks} ranks in filename")

            file_data = {'ranks': ranks, 'is_interleaved': 'interleaved' in csv_file.stem, 'source_file': csv_file.name}

            if mode == 'single':
                # Single size mode - use first row of data
                if 'size (MB)' in df.columns and len(df) > 1:
                    log.warning(f"File {csv_file.name} contains multiple sizes, using first row only in single-size mode.")

                # Determine latency columns (all except potentially 'size (MB)')
                latency_cols = [col for col in df.columns if col != 'size (MB)']
                if not latency_cols:
                    # Raise specific error
                    raise DataParsingError(f"No latency data columns found in {csv_file.name} (single mode)")

                try:
                    # Attempt to convert the first row's latency columns to float
                    latencies = df.iloc[0][latency_cols].values.astype(float)
                    # Check for NaNs after conversion
                    if np.isnan(latencies).any():
                        log.warning(f"NaN values found in latency data for {csv_file.name} (single mode). Check input CSV.")
                except ValueError as ve:
                     # Raise specific error with context
                     raise DataParsingError(f"Error converting single-size latency data to float in {csv_file.name}: {ve}. Problematic data: {df.iloc[0][latency_cols].values}")

                # Calculate stats, handle potential NaNs
                mean_val = np.nanmean(latencies) if not np.isnan(latencies).all() else np.nan
                std_val = np.nanstd(latencies) if not np.isnan(latencies).all() else np.nan
                min_val = np.nanmin(latencies) if not np.isnan(latencies).all() else np.nan
                max_val = np.nanmax(latencies) if not np.isnan(latencies).all() else np.nan

                file_data.update({
                    'latencies': latencies, 'mean': mean_val, 'std': std_val,
                    'min': min_val, 'max': max_val
                })
                data.append(file_data)

            else: # mode == 'multiple'
                # Check for required 'size (MB)' column in multi-size mode
                if 'size (MB)' not in df.columns:
                    log.warning(f"Skipping file {csv_file.name}: missing 'size (MB)' column required for multi-size mode."); continue

                memory_sizes = df['size (MB)'].values
                data_by_size = {}
                # Determine latency columns (all except 'size (MB)')
                latency_cols = [col for col in df.columns if col != 'size (MB)']
                if not latency_cols:
                    # Raise specific error
                    raise DataParsingError(f"No latency data columns found in {csv_file.name} (multi mode)")

                # Process each row (memory size)
                for i, size in enumerate(memory_sizes):
                    try:
                        # Attempt to convert latency data for this row to float
                        latencies = df.iloc[i][latency_cols].values.astype(float)
                        # Check for NaNs after conversion
                        if np.isnan(latencies).any():
                             log.warning(f"NaN values found in latency data for {csv_file.name}, size {size}MB. Check input CSV.")
                        
                        # Calculate stats, handle potential NaNs
                        mean_val = np.nanmean(latencies) if not np.isnan(latencies).all() else np.nan
                        std_val = np.nanstd(latencies) if not np.isnan(latencies).all() else np.nan
                        min_val = np.nanmin(latencies) if not np.isnan(latencies).all() else np.nan
                        max_val = np.nanmax(latencies) if not np.isnan(latencies).all() else np.nan
                        
                        data_by_size[size] = {
                            'latencies': latencies, 'mean': mean_val, 'std': std_val,
                            'min': min_val, 'max': max_val,
                        }
                    except ValueError as ve:
                        # Raise specific error with context
                        raise DataParsingError(f"Error converting multi-size latency data to float in {csv_file.name} at size {size}MB, row {i}: {ve}. Problematic data: {df.iloc[i][latency_cols].values}")
                    except IndexError:
                         # Raise specific error
                        raise DataParsingError(f"Error accessing data row {i} for size {size}MB in {csv_file.name}. CSV structure may be incorrect.")

                file_data.update({'memory_sizes': memory_sizes, 'data_by_size': data_by_size})
                data.append(file_data)

            log.debug(f"Successfully processed {csv_file.name}")
        # Catch pandas-specific parsing errors
        except pd.errors.ParserError as pe:
             raise DataParsingError(f"Error parsing CSV file {csv_file.name}: {pe}")
        # Catch file not found errors that might occur if file disappears mid-run (unlikely but defensive)
        except FileNotFoundError:
             raise DataParsingError(f"Error: File {csv_file.name} not found during processing loop.")
        # Catch any other unexpected error during file processing
        except Exception as e:
             # Log the full traceback for unexpected errors
             log.exception(f"Unexpected error processing {csv_file.name}")
             # Re-raise as a DataParsingError for consistent handling in main
             raise DataParsingError(f"Unexpected error processing {csv_file.name}: {e}")

    if not data:
        # Raise specific error if no data could be parsed at all
        raise DataParsingError("Error: No data successfully parsed from any CSV files.")
    # Return data sorted by rank count, and the detected/specified mode
    return sorted(data, key=lambda x: x['ranks']), mode

# --- Plotting Orchestration ---

def create_plots(data, plots_dir, mode='single', no_3d=False):
    """Creates and saves all required plots based on the analysis mode.

    Delegates plotting tasks to specific functions based on whether the
    mode is 'single' or 'multiple'. Separates sequential and interleaved
    data before passing it to the plotting functions.

    Args:
        data (list[dict]): The parsed benchmark data, as returned by
                           `parse_csv_files`.
        plots_dir (Path): The directory where plots should be saved.
        mode (str, optional): The analysis mode ('single' or 'multiple').
                              Defaults to 'single'.
        no_3d (bool, optional): If True, skips 3D plot generation in
                                'multiple' mode. Defaults to False.
    """
    log.info(f"Creating plots in {plots_dir} (mode: {mode})")
    plots_dir.mkdir(exist_ok=True)
    
    sequential_data = [d for d in data if not d['is_interleaved']]
    interleaved_data = [d for d in data if d['is_interleaved']]
    
    try:
        if mode == 'single':
            # Create single-size specific plots (2x2 layout)
            if sequential_data: create_single_size_plots(sequential_data, "Sequential", plots_dir)
            if interleaved_data: create_single_size_plots(interleaved_data, "Interleaved", plots_dir)
            # Create comparison plot if both exist
            if sequential_data and interleaved_data:
                 create_single_comparison_plot(sequential_data, interleaved_data, plots_dir)
        
        else: # mode == 'multiple'
            # Create multi-size specific plots
            create_multi_size_plots(sequential_data, interleaved_data, plots_dir)
            create_size_comparison_plots(sequential_data, interleaved_data, plots_dir) # Compare seq vs int per size
            if not no_3d:
                create_3d_plots(sequential_data, interleaved_data, plots_dir)

        log.info("Successfully created plots.")
    except Exception as e:
        log.exception("An error occurred during plot creation.")

# --- Single-Size Mode Plotting Helpers ---

def create_single_comparison_plot(sequential_data, interleaved_data, plots_dir):
    """Creates a plot comparing Sequential vs Interleaved data for single-size mode.

    Plots the mean latency with shaded min/max range for both data types
    against the number of MPI ranks.

    Args:
        sequential_data (list[dict]): Parsed data for sequential runs.
        interleaved_data (list[dict]): Parsed data for interleaved runs.
        plots_dir (Path): The directory to save the plot.
    """
    log.debug("Creating single-size comparison plot (Sequential vs Interleaved)...")
    try:
        s_ranks = [d['ranks'] for d in sequential_data]; s_means = [d['mean'] for d in sequential_data]
        s_mins = [d['min'] for d in sequential_data]; s_maxs = [d['max'] for d in sequential_data]
        i_ranks = [d['ranks'] for d in interleaved_data]; i_means = [d['mean'] for d in interleaved_data]
        i_mins = [d['min'] for d in interleaved_data]; i_maxs = [d['max'] for d in interleaved_data]
        
        plt.figure(figsize=(14, 8))
        plt.plot(s_ranks, s_means, 'b-', linewidth=2, label='Sequential Mean')
        plt.fill_between(s_ranks, s_mins, s_maxs, color='blue', alpha=0.2)
        plt.plot(i_ranks, i_means, 'r-', linewidth=2, label='Interleaved Mean')
        plt.fill_between(i_ranks, i_mins, i_maxs, color='red', alpha=0.2)
        
        plt.xlabel('Number of MPI Ranks', fontsize=12); plt.ylabel('Latency (ns)', fontsize=12)
        plt.title('Memory Latency Comparison: Sequential vs Interleaved', fontsize=14)
        plt.legend(fontsize=11); plt.grid(True)
        plot_path = plots_dir / 'single_size_comparison.png'
        plt.savefig(plot_path, bbox_inches='tight'); plt.close()
        log.debug(f"Saved single-size comparison plot to {plot_path}")
    except Exception as e:
        log.exception("Failed to create single-size comparison plot.")

def create_single_size_plots(data_set, title_prefix, plots_dir):
    """Creates the 2x2 analysis plot grid for single-size mode data.

    Generates a single figure with four subplots:
      1. Min/Max/Mean Latency vs. Ranks
      2. Box Plot of Latency Distribution vs. Ranks
      3. Sum of Latencies & Relative Efficiency vs. Ranks (Twin Axes)
      4. Heatmap of Latency per Rank ID vs. Ranks

    Args:
        data_set (list[dict]): The parsed data for a specific allocation type
                               (Sequential or Interleaved).
        title_prefix (str): The prefix for the plot title and filename
                            (e.g., "Sequential").
        plots_dir (Path): The directory to save the plot.
    """
    log.debug(f"Creating single-size 2x2 analysis plots for {title_prefix}...")
    try:
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(f'{title_prefix} Scaling Analysis', fontsize=16, y=1.01)
        
        ranks = sorted([d['ranks'] for d in data_set])
        data_map = {d['ranks']: d for d in data_set}
        if not ranks: log.warning(f"No data for {title_prefix} single-size plots."); return

        means = [data_map[r]['mean'] for r in ranks]
        stds = [data_map[r]['std'] for r in ranks]
        mins = [data_map[r]['min'] for r in ranks]
        maxs = [data_map[r]['max'] for r in ranks]
        sum_latencies = []
        all_latencies_list = []
        max_rank_id = 0
        for r in ranks:
            if r in data_map and 'latencies' in data_map[r] and isinstance(data_map[r]['latencies'], np.ndarray):
                lats = data_map[r]['latencies']
                sum_latencies.append(np.sum(lats))
                all_latencies_list.append(lats)
                max_rank_id = max(max_rank_id, len(lats) - 1)
            else:
                sum_latencies.append(np.nan); all_latencies_list.append(None)
                log.warning(f"Missing latency data for rank count {r} ({title_prefix})")

        # --- Subplot 1: Min/Max/Average ---
        ax1 = plt.subplot(2, 2, 1)
        ln1 = ax1.plot(ranks, mins, 'g-', label='Min'); ln2 = ax1.plot(ranks, means, 'b-', label='Mean')
        ln3 = ax1.plot(ranks, maxs, 'r-', label='Max')
        ax1.set_xlabel('Number of MPI Ranks'); ax1.set_ylabel('Latency (ns)')
        ax1.set_title('Memory Latency: Min/Max/Average'); ax1.legend(handles=ln1+ln2+ln3); ax1.grid(True)

        # --- Subplot 2: Box Plot ---
        ax2 = plt.subplot(2, 2, 2)
        box_data = [lats for lats in all_latencies_list if lats is not None]
        box_labels = [str(r) for r, lats in zip(ranks, all_latencies_list) if lats is not None]
        if box_data:
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_xlabel('Number of Ranks'); ax2.set_ylabel('Latency (ns)')
            ax2.set_title('Per-Rank Latency Distribution'); ax2.grid(True)
        else: log.warning(f"No valid latency data for {title_prefix} boxplot.")

        # --- Subplot 3: Sum & Efficiency ---
        ax3 = plt.subplot(2, 2, 3)
        ax3_twin = ax3.twinx(); lines3 = []
        line_sum, = ax3.plot(ranks, sum_latencies, 'm-p', linewidth=2, markersize=8, label='Sum of Latencies')
        ax3.set_xlabel('Number of MPI Ranks'); ax3.set_ylabel('Sum of Latencies (ns)', color='m')
        ax3.tick_params(axis='y', labelcolor='m'); ax3.grid(True); lines3.append(line_sum)
        if len(ranks) >= 2:
            baseline_idx = ranks.index(min(ranks)) # Use lowest rank count as baseline
            baseline_mean = means[baseline_idx]
            efficiency = [(baseline_mean / mean) if mean != 0 else np.inf for mean in means]
            line_eff, = ax3_twin.plot(ranks, efficiency, 'c-o', linewidth=2, label='Relative Efficiency')
            line_base = ax3_twin.axhline(y=1.0, color='r', linestyle='--', label=f'Baseline ({min(ranks)} ranks)')
            ax3_twin.set_ylabel('Relative Efficiency', color='c'); ax3_twin.tick_params(axis='y', labelcolor='c')
            lines3.extend([line_eff, line_base])
        else: log.warning(f"Not enough data points for scaling efficiency plot ({title_prefix})")
        ax3.set_title('Sum & Relative Efficiency'); ax3.legend(handles=lines3, loc='best')

        # --- Subplot 4: Heatmap ---
        ax4 = plt.subplot(2, 2, 4)
        heatmap_data = np.full((max_rank_id + 1, len(ranks)), np.nan)
        for col_idx, r in enumerate(ranks):
            if all_latencies_list[col_idx] is not None:
                lats = all_latencies_list[col_idx]; num_rows_to_fill = min(len(lats), max_rank_id + 1)
                heatmap_data[0:num_rows_to_fill, col_idx] = lats[0:num_rows_to_fill]
        heatmap_data_transposed = heatmap_data.T
        if not np.isnan(heatmap_data_transposed).all():
            extent = [-0.5, max_rank_id + 0.5, 
                      min(ranks) - 0.5 * (ranks[1]-ranks[0]) if len(ranks)>1 else min(ranks)-1, 
                      max(ranks) + 0.5 * (ranks[1]-ranks[0]) if len(ranks)>1 else max(ranks)+1]
            im = ax4.imshow(heatmap_data_transposed, cmap='viridis', aspect='auto', interpolation='nearest', origin='lower', extent=extent)
            ax4.set_xlabel('Rank ID')
            if max_rank_id < 20: ax4.set_xticks(np.arange(0, max_rank_id + 1, max(1, max_rank_id // 10)))
            ax4.set_ylabel('Number of MPI Ranks'); ax4.set_yticks(ranks); ax4.set_yticklabels([str(r) for r in ranks])
            ax4.set_title('Latency Heatmap per Rank')
            cbar = fig.colorbar(im, ax=ax4, orientation='vertical', shrink=0.8); cbar.set_label('Latency (ns)')
        else:
            log.warning(f"No valid data for heatmap ({title_prefix})")
            ax4.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_xticks([]); ax4.set_yticks([])

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plot_path = plots_dir / f'{title_prefix.lower()}_single_analysis.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300); plt.close(fig)
        log.debug(f"Saved single-size 2x2 analysis plots to {plot_path}")
        
    except Exception as e:
        log.exception(f"Failed to create single-size plots for {title_prefix}.")
        plt.close('all')


# --- Multi-Size Mode Plotting Helpers ---

# Helper: Plot Mean vs Size (adapted from analyze_multiple_sizes.py)
# def _create_mean_vs_size_plot(data_set, title_prefix, plots_dir):
#     # ... (code removed, docstring would go here if kept)

# Helper: Plot statistical metrics vs Size (from analyze_numa_results.py)
def _create_multi_size_stats_plot(data_set, title_prefix, plots_dir):
    """Creates the 2x2 plot: Median, Sum, P99, StdDev vs Memory Size.

    Args:
        data_set (list[dict]): The parsed multi-size data for a specific type.
        title_prefix (str): Prefix for title and filename (e.g., "Sequential").
        plots_dir (Path): Directory to save the plot.
    """
    log.debug(f"Creating Median/Sum/P99/StdDev vs Size plot (2x2) for {title_prefix}...")
    colors = plt.cm.tab20(np.linspace(0, 1, 20)); markers = ['o','s','^','v','<','>','p','*','h','D']
    try:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{title_prefix} Memory Latency Statistics vs Size', fontsize=16)
        
        ax_median = axes[0, 0]; ax_sum = axes[0, 1]
        ax_p99 = axes[1, 0]; ax_std = axes[1, 1]
        ax_median.get_shared_y_axes().join(ax_median, ax_p99) # Share Y for Median/P99
        
        lines, labels = [], []
        for i, d in enumerate(data_set):
            if 'data_by_size' not in d or not isinstance(d.get('data_by_size'), dict): continue
            sizes = sorted(d['data_by_size'].keys()); 
            if not sizes: continue
            medians, p99s, sums, stds, valid_sizes = [], [], [], [], []
            for size in sizes:
                if 'latencies' in d['data_by_size'][size] and len(d['data_by_size'][size]['latencies']) > 0:
                    lats = np.array(d['data_by_size'][size]['latencies'])
                    medians.append(np.median(lats)); p99s.append(np.percentile(lats, 99))
                    sums.append(np.sum(lats)); stds.append(np.std(lats))
                    valid_sizes.append(size)
            if not valid_sizes: continue
            color = colors[i % len(colors)]; marker = markers[i % len(markers)]; label = f"{d['ranks']} ranks"
            
            line_median, = ax_median.plot(valid_sizes, medians, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label)
            ax_sum.plot(valid_sizes, sums, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label)
            ax_p99.plot(valid_sizes, p99s, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label)
            ax_std.plot(valid_sizes, stds, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label)
            lines.append(line_median); labels.append(label)

        ax_median.set_title('Median Latency'); ax_median.set_ylabel('Latency (ns)')
        ax_median.set_xscale('log', base=2); ax_median.tick_params(labelbottom=False); ax_median.grid(True, which="both", ls="-", alpha=0.3)
        ax_sum.set_title('Sum of Latencies'); ax_sum.set_ylabel('Total Latency (ns)')
        ax_sum.set_xscale('log', base=2); ax_sum.tick_params(labelbottom=False); ax_sum.grid(True, which="both", ls="-", alpha=0.3)
        ax_p99.set_title('P99 Latency'); ax_p99.set_ylabel('Latency (ns)')
        ax_p99.set_xlabel('Memory Size (MB)'); ax_p99.set_xscale('log', base=2); ax_p99.grid(True, which="both", ls="-", alpha=0.3)
        ax_std.set_title('Standard Deviation'); ax_std.set_ylabel('Latency Std Dev (ns)')
        ax_std.set_xlabel('Memory Size (MB)'); ax_std.set_xscale('log', base=2); ax_std.grid(True, which="both", ls="-", alpha=0.3)

        if lines: fig.legend(lines, labels, bbox_to_anchor=(0.97, 0.5), loc='center left', borderaxespad=0., fontsize='large')
        fig.tight_layout(rect=[0, 0, 0.88, 0.95]) 

        if not any(ax.has_data() for ax in axes.flatten()):
             log.warning(f"No data plotted for {title_prefix} stats... Skipping save.")
             plt.close(fig); return
        plot_path = plots_dir / f'{title_prefix.lower()}_multi_size_stats.png' 
        plt.savefig(plot_path, bbox_inches='tight', dpi=300); plt.close(fig)
        log.debug(f"Saved Multi-size Stats plot (2x2 layout) to {plot_path}")
        
    except Exception as e:
        log.exception(f"Failed to create {title_prefix} Multi-size Stats plot (2x2 layout).")
        plt.close('all')

# Main function calling the multi-size 2D plotting helpers
def create_multi_size_plots(sequential_data, interleaved_data, plots_dir):
    """Creates all 2D plots specific to multiple memory size analysis mode.

    Currently generates the 2x2 statistical plot (Median, Sum, P99, StdDev
    vs Memory Size).

    Args:
        sequential_data (list[dict]): Parsed data for sequential runs.
        interleaved_data (list[dict]): Parsed data for interleaved runs.
        plots_dir (Path): The directory to save plots.
    """
    log.debug("Creating multi-size 2D plots...")
    if sequential_data:
        _create_multi_size_stats_plot(sequential_data, "Sequential", plots_dir)
        # _create_mean_vs_size_plot(sequential_data, "Sequential", plots_dir) # Removed call
    if interleaved_data:
        _create_multi_size_stats_plot(interleaved_data, "Interleaved", plots_dir)
        # _create_mean_vs_size_plot(interleaved_data, "Interleaved", plots_dir) # Removed call

# --- 3D Plotting Helpers (Multi-Size) ---

# Helper: Fallback 2D Median plot (originally from analyze_numa_results.py)
def _create_fallback_2d_median_plot(data_set, title_prefix, plots_dir):
    """Creates a 2D plot of Median Latency vs Size as a fallback for 3D errors.

    Plots median latency curves for each rank count against memory size.

    Args:
        data_set (list[dict]): The parsed multi-size data for a specific type.
        title_prefix (str): Prefix for title and filename (e.g., "Sequential").
        plots_dir (Path): Directory to save the plot.
    """
    log.warning(f"Creating fallback 2D plot (Median Latency) for {title_prefix} due to 3D plot error.")
    try:
        plt.figure(figsize=(12, 8))
        latency_data = {}
        for d in data_set:
            if 'data_by_size' not in d: continue
            rank = d['ranks']; latency_data[rank] = {}
            for size, stats in d['data_by_size'].items():
                if 'latencies' in stats and len(stats['latencies']) > 0:
                    latency_data[rank][size] = np.median(stats['latencies']) 
        
        if not latency_data: log.error("No data for fallback 2D median plot."); plt.close(); return
             
        for rank in sorted(latency_data.keys()):
            if not latency_data[rank]: continue
            sizes = sorted(latency_data[rank].keys())
            latencies = [latency_data[rank][size] for size in sizes]
            if sizes: plt.plot(sizes, latencies, marker='o', linewidth=2, label=f'{rank} ranks')
        
        plt.xscale('log', base=2); plt.xlabel('Memory Size (MB)'); plt.ylabel('Median Latency (ns)')
        plt.title(f'{title_prefix} Median Memory Latency by Rank Count (Fallback)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend()
        plot_path = plots_dir / f'fallback_{title_prefix.lower()}_median_latency.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300); plt.close()
        log.debug(f"Saved fallback 2D Median plot to {plot_path}")
    except Exception as e:
        log.exception(f"Failed to create fallback 2D Median plot for {title_prefix}.")
        plt.close()

# Helper: 3D plot based on Median (adapted from analyze_numa_results.py)
def _create_3d_median_plot(data_set, title_prefix, plots_dir):
    """Creates the 3D surface plot showing Median Latency vs Ranks and Size.

    Generates a figure with two views (Side and Flat) and a central colorbar.
    Requires data with at least 2 rank counts and 2 common memory sizes.
    Calls `_create_fallback_2d_median_plot` if 3D plotting fails.

    Args:
        data_set (list[dict]): The parsed multi-size data for a specific type.
        title_prefix (str): Prefix for title and filename (e.g., "Sequential").
        plots_dir (Path): Directory to save the plot.
    """
    log.debug(f"Creating 3D Median latency plot for {title_prefix}...")
    try:
        if not all('data_by_size' in d and d['data_by_size'] for d in data_set):
            log.warning(f"Skipping 3D median plot for {title_prefix}: missing multi-size data."); return
            
        all_ranks = sorted(list({d['ranks'] for d in data_set}))
        all_sizes_sets = [set(d['data_by_size'].keys()) for d in data_set if 'data_by_size' in d]
        if not all_sizes_sets: log.warning(f"Skipping 3D median plot for {title_prefix}: No size data."); return
        common_sizes = set.intersection(*all_sizes_sets) if len(all_sizes_sets) > 1 else all_sizes_sets[0]
        if not common_sizes: log.warning(f"Skipping 3D median plot for {title_prefix}: No common sizes."); return 
        all_sizes = sorted(list(common_sizes))
        
        if len(all_ranks) < 2 or len(all_sizes) < 2:
            log.warning(f"Skipping 3D median plot for {title_prefix}: Need >= 2 ranks and >= 2 sizes."); return
            
        X, Y = np.meshgrid(all_ranks, all_sizes)
        Z = np.full(X.shape, np.nan)
        rank_map = {rank: i for i, rank in enumerate(all_ranks)}
        size_map = {size: i for i, size in enumerate(all_sizes)}
        
        for d in data_set:
            if 'data_by_size' not in d or d['ranks'] not in rank_map: continue
            rank_idx = rank_map[d['ranks']]
            for size, stats in d['data_by_size'].items():
                if size in size_map and 'latencies' in stats and len(stats['latencies']) > 0:
                    Z[size_map[size], rank_idx] = np.median(stats['latencies']) 
        
        Z_masked = np.ma.masked_invalid(Z)
        if Z_masked.mask.all(): log.warning(f"Skipping 3D median plot for {title_prefix}: All data invalid."); return
             
        fig = plt.figure(figsize=(22, 9)); fig.suptitle(f'{title_prefix} Median Memory Latency Surface', fontsize=16, y=0.98)
        pos_left = [0.02, 0.05, 0.45, 0.85]; pos_right = [0.53, 0.05, 0.45, 0.85]; pos_cbar = [0.48, 0.15, 0.015, 0.6]

        ax_side = fig.add_subplot(1, 2, 1, projection='3d'); ax_side.set_position(pos_left)
        surf = ax_side.plot_surface(X, Y, Z_masked, cmap='jet', linewidth=0, antialiased=True, edgecolor='none')
        ax_side.set_xlabel('Number of Ranks'); ax_side.set_ylabel('Memory Size (MB)'); ax_side.set_zlabel('Median Latency (ns)')
        ax_side.set_title('Side View (elev=30, azim=120)'); ax_side.set_xticks(all_ranks); ax_side.set_xticklabels([str(r) for r in all_ranks])
        if len(all_sizes) > 1: ax_side.set_yticks(all_sizes); ax_side.set_yticklabels([str(s) for s in all_sizes])
        ax_side.grid(True); ax_side.view_init(elev=30, azim=120)
        
        ax_flat = fig.add_subplot(1, 2, 2, projection='3d'); ax_flat.set_position(pos_right)
        ax_flat.plot_surface(X, Y, Z_masked, cmap='jet', linewidth=0, antialiased=True, edgecolor='none')
        ax_flat.set_xlabel('Number of Ranks'); ax_flat.set_ylabel('Memory Size (MB)'); ax_flat.set_zlabel('Median Latency (ns)')
        ax_flat.set_title('Flat View (elev=10, azim=170)'); ax_flat.set_xticks(all_ranks); ax_flat.set_xticklabels([str(r) for r in all_ranks])
        if len(all_sizes) > 1: ax_flat.set_yticks(all_sizes); ax_flat.set_yticklabels([str(s) for s in all_sizes])
        ax_flat.grid(True); ax_flat.view_init(elev=10, azim=170)
        
        cbar_ax = fig.add_axes(pos_cbar); fig.colorbar(surf, cax=cbar_ax, label='Median Memory Latency (ns)')
        
        plot_path = plots_dir / f'{title_prefix.lower()}_3d_median_latency.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300); plt.close(fig)
        log.debug(f"Saved 3D Median plot to {plot_path}")
                
    except Exception as e:
        log.exception(f"Error creating 3D Median plot ({title_prefix}). Trying fallback...")
        plt.close('all')
        # Fallback using existing fallback function (based on median)
        _create_fallback_2d_median_plot(data_set, title_prefix, plots_dir)

# Main function calling 3D plotting helpers
def create_3d_plots(sequential_data, interleaved_data, plots_dir):
    """Creates 3D plots (Median based only) for multi-size mode.

    Calls the helper function to generate the Median latency 3D surface plots
    for both sequential and interleaved data if available.

    Args:
        sequential_data (list[dict]): Parsed data for sequential runs.
        interleaved_data (list[dict]): Parsed data for interleaved runs.
        plots_dir (Path): The directory to save plots.
    """
    log.info("Creating 3D plots (Median based only)...")
    if sequential_data:
         _create_3d_median_plot(sequential_data, "Sequential", plots_dir)
    if interleaved_data:
         _create_3d_median_plot(interleaved_data, "Interleaved", plots_dir)


# --- Size Comparison Plotting (Multi-Size) ---

def create_size_comparison_plots(sequential_data, interleaved_data, plots_dir):
    """Creates comparison plots (Sequential vs Interleaved) for common memory sizes.

    For each memory size found in *both* sequential and interleaved data,
    generates a plot comparing their mean latencies (with min/max range)
    against the number of MPI ranks.

    Args:
        sequential_data (list[dict]): Parsed data for sequential runs.
        interleaved_data (list[dict]): Parsed data for interleaved runs.
        plots_dir (Path): The directory to save plots.
    """
    log.debug("Creating size comparison plots (multiple sizes)...")
    if not (interleaved_data and sequential_data):
        log.info("Skipping size comparison plots (need both sequential and interleaved multi-size data).")
        return
        
    try:
        interleaved_sizes_list = [set(d['data_by_size'].keys()) for d in interleaved_data if 'data_by_size' in d]
        sequential_sizes_list = [set(d['data_by_size'].keys()) for d in sequential_data if 'data_by_size' in d]
        if not interleaved_sizes_list or not sequential_sizes_list:
             log.warning("Cannot create size comparison plots: Missing multi-size data for one type."); return
             
        common_interleaved = set.intersection(*interleaved_sizes_list) if len(interleaved_sizes_list) > 1 else interleaved_sizes_list[0]
        common_sequential = set.intersection(*sequential_sizes_list) if len(sequential_sizes_list) > 1 else sequential_sizes_list[0]
        common_sizes = sorted(list(common_interleaved.intersection(common_sequential)))
        
        if not common_sizes: log.warning("No common memory sizes found for comparison plots."); return
        log.debug(f"Creating comparison plots for common sizes: {common_sizes}")
        
        for size in common_sizes:
            try:
                plt.figure(figsize=(12, 8))
                seq_ranks, seq_means, seq_mins, seq_maxs = [], [], [], []
                for d in sorted(sequential_data, key=lambda x: x['ranks']):
                    if 'data_by_size' in d and size in d['data_by_size']:
                        stats = d['data_by_size'][size]
                        seq_ranks.append(d['ranks']); seq_means.append(stats['mean'])
                        seq_mins.append(stats['min']); seq_maxs.append(stats['max'])
                
                int_ranks, int_means, int_mins, int_maxs = [], [], [], []
                for d in sorted(interleaved_data, key=lambda x: x['ranks']):
                     if 'data_by_size' in d and size in d['data_by_size']:
                        stats = d['data_by_size'][size]
                        int_ranks.append(d['ranks']); int_means.append(stats['mean'])
                        int_mins.append(stats['min']); int_maxs.append(stats['max'])
                
                if not seq_ranks or not int_ranks:
                     log.warning(f"Skipping comparison plot for size {size}MB: Missing data."); plt.close(); continue
                     
                plt.plot(seq_ranks, seq_means, 'b-', linewidth=2, label='Sequential Mean')
                plt.fill_between(seq_ranks, seq_mins, seq_maxs, color='blue', alpha=0.2)
                plt.plot(int_ranks, int_means, 'r-', linewidth=2, label='Interleaved Mean')
                plt.fill_between(int_ranks, int_mins, int_maxs, color='red', alpha=0.2)
                
                plt.xlabel('Number of MPI Ranks'); plt.ylabel('Latency (ns)')
                plt.title(f'Memory Latency Comparison ({size} MB): Sequential vs Interleaved')
                plt.legend(); plt.grid(True)
                plot_path = plots_dir / f'multi_size_comparison_{size}MB.png'
                plt.savefig(plot_path, bbox_inches='tight'); plt.close()
                log.debug(f"Saved size comparison plot to {plot_path}")
                
            except Exception as e_size: log.exception(f"Failed to create comparison plot for size {size}MB."); plt.close('all')
                 
    except Exception as e: log.exception("Error creating size comparison plots."); plt.close('all')

# --- Summary File Creation ---

def create_summary(data, job_dir, mode='single'):
    """Creates a text summary file with statistical analysis results.

    Writes calculated statistics (mean, std, min, max, CoV) to
    `analysis_summary.txt` in the plots subdirectory.
    The content format depends on the analysis mode ('single' or 'multiple').

    Args:
        data (list[dict]): The parsed benchmark data.
        job_dir (Path): The path to the main job directory.
        mode (str): The analysis mode ('single' or 'multiple') used.
    """
    log.info("Creating analysis summary file...")
    summary_file = job_dir / 'plots' / 'analysis_summary.txt'
    plots_dir = job_dir / 'plots' # Ensure plots_dir exists
    plots_dir.mkdir(exist_ok=True) 
    try:
        with open(summary_file, 'w') as f:
            f.write("NUMA Benchmark Analysis Summary\n")
            f.write("=============================\n\n")
            f.write(f"Analysis Mode: {mode}\n")
            f.write(f"Source Directory: {job_dir.resolve()}\n\n")

            if mode == 'single':
                f.write("Statistical Analysis per Rank Count (Single Size):\n")
                f.write("-----------------------------------------------\n")
                for d in data:
                    alloc_type = "Interleaved" if d['is_interleaved'] else "Sequential"
                    f.write(f"\n{d['ranks']} ranks ({alloc_type} - {d.get('source_file', 'N/A')}):\n")
                    f.write(f"  Mean latency: {d['mean']:.2f} ns\n")
                    f.write(f"  Std dev:      {d['std']:.2f} ns\n")
                    f.write(f"  Min latency:  {d['min']:.2f} ns\n")
                    f.write(f"  Max latency:  {d['max']:.2f} ns\n")
                    if d['mean'] != 0:
                         f.write(f"  CoV:          {(d['std'] / d['mean']) * 100:.2f}%\n")
                    else:
                         f.write("  CoV:          N/A (mean is zero)\n")
            
            else: # mode == 'multiple'
                f.write("Statistical Analysis per Rank Count and Memory Size:\n")
                f.write("------------------------------------------------\n")
                for d in sorted(data, key=lambda x: x['ranks']):
                    alloc_type = "Interleaved" if d['is_interleaved'] else "Sequential"
                    f.write(f"\n{d['ranks']} ranks ({alloc_type} - {d.get('source_file', 'N/A')}):\n")
                    
                    if 'data_by_size' not in d or not d['data_by_size']:
                         f.write("  No multi-size data available for this configuration.\n"); continue
                         
                    for size in sorted(d['data_by_size'].keys()):
                        stats = d['data_by_size'][size]
                        f.write(f"\n  Memory Size: {size} MB\n")
                        f.write(f"    Mean latency: {stats['mean']:.2f} ns\n")
                        f.write(f"    Std dev:      {stats['std']:.2f} ns\n")
                        f.write(f"    Min latency:  {stats['min']:.2f} ns\n")
                        f.write(f"    Max latency:  {stats['max']:.2f} ns\n")
                        f.write(f"    Range:        {stats['max'] - stats['min']:.2f} ns\n")
                        if stats['mean'] != 0:
                             cov = (stats['std'] / stats['mean']) * 100
                             f.write(f"    CoV:          {cov:.2f}%\n")
                        else:
                             f.write("    CoV:          N/A (mean is zero)\n")
        log.info(f"Successfully created summary file: {summary_file}")
    except IOError as e: log.error(f"Error writing summary file {summary_file}: {e}")
    except Exception as e: log.exception("Unexpected error during summary creation.")

# --- Main Execution ---

def main():
    """Main execution function.

    Parses arguments, sets up logging, creates the plots directory,
    parses CSV data, creates plots based on the detected/specified mode,
    and generates the summary text file.
    Handles potential errors during parsing and analysis.
    """
    args = parse_args()

    # Setup logging level based on verbosity
    if args.verbose: log.setLevel(logging.DEBUG); log.info("Verbose logging enabled.")
    else: log.setLevel(logging.INFO)

    job_dir = args.job_dir
    # Define plots_dir early and ensure it exists
    plots_dir = job_dir / 'plots'
    try:
         plots_dir.mkdir(exist_ok=True); log.debug(f"Ensured plots directory exists: {plots_dir}")
    except OSError as e:
         # Log specific error for directory creation failure
         log.error(f"Could not create plots directory {plots_dir}: {e}");
         sys.exit(1) # Exit here, as we can't save plots/summary

    # --- Run Analysis within a try block to catch custom errors --- 
    try:
        # Parse data and get detected/specified mode
        data, detected_mode = parse_csv_files(job_dir, args.mode, verbose=args.verbose)

        # Create plots based on mode
        # Plotting errors are logged within create_plots but don't stop summary generation
        create_plots(data, plots_dir, detected_mode, args.no_3d)

        # Create summary file
        # Summary creation errors are logged within create_summary
        create_summary(data, job_dir, detected_mode)

        log.info(f"Analysis complete. Results saved in {job_dir}")
        log.info(f"Plots saved in {plots_dir}")
        log.info(f"Summary saved in {plots_dir}/analysis_summary.txt")

    # Catch specific data parsing errors
    except DataParsingError as e:
        log.error(f"Data Parsing Failed: {e}")
        sys.exit(1) # Exit on critical data parsing errors
    # Catch unexpected errors during the main analysis workflow
    except Exception as e:
        log.exception(f"A critical error occurred during the analysis workflow: {e}") # Log full traceback
        sys.exit(1)

if __name__ == "__main__":
    main()