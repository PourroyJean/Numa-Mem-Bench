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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze NUMA benchmark results')
    parser.add_argument('job_dir', type=Path, help='Directory containing benchmark results')
    parser.add_argument('--mode', choices=['single', 'multiple'], default=None,
                      help='Analysis mode: single or multiple memory sizes (default: auto-detect)')
    parser.add_argument('--no-3d', action='store_true',
                      help='Skip 3D visualization in multiple size mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose (DEBUG) logging')
    # Parse and return args, logging setup moved to main
    args = parser.parse_args()
    return args

def detect_data_mode(csv_files, verbose=False):
    """Detect whether the data contains multiple memory sizes."""
    if not csv_files:
        log.debug("No CSV files found, defaulting to single size mode.")
        return 'single'
    
    # Check the first CSV file for memory size column
    try:
        df = pd.read_csv(csv_files[0])
        if 'size (MB)' in df.columns and len(df) > 1:
            if verbose:
                log.debug(f"Detected multiple memory sizes in {csv_files[0].name}")
            return 'multiple'
        else:
            if verbose:
                log.debug(f"File {csv_files[0].name} appears to be single-size data.")
            return 'single'
    except pd.errors.EmptyDataError:
        log.warning(f"File {csv_files[0].name} is empty, cannot determine mode accurately. Defaulting to single.")
        return 'single'
    except Exception as e:
        log.warning(f"Could not accurately determine data mode from {csv_files[0].name}: {e}. Defaulting to single.")
        return 'single'

def parse_csv_files(job_dir, mode=None, verbose=False):
    """Parse all CSV files in the job directory and extract relevant data."""
    log.info(f"Starting to parse CSV files in {job_dir}")
    data = []
    
    if not job_dir.exists() or not job_dir.is_dir():
        log.error(f"Error: Directory {job_dir} does not exist or is not a directory")
        sys.exit(1)
    
    # Get all CSV files with benchmark results
    csv_files = [f for f in job_dir.glob("*.csv") if f.stem.startswith(('sequential_', 'interleaved_'))]
    log.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        log.error(f"Error: No valid CSV files (sequential_*.csv or interleaved_*.csv) found in {job_dir}")
        sys.exit(1)
    
    # Auto-detect mode if not specified
    if mode is None:
        mode = detect_data_mode(csv_files, verbose=verbose) # Pass verbose flag
        log.info(f"Auto-detected analysis mode: {mode}")
    else:
        log.info(f"Using specified analysis mode: {mode}")
    
    # Parse each file
    for csv_file in csv_files:
        log.debug(f"Processing file: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                log.warning(f"Skipping empty file: {csv_file.name}")
                continue
            if len(df.columns) < 2:
                log.error(f"Error: Invalid CSV format in {csv_file.name} - expected at least 2 columns, found {len(df.columns)}")
                sys.exit(1)
            
            # Extract number of ranks from filename
            match = re.search(r'(\d+)ranks', csv_file.stem)
            if not match:
                log.error(f"Error: Could not find number of ranks (e.g., '16ranks') in filename {csv_file.name}")
                sys.exit(1)
            ranks = int(match.group(1))
            log.debug(f"Found {ranks} ranks in filename")
            
            file_data = {
                'ranks': ranks,
                'is_interleaved': 'interleaved' in csv_file.stem,
                'source_file': csv_file.name
            }
            
            if mode == 'single':
                # Single size mode - use first row of data
                if 'size (MB)' in df.columns and len(df) > 1:
                    log.warning(f"File {csv_file.name} contains multiple sizes, but running in single-size mode. Using first row only.")
                
                # Determine latency columns (all except potentially 'size (MB)')
                latency_cols = [col for col in df.columns if col != 'size (MB)']
                if not latency_cols:
                    log.error(f"Error: No latency data columns found in {csv_file.name}")
                    sys.exit(1)
                latencies = df.iloc[0][latency_cols].values.astype(float)
                
                file_data.update({
                    'latencies': latencies,
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies)
                })
                data.append(file_data)

            else: # mode == 'multiple'
                if 'size (MB)' not in df.columns:
                    log.error(f"Error: File {csv_file.name} does not have 'size (MB)' column required for multi-size mode.")
                    # Option: skip this file or exit? Let's skip for now.
                    # sys.exit(1) 
                    log.warning(f"Skipping file {csv_file.name} due to missing 'size (MB)' column in multi-size mode.")
                    continue 
                
                memory_sizes = df['size (MB)'].values
                data_by_size = {}
                latency_cols = [col for col in df.columns if col != 'size (MB)']
                if not latency_cols:
                    log.error(f"Error: No latency data columns found in {csv_file.name}")
                    sys.exit(1)

                for i, size in enumerate(memory_sizes):
                    try:
                        latencies = df.iloc[i][latency_cols].values.astype(float)
                        data_by_size[size] = {
                'latencies': latencies,
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                        }
                    except ValueError as ve:
                        log.error(f"Error converting latency data to float in {csv_file.name} at size {size}MB, row {i}: {ve}")
                        log.error(f"Problematic data: {df.iloc[i][latency_cols].values}")
                        sys.exit(1)
                    except IndexError:
                        log.error(f"Error accessing data row {i} for size {size}MB in {csv_file.name}")
                        sys.exit(1)
                
                file_data.update({
                    'memory_sizes': memory_sizes,
                    'data_by_size': data_by_size
                })
                data.append(file_data)
            
            log.debug(f"Successfully processed {csv_file.name}")
        except pd.errors.ParserError as pe:
             log.error(f"Error parsing CSV file {csv_file.name}: {pe}")
             sys.exit(1)
        except FileNotFoundError:
             log.error(f"Error: File {csv_file.name} not found during processing loop (should not happen!).")
             sys.exit(1)
        except Exception as e:
            log.exception(f"An unexpected error occurred while processing {csv_file.name}") # Logs exception info
            sys.exit(1)
    
    if not data:
        log.error("Error: No data successfully parsed from any CSV files.")
        sys.exit(1)

    return sorted(data, key=lambda x: x['ranks']), mode

def create_plots(data, plots_dir, mode='single', no_3d=False):
    """Create all required plots and save them to the plots directory."""
    log.info(f"Creating plots in {plots_dir}")
    plots_dir.mkdir(exist_ok=True)
    
    try:
        if mode == 'single':
            create_single_size_plots(data, plots_dir)
        else:
            create_multiple_size_plots(data, plots_dir, no_3d)
        log.info("Successfully created plots.")
    except Exception as e:
        log.exception("An error occurred during plot creation.")
        # Continue to summary creation if possible, but log the error

def create_single_size_plots(data, plots_dir):
    """Create plots for single memory size analysis."""
    log.debug("Creating single-size plots...")
    # Separate interleaved and non-interleaved data
    interleaved_data = [d for d in data if d['is_interleaved']]
    sequential_data = [d for d in data if not d['is_interleaved']]
    
    # Create comparison plots if we have both types of data
    if interleaved_data and sequential_data:
        log.debug("Creating comparison plot (single-size)")
        create_comparison_plots(sequential_data, interleaved_data, plots_dir)
    else:
        log.debug("Skipping comparison plot (only one data type found).")
        
    # Create individual plots for each data type
    for data_set, title_prefix in [(sequential_data, "Sequential"), (interleaved_data, "Interleaved")]:
        if not data_set:
            log.debug(f"Skipping {title_prefix} plots (no data).")
            continue
        
        log.debug(f"Creating individual {title_prefix} plots (single-size)")
        create_individual_plots(data_set, title_prefix, plots_dir)

def create_multiple_size_plots(data, plots_dir, no_3d=False):
    """Create plots for multiple memory size analysis."""
    log.debug("Creating multiple-size plots...")
    # Separate interleaved and non-interleaved data
    interleaved_data = [d for d in data if d['is_interleaved']]
    sequential_data = [d for d in data if not d['is_interleaved']]
    
    # Create memory size plots
    log.debug("Creating memory size vs latency plots")
    create_memory_size_plots(sequential_data, interleaved_data, plots_dir)
    
    # Create 3D surface plots if enabled
    if not no_3d:
        log.debug("Creating 3D surface plots (Median Latency)")
        create_3d_surface_plots(sequential_data, interleaved_data, plots_dir)
    else:
        log.info("Skipping 3D surface plots as requested.")
        
    # Create size comparison plots
    log.debug("Creating size comparison plots (multiple-size)")
    create_size_comparison_plots(sequential_data, interleaved_data, plots_dir)

def create_comparison_plots(sequential_data, interleaved_data, plots_dir):
    """Create comparison plots between sequential and interleaved data (single-size mode)."""
    try:
        s_ranks = [d['ranks'] for d in sequential_data]
        s_means = [d['mean'] for d in sequential_data]
        s_mins = [d['min'] for d in sequential_data]
        s_maxs = [d['max'] for d in sequential_data]
        
        i_ranks = [d['ranks'] for d in interleaved_data]
        i_means = [d['mean'] for d in interleaved_data]
        i_mins = [d['min'] for d in interleaved_data]
        i_maxs = [d['max'] for d in interleaved_data]
        
        plt.figure(figsize=(14, 8))
        plt.plot(s_ranks, s_means, 'b-', linewidth=2, label='Sequential Mean')
        plt.fill_between(s_ranks, s_mins, s_maxs, color='blue', alpha=0.2)
        
        plt.plot(i_ranks, i_means, 'r-', linewidth=2, label='Interleaved Mean')
        plt.fill_between(i_ranks, i_mins, i_maxs, color='red', alpha=0.2)
        
        plt.xlabel('Number of MPI Ranks', fontsize=12)
        plt.ylabel('Latency (ns)', fontsize=12)
        plt.title('Memory Latency Comparison: Sequential vs Interleaved', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True)
        plot_path = plots_dir / 'ccd_comparison.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        log.debug(f"Saved comparison plot to {plot_path}")
    except Exception as e:
        log.exception("Failed to create single-size comparison plot.")

def create_individual_plots(data_set, title_prefix, plots_dir):
    """Create individual plots for a single data type (single-size mode)."""
    try:
        # Create a figure with 2x2 subplots
        fig = plt.figure(figsize=(16, 14)) # Increased size slightly for heatmap legend
        fig.suptitle(f'{title_prefix} Scaling Analysis', fontsize=16, y=1.01) # Adjust y slightly
        
        # --- Prepare data --- 
        ranks = sorted([d['ranks'] for d in data_set])
        data_map = {d['ranks']: d for d in data_set}
        means = [data_map[r]['mean'] for r in ranks]
        stds = [data_map[r]['std'] for r in ranks]
        mins = [data_map[r]['min'] for r in ranks]
        maxs = [data_map[r]['max'] for r in ranks]
        sum_latencies = []
        all_latencies_list = [] # Store actual latency arrays
        max_rank_id = 0
        for r in ranks:
            if r in data_map and 'latencies' in data_map[r] and isinstance(data_map[r]['latencies'], np.ndarray):
                lats = data_map[r]['latencies']
                sum_latencies.append(np.sum(lats))
                all_latencies_list.append(lats)
                max_rank_id = max(max_rank_id, len(lats) - 1)
            else:
                sum_latencies.append(np.nan)
                all_latencies_list.append(None) # Placeholder for missing data
                log.warning(f"Missing latency data for rank count {r} ({title_prefix})")

        # --- Subplots --- 
        
        # First subplot: Min/Max/Average
        ax1 = plt.subplot(2, 2, 1)
        ln1 = ax1.plot(ranks, mins, 'g-', label='Min')
        ln2 = ax1.plot(ranks, means, 'b-', label='Mean')
        ln3 = ax1.plot(ranks, maxs, 'r-', label='Max')
        ax1.set_xlabel('Number of MPI Ranks')
        ax1.set_ylabel('Latency (ns)')
        ax1.set_title(f'Memory Latency: Min/Max/Average')
        ax1.legend(handles=ln1+ln2+ln3)
        ax1.grid(True)

        # Second subplot: Box Plot of Per-Rank Performance Distribution
        ax2 = plt.subplot(2, 2, 2)
        box_data = [lats for lats in all_latencies_list if lats is not None]
        box_labels = [str(r) for r, lats in zip(ranks, all_latencies_list) if lats is not None]
        if box_data:
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_xlabel('Number of Ranks')
            ax2.set_ylabel('Latency (ns)')
            ax2.set_title(f'Per-Rank Latency Distribution')
            ax2.grid(True)
        else:
            log.warning(f"No valid latency data found for {title_prefix} boxplot.")

        # Third subplot: Sum of Latencies (Left) & Scaling Efficiency (Right)
        ax3 = plt.subplot(2, 2, 3)
        ax3_twin = ax3.twinx()
        lines3 = []
        line_sum, = ax3.plot(ranks, sum_latencies, 'm-p', linewidth=2, markersize=8, label='Sum of Latencies')
        ax3.set_xlabel('Number of MPI Ranks')
        ax3.set_ylabel('Sum of Latencies (ns)', color='m')
        ax3.tick_params(axis='y', labelcolor='m')
        ax3.grid(True)
        lines3.append(line_sum)
        if len(ranks) >= 2:
            baseline_idx = ranks.index(4) if 4 in ranks else 0
            baseline_mean = means[baseline_idx]
            efficiency = [(baseline_mean / mean) if mean != 0 else np.inf for mean in means]
            line_eff, = ax3_twin.plot(ranks, efficiency, 'c-o', linewidth=2, label='Relative Efficiency')
            line_base = ax3_twin.axhline(y=1.0, color='r', linestyle='--', label='Baseline (4 ranks)')
            ax3_twin.set_ylabel('Relative Efficiency', color='c')
            ax3_twin.tick_params(axis='y', labelcolor='c')
            lines3.extend([line_eff, line_base])
        else:
            log.warning(f"Not enough data points for scaling efficiency plot ({title_prefix})")
        ax3.set_title(f'Sum & Relative Efficiency')
        if lines3:
            ax3.legend(handles=lines3, loc='best') # Reverted legend location as requested

        # Fourth subplot: Heatmap of Latency per Rank
        ax4 = plt.subplot(2, 2, 4)
        
        # Create the heatmap data matrix (rank_id x num_ranks_run)
        heatmap_data = np.full((max_rank_id + 1, len(ranks)), np.nan)
        for col_idx, r in enumerate(ranks):
            if all_latencies_list[col_idx] is not None:
                lats = all_latencies_list[col_idx]
                num_rows_to_fill = min(len(lats), max_rank_id + 1)
                heatmap_data[0:num_rows_to_fill, col_idx] = lats[0:num_rows_to_fill]

        # --- Swap Axes: Transpose data and adjust plotting --- 
        heatmap_data_transposed = heatmap_data.T # Transpose for swapped axes
        
        if not np.isnan(heatmap_data_transposed).all(): # Check if there's any valid data
            # extent=[xmin, xmax, ymin, ymax] -> [rank_id_min, rank_id_max, num_ranks_min, num_ranks_max]
            # Use origin='lower' for natural y-axis order (small ranks at bottom)
            extent = [-0.5, max_rank_id + 0.5, min(ranks) - 0.5 * (ranks[1]-ranks[0]) if len(ranks)>1 else min(ranks)-1, max(ranks) + 0.5 * (ranks[1]-ranks[0]) if len(ranks)>1 else max(ranks)+1] # Adjust extent for centering y-ticks
            im = ax4.imshow(heatmap_data_transposed, cmap='viridis', aspect='auto', 
                            interpolation='nearest', origin='lower',
                            extent=extent)
            
            # X-axis: Rank ID
            ax4.set_xlabel('Rank ID')
            if max_rank_id < 20: 
                ax4.set_xticks(np.arange(0, max_rank_id + 1, max(1, max_rank_id // 10)))
            # Add ticks for rank 0 and max_rank_id if not already covered
            # current_xticks = ax4.get_xticks()
            # if 0 not in current_xticks: ax4.set_xticks(np.append(current_xticks, 0))
            # if max_rank_id not in current_xticks: ax4.set_xticks(np.append(current_xticks, max_rank_id))
            # ax4.set_xticks(sorted(ax4.get_xticks())) # Sort ticks
            
            # Y-axis: Number of MPI Ranks
            ax4.set_ylabel('Number of MPI Ranks')
            ax4.set_yticks(ranks)
            ax4.set_yticklabels([str(r) for r in ranks])
            
            ax4.set_title('Latency Heatmap per Rank')
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax4, orientation='vertical', shrink=0.8)
            cbar.set_label('Latency (ns)')
        else:
            log.warning(f"No valid data for heatmap ({title_prefix})")
            ax4.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_xticks([])
            ax4.set_yticks([])

        # Adjust layout and save
        fig.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout considering suptitle
        plot_path = plots_dir / f'{title_prefix.lower()}_analysis.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        log.debug(f"Saved individual plots to {plot_path}")
        
    except Exception as e:
        log.exception(f"Failed to create individual {title_prefix} plots.")
        plt.close('all')

def create_memory_size_plots(sequential_data, interleaved_data, plots_dir):
    """Create plots showing MEDIAN, SUM, P99, and STD DEV latency as a function of memory size using a 2x2 subplot layout."""
    log.debug("Creating memory size vs Median/Sum/P99/StdDev latency plots (2x2 subplotted)...")
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D']
    
    for data_set, title_prefix in [(sequential_data, "Sequential"), (interleaved_data, "Interleaved")]:
        if not data_set:
            continue
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{title_prefix} Memory Latency vs Size', fontsize=16)
            
            # Assign axes according to the new layout
            ax_median = axes[0, 0]
            ax_sum = axes[0, 1]    # Sum is now Top Right
            ax_p99 = axes[1, 0]    # P99 is now Bottom Left
            ax_std = axes[1, 1]    # Std Dev is Bottom Right
            
            # Make Median (TL) and P99 (BL) share Y axis
            ax_median.get_shared_y_axes().join(ax_median, ax_p99)
            
            # Sum (TR) and Std Dev (BR) have independent Y axes
            
            sorted_data = sorted(data_set, key=lambda x: x['ranks'])
            lines, labels = [], []

            # --- Loop to calculate stats and plot data --- 
            for i, d in enumerate(sorted_data):
                if 'data_by_size' not in d or not isinstance(d.get('data_by_size'), dict): log.warning(...); continue
                sizes = sorted(d['data_by_size'].keys()); 
                if not sizes: log.warning(...); continue
                medians, p99s, sums, stds, valid_sizes = [], [], [], [], []
                for size in sizes:
                    if 'latencies' in d['data_by_size'][size] and len(d['data_by_size'][size]['latencies']) > 0:
                        latencies_arr = np.array(d['data_by_size'][size]['latencies'])
                        medians.append(np.median(latencies_arr)); p99s.append(np.percentile(latencies_arr, 99))
                        sums.append(np.sum(latencies_arr)); stds.append(np.std(latencies_arr))
                        valid_sizes.append(size)
                    else: log.warning(...)
                if not valid_sizes: log.warning(...); continue
                color = colors[i % len(colors)]; marker = markers[i % len(markers)]; label = f"{d['ranks']} ranks"
                
                # Plot according to new layout
                line_median, = ax_median.plot(valid_sizes, medians, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label)
                ax_sum.plot(valid_sizes, sums, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label) # Plot Sum Top Right
                ax_p99.plot(valid_sizes, p99s, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label) # Plot P99 Bottom Left
                ax_std.plot(valid_sizes, stds, marker=marker, color=color, linestyle='-', linewidth=2, markersize=8, label=label) # Plot Std Dev Bottom Right
                
                lines.append(line_median); labels.append(label)
            # --- End loop ---
            
            # --- Configure Axes based on new layout --- 
            # Top Row (Median, Sum)
            ax_median.set_title('Median Latency'); ax_median.set_ylabel('Latency (ns)'); ax_median.set_xscale('log', base=2); ax_median.tick_params(labelbottom=False); ax_median.grid(True, which="both", ls="-", alpha=0.3)
            ax_sum.set_title('Sum of Latencies'); ax_sum.set_ylabel('Total Latency (ns)'); ax_sum.set_xscale('log', base=2); ax_sum.tick_params(labelbottom=False); ax_sum.grid(True, which="both", ls="-", alpha=0.3)
            
            # Bottom Row (P99, Std Dev)
            ax_p99.set_title('P99 Latency (Higher than 99% of measurements)'); ax_p99.set_ylabel('Latency (ns)'); ax_p99.set_xlabel('Memory Size (MB)'); ax_p99.set_xscale('log', base=2); ax_p99.grid(True, which="both", ls="-", alpha=0.3)
            ax_std.set_title('Standard Deviation'); ax_std.set_ylabel('Latency Std Dev (ns)'); ax_std.set_xlabel('Memory Size (MB)'); ax_std.set_xscale('log', base=2); ax_std.grid(True, which="both", ls="-", alpha=0.3)

            # --- Legend and Layout --- 
            if lines:
                fig.legend(lines, labels, 
                           bbox_to_anchor=(0.97, 0.5), 
                           loc='center left', 
                           borderaxespad=0.,
                           fontsize='large')
            fig.tight_layout(rect=[0, 0, 0.88, 0.95]) 

            # --- Save Figure --- 
            if not ax_median.has_data() and not ax_p99.has_data() and not ax_sum.has_data() and not ax_std.has_data():
                 log.warning(f"No data was plotted for {title_prefix} ... Skipping save.")
                 plt.close(fig); continue
            # Filename reflects content, not layout
            plot_path = plots_dir / f'{title_prefix.lower()}_median_p99_sum_stddev_size_latency.png' 
            plt.savefig(plot_path, bbox_inches='tight', dpi=300); plt.close(fig)
            log.debug(f"Saved Median/Sum/P99/StdDev memory size vs latency plot (2x2 layout) to {plot_path}")
            
        except Exception as e:
            log.exception(f"Failed to create {title_prefix} Median/Sum/P99/StdDev memory size vs latency plot (2x2 layout).")
            plt.close('all')

def create_3d_surface_plots(sequential_data, interleaved_data, plots_dir):
    """Create 3D surface plots (Side and Flat views) with a central colorbar, showing MEDIAN latency."""
    log.debug("Creating combined 3D surface plots (Median Latency - Side/Flat Views with Central Colorbar)...")
    
    for data_set, title_prefix in [(sequential_data, "Sequential"), (interleaved_data, "Interleaved")]:
        if not data_set:
            continue
        
        try:
            # Check if data is suitable for 3D plotting
            if not all('data_by_size' in d and d['data_by_size'] for d in data_set):
                log.warning(f"Skipping 3D plot for {title_prefix} as some data points lack multi-size information.")
                continue
                
            all_ranks = sorted(list({d['ranks'] for d in data_set}))
            all_sizes_sets = [set(d['data_by_size'].keys()) for d in data_set if 'data_by_size' in d]
            if not all_sizes_sets:
                 log.warning(f"Skipping 3D plot for {title_prefix}: No valid size data found.")
                 continue
            common_sizes = set.intersection(*all_sizes_sets) if len(all_sizes_sets) > 1 else all_sizes_sets[0]
            if not common_sizes:
                log.warning(f"Skipping 3D plot for {title_prefix}: No common memory sizes found.")
                continue 
            all_sizes = sorted(list(common_sizes))
            
            if len(all_ranks) < 2 or len(all_sizes) < 2:
                log.warning(f"Skipping 3D plot for {title_prefix}: Need at least 2 ranks and 2 sizes.")
                continue
                
            X, Y = np.meshgrid(all_ranks, all_sizes)
            Z = np.full(X.shape, np.nan) # Use NaN for missing data points
            
            rank_map = {rank: i for i, rank in enumerate(all_ranks)}
            size_map = {size: i for i, size in enumerate(all_sizes)}
            
            for d in data_set:
                if 'data_by_size' not in d or d['ranks'] not in rank_map:
                    continue
                rank_idx = rank_map[d['ranks']]
                for size, stats in d['data_by_size'].items():
                    if size in size_map and 'latencies' in stats and len(stats['latencies']) > 0:
                        size_idx = size_map[size]
                        # Calculate and store MEDIAN latency
                        Z[size_idx, rank_idx] = np.median(stats['latencies']) 
            
            Z_masked = np.ma.masked_invalid(Z)
            if Z_masked.mask.all():
                 log.warning(f"Skipping 3D plot for {title_prefix}: All data points are missing/invalid after meshing.")
                 continue
                 
            # Create a single figure for both views
            fig = plt.figure(figsize=(22, 9)) # Made slightly wider for colorbar gap
            fig.suptitle(f'{title_prefix} Median Memory Latency Surface', fontsize=16, y=0.98)
            
            # Define subplot positions manually to create space for colorbar
            # [left, bottom, width, height]
            pos_left = [0.02, 0.05, 0.45, 0.85] 
            pos_right = [0.53, 0.05, 0.45, 0.85]
            pos_cbar = [0.48, 0.15, 0.015, 0.6] # Positioned between left and right

            # --- First Subplot: Side View ---
            ax_side = fig.add_subplot(1, 2, 1, projection='3d')
            ax_side.set_position(pos_left) # Set position manually
            surf = ax_side.plot_surface(X, Y, Z_masked, cmap='jet', 
                                      linewidth=0, antialiased=True, edgecolor='none')
            ax_side.set_xlabel('Number of Cores', fontsize=12); ax_side.set_ylabel('Memory Size (MB)', fontsize=12)
            ax_side.set_zlabel('Median Latency (ns)', fontsize=12); ax_side.set_title('Side View (elev=30, azim=120)', fontsize=14)
            ax_side.set_xticks(all_ranks); ax_side.set_xticklabels([str(r) for r in all_ranks])
            if len(all_sizes) > 1: ax_side.set_yticks(all_sizes); ax_side.set_yticklabels([str(s) for s in all_sizes])
            ax_side.grid(True); ax_side.view_init(elev=30, azim=120)
            
            # --- Second Subplot: Flat View ---
            ax_flat = fig.add_subplot(1, 2, 2, projection='3d')
            ax_flat.set_position(pos_right) # Set position manually
            ax_flat.plot_surface(X, Y, Z_masked, cmap='jet', 
                                 linewidth=0, antialiased=True, edgecolor='none')
            ax_flat.set_xlabel('Number of Cores', fontsize=12); ax_flat.set_ylabel('Memory Size (MB)', fontsize=12)
            ax_flat.set_zlabel('Median Latency (ns)', fontsize=12); ax_flat.set_title('Flat View (elev=10, azim=170)', fontsize=14)
            ax_flat.set_xticks(all_ranks); ax_flat.set_xticklabels([str(r) for r in all_ranks])
            if len(all_sizes) > 1: ax_flat.set_yticks(all_sizes); ax_flat.set_yticklabels([str(s) for s in all_sizes])
            ax_flat.grid(True); ax_flat.view_init(elev=10, azim=170)
            
            # Add a color bar in the dedicated axes between the plots
            cbar_ax = fig.add_axes(pos_cbar)
            fig.colorbar(surf, cax=cbar_ax, label='Median Memory Latency (ns)')
            
            # Removed plt.subplots_adjust and fig.tight_layout as positions are manual
            
            plot_path = plots_dir / f'{title_prefix.lower()}_3d_median_latency.png' # Updated filename
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            log.debug(f"Saved combined 3D Median plot (Side/Flat, Center Cbar) to {plot_path}")
                    
        except Exception as e:
            log.exception(f"Error creating combined 3D Median plot ({title_prefix}).")
            plt.close('all')

def create_fallback_2d_plot(data_set, title_prefix, plots_dir):
    """Create a 2D fallback visualization (using MEDIAN) when 3D plotting fails."""
    log.warning(f"Creating fallback 2D plot (Median Latency) for {title_prefix} due to 3D plot error.")
    try:
        plt.figure(figsize=(12, 8))
        
        latency_data = {}
        for d in data_set:
            if 'data_by_size' not in d:
                continue
            rank = d['ranks']
            latency_data[rank] = {}
            for size, stats in d['data_by_size'].items():
                if 'latencies' in stats and len(stats['latencies']) > 0:
                    # Calculate and store MEDIAN latency for fallback plot
                    latency_data[rank][size] = np.median(stats['latencies']) 
        
        if not latency_data:
             log.error("No data available for fallback 2D plot.")
             plt.close()
             return
             
        for rank in sorted(latency_data.keys()):
            if not latency_data[rank]: # Skip if no valid sizes for this rank
                 continue
            sizes = sorted(latency_data[rank].keys())
            latencies = [latency_data[rank][size] for size in sizes]
            if sizes: # Only plot if there is data
                 plt.plot(sizes, latencies, marker='o', linewidth=2, label=f'{rank} ranks')
        
        plt.xscale('log', base=2)
        plt.xlabel('Memory Size (MB)')
        plt.ylabel('Median Latency (ns)')
        plt.title(f'{title_prefix} Median Memory Latency by Rank Count (Fallback)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        
        plot_path = plots_dir / f'fallback_{title_prefix.lower()}_median_latency.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        log.debug(f"Saved fallback 2D Median plot to {plot_path}")
    except Exception as e:
        log.exception(f"Failed to create fallback 2D Median plot for {title_prefix}.")
        plt.close()

def create_size_comparison_plots(sequential_data, interleaved_data, plots_dir):
    """Create comparison plots for different memory sizes (multi-size mode)."""
    log.debug("Creating size comparison plots (multiple sizes)...")
    if not (interleaved_data and sequential_data):
        log.info("Skipping size comparison plots (need both sequential and interleaved multi-size data).")
        return
        
    try:
        # Find common memory sizes across all files of both types
        interleaved_sizes_list = [set(d['data_by_size'].keys()) for d in interleaved_data if 'data_by_size' in d]
        sequential_sizes_list = [set(d['data_by_size'].keys()) for d in sequential_data if 'data_by_size' in d]
        
        if not interleaved_sizes_list or not sequential_sizes_list:
            log.warning("Cannot create size comparison plots: Missing multi-size data for one or both types.")
            return
            
        common_interleaved = set.intersection(*interleaved_sizes_list) if len(interleaved_sizes_list) > 1 else interleaved_sizes_list[0]
        common_sequential = set.intersection(*sequential_sizes_list) if len(sequential_sizes_list) > 1 else sequential_sizes_list[0]
        common_sizes = sorted(list(common_interleaved.intersection(common_sequential)))
        
        if not common_sizes:
            log.warning("Cannot create size comparison plots: No common memory sizes found between sequential and interleaved data.")
            return
            
        log.debug(f"Creating comparison plots for common sizes: {common_sizes}")
        
        for size in common_sizes:
            try:
                plt.figure(figsize=(12, 8))
                
                seq_ranks, seq_means, seq_mins, seq_maxs = [], [], [], []
                for d in sorted(sequential_data, key=lambda x: x['ranks']):
                    if 'data_by_size' in d and size in d['data_by_size']:
                        stats = d['data_by_size'][size]
                        seq_ranks.append(d['ranks'])
                        seq_means.append(stats['mean'])
                        seq_mins.append(stats['min'])
                        seq_maxs.append(stats['max'])
                
                int_ranks, int_means, int_mins, int_maxs = [], [], [], []
                for d in sorted(interleaved_data, key=lambda x: x['ranks']):
                    if 'data_by_size' in d and size in d['data_by_size']:
                        stats = d['data_by_size'][size]
                        int_ranks.append(d['ranks'])
                        int_means.append(stats['mean'])
                        int_mins.append(stats['min'])
                        int_maxs.append(stats['max'])
                
                # Only plot if we have data for both types at this size
                if not seq_ranks or not int_ranks:
                    log.warning(f"Skipping comparison plot for size {size}MB: Missing data for sequential or interleaved.")
                    plt.close() # Close the unused figure
                    continue
                    
                plt.plot(seq_ranks, seq_means, 'b-', linewidth=2, label='Sequential Mean')
                plt.fill_between(seq_ranks, seq_mins, seq_maxs, color='blue', alpha=0.2)
                
                plt.plot(int_ranks, int_means, 'r-', linewidth=2, label='Interleaved Mean')
                plt.fill_between(int_ranks, int_mins, int_maxs, color='red', alpha=0.2)
                
                plt.xlabel('Number of MPI Ranks', fontsize=12)
                plt.ylabel('Latency (ns)', fontsize=12)
                plt.title(f'Memory Latency Comparison ({size} MB): Sequential vs Interleaved', fontsize=14)
                plt.legend(fontsize=11)
                plt.grid(True)
                plot_path = plots_dir / f'comparison_{size}MB.png'
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                log.debug(f"Saved size comparison plot to {plot_path}")
                
            except Exception as e_size:
                log.exception(f"Failed to create comparison plot for size {size}MB.")
                plt.close('all')
                
    except Exception as e:
        log.exception("An error occurred while creating size comparison plots.")
        plt.close('all')

def create_summary(data, job_dir, mode='single'):
    """Create a summary text file with statistical analysis."""
    log.info("Creating analysis summary file...")
    summary_file = job_dir / 'plots' / 'analysis_summary.txt'
    try:
        with open(summary_file, 'w') as f:
            f.write("NUMA Benchmark Analysis Summary\n")
            f.write("=============================\n\n")
            
            if mode == 'single':
                f.write("Statistical Analysis per Rank Count (Single Size):\n")
                f.write("-----------------------------------------------\n")
                for d in data:
                    allocation_type = "Interleaved" if d['is_interleaved'] else "Sequential"
                    f.write(f"\n{d['ranks']} ranks ({allocation_type} - {d.get('source_file', 'N/A')}):\n")
                    f.write(f"  Mean latency: {d['mean']:.2f} ns\n")
                    f.write(f"  Std dev: {d['std']:.2f} ns\n")
                    f.write(f"  Min latency: {d['min']:.2f} ns\n")
                    f.write(f"  Max latency: {d['max']:.2f} ns\n")
            else:  # mode == 'multiple'
                f.write("Statistical Analysis per Rank Count and Memory Size:\n")
                f.write("------------------------------------------------\n")
                for d in sorted(data, key=lambda x: x['ranks']):
                    allocation_type = "Interleaved" if d['is_interleaved'] else "Sequential"
                    f.write(f"\n{d['ranks']} ranks ({allocation_type} - {d.get('source_file', 'N/A')}):\n")
                    
                    if 'data_by_size' not in d or not d['data_by_size']:
                        f.write("  No multi-size data available.\n")
                        continue
                        
                    for size in sorted(d['data_by_size'].keys()):
                        stats = d['data_by_size'][size]
                        f.write(f"\n  Memory Size: {size} MB\n")
                        f.write(f"    Mean latency: {stats['mean']:.2f} ns\n")
                        f.write(f"    Std dev: {stats['std']:.2f} ns\n")
                        f.write(f"    Min latency: {stats['min']:.2f} ns\n")
                        f.write(f"    Max latency: {stats['max']:.2f} ns\n")
                        f.write(f"    Range: {stats['max'] - stats['min']:.2f} ns\n")
                        # Calculate CoV safely, avoiding division by zero
                        mean_val = stats['mean']
                        if mean_val != 0:
                            cov = (stats['std'] / mean_val) * 100
                            f.write(f"    Coefficient of variation: {cov:.2f}%\n")
                        else:
                            f.write("    Coefficient of variation: N/A (mean is zero)\n")
        log.info(f"Successfully created summary file: {summary_file}")
    except IOError as e:
        log.error(f"Error writing summary file {summary_file}: {e}")
    except Exception as e:
        log.exception("An unexpected error occurred during summary creation.")

def main():
    # Parse arguments first
    args = parse_args()
    
    # Set logging level based on verbosity argument HERE
    if args.verbose:
        log.setLevel(logging.DEBUG)
        log.info("Verbose logging enabled.")
    else:
        log.setLevel(logging.INFO)
        
    job_dir = args.job_dir
    plots_dir = job_dir / 'plots'
    # Create plots directory early
    try:
         plots_dir.mkdir(exist_ok=True)
         log.debug(f"Ensured plots directory exists: {plots_dir}")
    except OSError as e:
         log.error(f"Could not create plots directory {plots_dir}: {e}")
         sys.exit(1)
    
    # Parse data and detect mode, passing verbose flag
    data, detected_mode = parse_csv_files(job_dir, args.mode, verbose=args.verbose)
    
    # Create plots
    create_plots(data, plots_dir, detected_mode, args.no_3d)
    
    # Create summary
    create_summary(data, job_dir, detected_mode)
    
    log.info(f"Analysis complete. Results saved in {job_dir}")
    log.info(f"Plots saved in {plots_dir}")
    log.info(f"Summary saved in {plots_dir}/analysis_summary.txt")

if __name__ == "__main__":
    main() 