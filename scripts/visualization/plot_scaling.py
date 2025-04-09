#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats

def load_numa_mapping(mapping_file):
    """Load NUMA mapping information from CSV file."""
    try:
        mapping_df = pd.read_csv(mapping_file)
        # Group ranks by NUMA node
        numa_groups = {}
        for numa_node in mapping_df['cpu_numa'].unique():
            ranks = mapping_df[mapping_df['cpu_numa'] == numa_node]['rank'].tolist()
            numa_groups[f'NUMA {numa_node}'] = ranks
        return numa_groups
    except Exception as e:
        print(f"Error loading NUMA mapping file: {e}")
        return None

def analyze_by_size(df):
    """Analyze statistics for each memory size."""
    print("\n=== Statistics by Memory Size ===")
    print("Format: mean ± std (min [rank] - max [rank])")
    
    sizes = df['size (MB)']
    rank_columns = [col for col in df.columns if col != 'size (MB)']
    
    for idx, size in enumerate(sizes):
        # Get all latencies for this size
        latencies = df.iloc[idx][rank_columns].values
        
        # Calculate basic statistics
        mean = np.mean(latencies)
        std = np.std(latencies)
        
        # Find min and max values with their ranks
        min_val = np.min(latencies)
        max_val = np.max(latencies)
        min_rank = rank_columns[np.argmin(latencies)]
        max_rank = rank_columns[np.argmax(latencies)]
        
        # Identify outliers (values outside 1.5 * IQR)
        Q1 = np.percentile(latencies, 25)
        Q3 = np.percentile(latencies, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        low_outliers = latencies[latencies < lower_bound]
        high_outliers = latencies[latencies > upper_bound]
        
        print(f"\nSize {size:4} MB:")
        print(f"  {mean:6.2f} ± {std:6.2f} ({min_val:6.2f} [{min_rank}] - {max_val:6.2f} [{max_rank}])")
        
        if len(low_outliers) > 0:
            print(f"  Low outliers ({len(low_outliers)}): {', '.join(f'{x:.2f}' for x in sorted(low_outliers))}")
        if len(high_outliers) > 0:
            print(f"  High outliers ({len(high_outliers)}): {', '.join(f'{x:.2f}' for x in sorted(high_outliers))}")

def analyze_by_numa(df, numa_groups):
    """Analyze statistics grouped by NUMA nodes."""
    if not numa_groups:
        print("\nWarning: No NUMA mapping information available")
        return
        
    print("\n=== Statistics by NUMA Node ===")
    
    sizes = df['size (MB)']
    rank_columns = [col for col in df.columns if col != 'size (MB)']
    
    for size_idx, size in enumerate(sizes):
        print(f"\nSize {size:4} MB:")
        
        for numa_name, ranks in numa_groups.items():
            # Get ranks that exist in our data
            valid_ranks = [str(r) for r in ranks if str(r) in rank_columns]
            if not valid_ranks:
                continue
                
            # Get latencies for this NUMA group
            latencies = df.iloc[size_idx][valid_ranks].values
            
            # Calculate statistics
            mean = np.mean(latencies)
            std = np.std(latencies)
            min_val = np.min(latencies)
            max_val = np.max(latencies)
            
            print(f"  {numa_name:8}: {mean:6.2f} ± {std:6.2f} ({min_val:6.2f} - {max_val:6.2f})")

def analyze_rank_performance(df, numa_groups):
    """Analyze overall performance of each rank across all sizes."""
    print("\n=== Overall Rank Performance ===")
    
    rank_columns = [col for col in df.columns if col != 'size (MB)']
    
    # Calculate average latency for each rank across all sizes
    rank_scores = {}
    for rank in rank_columns:
        avg_latency = df[rank].mean()
        min_latency = df[rank].min()
        max_latency = df[rank].max()
        rank_scores[rank] = {
            'avg': avg_latency,
            'min': min_latency,
            'max': max_latency
        }
    
    # Sort ranks by average latency
    sorted_ranks = sorted(rank_scores.items(), key=lambda x: x[1]['avg'])
    
    # Print top 3 best performers
    print("\nTop 3 Best Performing Ranks:")
    for rank, scores in sorted_ranks[:3]:
        numa_node = None
        for numa_name, ranks in numa_groups.items():
            if int(rank) in ranks:
                numa_node = numa_name
                break
        numa_info = f" ({numa_node})" if numa_node else ""
        print(f"  Rank {rank:>2}: {scores['avg']:6.2f} ns avg (range: {scores['min']:6.2f} - {scores['max']:6.2f}){numa_info}")
    
    # Print top 3 worst performers
    print("\nTop 3 Worst Performing Ranks:")
    for rank, scores in sorted_ranks[-3:]:
        numa_node = None
        for numa_name, ranks in numa_groups.items():
            if int(rank) in ranks:
                numa_node = numa_name
                break
        numa_info = f" ({numa_node})" if numa_node else ""
        print(f"  Rank {rank:>2}: {scores['avg']:6.2f} ns avg (range: {scores['min']:6.2f} - {scores['max']:6.2f}){numa_info}")

def parse_rank_list(rank_str):
    """Parse a string like '1,3-5,6' into a list of ranks [1,3,4,5,6]."""
    if not rank_str:
        return None
        
    ranks = set()
    parts = rank_str.split(',')
    
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            ranks.update(range(start, end + 1))
        else:
            ranks.add(int(part))
    
    return sorted(list(ranks))

def filter_rank_columns(df, selected_ranks):
    """Filter DataFrame to keep only selected ranks."""
    if selected_ranks is None:
        return df
        
    # Convert selected ranks to strings to match column names
    selected_ranks = [str(r) for r in selected_ranks]
    # Keep 'size (MB)' column and selected rank columns
    columns_to_keep = ['size (MB)'] + [col for col in df.columns if col != 'size (MB)' and col in selected_ranks]
    return df[columns_to_keep]

def plot_graphs(df, numa_groups, output_file=None):
    """Create a single figure with two subplots: individual ranks and NUMA groups."""
    print("\nCreating plots...")
    
    # Create a single figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    
    # First subplot: Individual ranks
    ax1 = plt.subplot(211)
    sizes = df['size (MB)']
    rank_columns = [col for col in df.columns if col != 'size (MB)']
    
    # Plot each rank's data
    print("Plotting data for each rank...")
    lines = []  # Store line objects for legend
    labels = []  # Store labels for legend
    
    for rank in rank_columns:
        # Get NUMA node for this rank if available
        numa_node = None
        if numa_groups:
            for numa_name, ranks in numa_groups.items():
                if int(rank) in ranks:
                    numa_node = numa_name
                    break
        
        # Create label with NUMA information if available
        label = f'Rank {rank}'
        if numa_node:
            label += f' ({numa_node})'
            
        # Plot the line without adding it to the legend
        line = ax1.plot(sizes, df[rank], marker='o')[0]
        lines.append(line)
        labels.append(label)
    
    # Select a subset of ranks for the legend (evenly distributed)
    if len(lines) > 16:
        # Take the first 16 ranks
        legend_indices = list(range(16))
        # Create legend with subset
        ax1.legend([lines[i] for i in legend_indices],
                  [labels[i] for i in legend_indices],
                  bbox_to_anchor=(1.05, 1), loc='upper left',
                  title=f"Showing first 16 of {len(lines)} ranks")
    else:
        # If 16 or fewer ranks, show all in legend
        ax1.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Customize the first subplot
    ax1.set_xscale('log', base=2)  # Logarithmic scale for x-axis
    ax1.grid(True, which="both", ls="-", alpha=0.2)  # Add grid
    ax1.set_xlabel('Memory Size (MB)')
    ax1.set_ylabel('Latency (ns)')
    ax1.set_title('Memory Latency vs Size for Different Ranks')
    
    # Second subplot: NUMA groups
    if numa_groups:
        print("Plotting NUMA group data...")
        ax2 = plt.subplot(212)
        
        # Plot data for each NUMA node
        for numa_name, ranks in numa_groups.items():
            # Get ranks that exist in our data
            valid_ranks = [str(r) for r in ranks if str(r) in rank_columns]
            if not valid_ranks:
                continue
                
            # Calculate mean and std for each size
            means = []
            stds = []
            for size_idx in range(len(sizes)):
                latencies = df.iloc[size_idx][valid_ranks].values
                means.append(np.mean(latencies))
                stds.append(np.std(latencies))
            
            # Plot with error bars
            ax2.errorbar(sizes, means, yerr=stds, 
                        marker='o', 
                        label=numa_name,
                        capsize=5)  # Add caps to error bars
        
        # Customize the second subplot
        ax2.set_xscale('log', base=2)  # Logarithmic scale for x-axis
        ax2.grid(True, which="both", ls="-", alpha=0.2)  # Add grid
        ax2.set_xlabel('Memory Size (MB)')
        ax2.set_ylabel('Average Latency (ns)')
        ax2.set_title('Memory Latency by NUMA Node')
        ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or display
    if output_file:
        print(f"\nSaving plots to {output_file}...")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graphs saved successfully")
        plt.close()
    else:
        print("\nDisplaying interactive plots...")
        plt.show()

def verify_rank_consistency(df, numa_groups):
    """Verify that all ranks in the input file are present in the NUMA mapping."""
    if not numa_groups:
        print("\nError: No NUMA mapping information available")
        sys.exit(1)
        
    # Get all ranks from input file
    input_ranks = set(int(col) for col in df.columns if col != 'size (MB)')
    
    # Get all ranks from NUMA mapping
    numa_ranks = set()
    for ranks in numa_groups.values():
        numa_ranks.update(ranks)
    
    # Check for missing ranks
    missing_in_numa = input_ranks - numa_ranks
    missing_in_input = numa_ranks - input_ranks
    
    if missing_in_numa:
        print("\nError: The following ranks from the input file are not mapped to NUMA nodes:")
        print(f"  {sorted(missing_in_numa)}")
        sys.exit(1)
        
    if missing_in_input:
        print("\nError: The following ranks from the NUMA mapping are not present in the input file:")
        print(f"  {sorted(missing_in_input)}")
        sys.exit(1)
        
    return True

def plot_latency_graph(csv_file, output_file=None, rank_filter=None, mapping_file=None):
    print(f"\n=== Memory Latency Graph Generator ===")
    print(f"Input CSV file: {csv_file}")
    if output_file:
        print(f"Output PNG file: {output_file}")
    else:
        print("No output file specified - will display graph interactively")
    
    if rank_filter:
        selected_ranks = parse_rank_list(rank_filter)
        print(f"Filtering ranks: {selected_ranks}")
    else:
        selected_ranks = None
    
    # Load NUMA mapping information
    numa_groups = None
    if mapping_file:
        print(f"\nLoading NUMA mapping from: {mapping_file}")
        numa_groups = load_numa_mapping(mapping_file)
        if not numa_groups:
            print("Error: Failed to load NUMA mapping")
            sys.exit(1)
        print("NUMA mapping loaded successfully")
    
    # Read the CSV file
    print("\nReading CSV file...")
    df = pd.read_csv(csv_file)
    
    # Filter ranks if specified
    df = filter_rank_columns(df, selected_ranks)
    
    # Get the size column and all other columns (ranks)
    sizes = df['size (MB)']
    rank_columns = [col for col in df.columns if col != 'size (MB)']
    print(f"Analyzing {len(rank_columns)} ranks: {', '.join(rank_columns)}")
    
    # Verify rank consistency
    if mapping_file:
        verify_rank_consistency(df, numa_groups)
    
    # Perform statistical analysis
    analyze_by_size(df)
    analyze_by_numa(df, numa_groups)
    analyze_rank_performance(df, numa_groups)
    
    # Create both plots on the same figure
    plot_graphs(df, numa_groups, output_file)
    
    print("\n=== Done ===")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and plot memory latency data')
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('--output', help='Output PNG file')
    parser.add_argument('--ranks', help='Filter specific ranks (e.g., "1,3-5,6")')
    parser.add_argument('--mapping', help='NUMA mapping CSV file')
    
    args = parser.parse_args()
    
    plot_latency_graph(args.csv_file, args.output, args.ranks, args.mapping) 