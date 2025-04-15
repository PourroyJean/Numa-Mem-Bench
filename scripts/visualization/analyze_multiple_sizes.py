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

def parse_csv_files(job_dir):
    """Parse all CSV files in the job directory and extract relevant data for multiple memory sizes."""
    print(f"Starting to parse CSV files in {job_dir}")
    data = []
    job_dir = Path(job_dir)
    
    if not job_dir.exists():
        print(f"Error: Directory {job_dir} does not exist")
        sys.exit(1)
    
    # Get all CSV files with benchmark results
    csv_files = [f for f in job_dir.glob("*.csv") if f.stem.startswith(('sequential_', 'interleaved_'))]
    print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        print(f"Error: No valid CSV files found in {job_dir}")
        sys.exit(1)
    
    # Parse each file
    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            if len(df.columns) < 2:
                print(f"Error: Invalid CSV format in {csv_file}")
                sys.exit(1)
            
            # Check if the file contains multiple memory sizes
            if 'size (MB)' not in df.columns:
                print(f"Error: File {csv_file.name} does not have 'size (MB)' column")
                sys.exit(1)
                
            # Extract number of ranks from filename
            match = re.search(r'(\d+)ranks', csv_file.stem)
            if not match:
                print(f"Error: Could not find ranks number in filename {csv_file.name}")
                sys.exit(1)
            ranks = int(match.group(1))
            print(f"Found {ranks} ranks in filename")
            
            # Process data for each memory size
            memory_sizes = df['size (MB)'].values
            data_by_size = {}
            
            for i, size in enumerate(memory_sizes):
                latencies = df.iloc[i, 1:].values  # All columns except 'size (MB)'
                data_by_size[size] = {
                    'latencies': latencies,
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                }
            
            data.append({
                'ranks': ranks,
                'is_interleaved': 'interleaved' in csv_file.stem,
                'memory_sizes': memory_sizes,
                'data_by_size': data_by_size
            })
            print(f"Successfully processed {csv_file.name} with {len(memory_sizes)} memory sizes")
        except Exception as e:
            print(f"Error parsing {csv_file}: {e}")
            sys.exit(1)
    
    return sorted(data, key=lambda x: x['ranks'])

def create_memory_size_plots(data, plots_dir):
    """Create plots showing latency as a function of memory size for each rank count."""
    print("\nCreating memory size plots...")
    
    # Separate interleaved and non-interleaved data
    interleaved_data = [d for d in data if d['is_interleaved']]
    sequential_data = [d for d in data if not d['is_interleaved']]
    
    # Plot for each data set
    for data_set, title_prefix in [(sequential_data, "Sequential"), (interleaved_data, "Interleaved")]:
        if not data_set:
            continue
            
        plt.figure(figsize=(12, 8))
        for d in data_set:
            sizes = sorted(d['data_by_size'].keys())
            means = [d['data_by_size'][size]['mean'] for size in sizes]
            plt.plot(sizes, means, marker='o', label=f"{d['ranks']} ranks")
        
        plt.xscale('log', base=2)  # Log scale for memory sizes
        plt.xlabel('Memory Size (MB)')
        plt.ylabel('Mean Latency (ns)')
        plt.title(f'{title_prefix} Memory Latency vs Size')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.savefig(plots_dir / f'{title_prefix.lower()}_size_latency.png')
        plt.close()
        
        # Also create min/max plots with shaded areas
        plt.figure(figsize=(12, 8))
        for d in data_set:
            sizes = sorted(d['data_by_size'].keys())
            means = [d['data_by_size'][size]['mean'] for size in sizes]
            mins = [d['data_by_size'][size]['min'] for size in sizes]
            maxs = [d['data_by_size'][size]['max'] for size in sizes]
            
            plt.plot(sizes, means, marker='o', label=f"{d['ranks']} ranks")
            plt.fill_between(sizes, mins, maxs, alpha=0.2)
        
        plt.xscale('log', base=2)
        plt.xlabel('Memory Size (MB)')
        plt.ylabel('Latency (ns)')
        plt.title(f'{title_prefix} Memory Latency vs Size (with Min/Max)')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.savefig(plots_dir / f'{title_prefix.lower()}_size_latency_minmax.png')
        plt.close()

def create_3d_surface_plot(data, plots_dir):
    """Create 3D surface plots showing latency as a function of memory size and number of ranks with different viewing angles."""
    print("\nCreating 3D surface plots...")
    
    # Separate interleaved and non-interleaved data
    interleaved_data = [d for d in data if d['is_interleaved']]
    sequential_data = [d for d in data if not d['is_interleaved']]
    
    # Define multiple viewing angles for different perspectives
    view_angles = [
        {"elev": 30, "azim": 120, "suffix": "side"},  # Side view (default)
        {"elev": 45, "azim": 225, "suffix": "corner"},  # Corner view
        {"elev": 10, "azim": 170, "suffix": "flat"}   # Flatter view to see patterns
    ]
    
    for data_set, title_prefix in [(sequential_data, "Sequential"), (interleaved_data, "Interleaved")]:
        if not data_set:
            continue
        
        try:
            # Extract all unique ranks and memory sizes
            all_ranks = sorted(list({d['ranks'] for d in data_set}))
            all_sizes = sorted(list({size for d in data_set for size in d['data_by_size'].keys()}))
            
            # Create mesh grid for 3D surface
            X, Y = np.meshgrid(all_ranks, all_sizes)
            Z = np.zeros(X.shape)
            
            # Fill Z matrix with mean latency values
            for i, size in enumerate(all_sizes):
                for j, ranks in enumerate(all_ranks):
                    matching_data = [d for d in data_set if d['ranks'] == ranks]
                    if matching_data and size in matching_data[0]['data_by_size']:
                        Z[i, j] = matching_data[0]['data_by_size'][size]['mean']
            
            # Create a masked array to handle missing data
            Z_masked = np.ma.masked_where(Z == 0, Z)
            
            # Generate plots for each viewing angle
            for view in view_angles:
                # Create 3D plot with the current viewing angle
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Use plot_surface with careful settings
                surf = ax.plot_surface(X, Y, Z_masked, cmap='viridis', 
                                      linewidth=0, antialiased=True, edgecolor='none')
                
                # Add labels and title
                ax.set_xlabel('Number of Cores', fontsize=12)
                ax.set_ylabel('Memory Size (MB)', fontsize=12)
                ax.set_zlabel('Memory Latency (ns)', fontsize=12)
                ax.set_title(f'{title_prefix} Memory Latency Surface', fontsize=14)
                
                # Manually set tick positions for better visualization
                if len(all_sizes) > 1:
                    ax.set_yticks(all_sizes)
                    ax.set_yticklabels([str(s) for s in all_sizes])
                
                # Add gridlines for better perspective
                ax.grid(True)
                
                # Add color bar
                cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
                cbar.set_label('Memory Latency (ns)', fontsize=12)
                
                # Set the current view angle
                ax.view_init(elev=view["elev"], azim=view["azim"])
                
                # Save the figure with the appropriate suffix for the view angle
                plt.savefig(plots_dir / f'3d_{title_prefix.lower()}_latency_{view["suffix"]}.png', 
                           bbox_inches='tight', dpi=300)
                plt.close()
            
            # Create an alternative 2D heatmap visualization for the same data
            plt.figure(figsize=(12, 8))
            
            # Use pcolormesh which can handle a wider range of data
            pcm = plt.pcolormesh(X, Y, Z_masked, cmap='viridis', shading='auto')
            plt.colorbar(pcm, label='Memory Latency (ns)')
            
            plt.xlabel('Number of Cores')
            plt.ylabel('Memory Size (MB)')
            plt.title(f'{title_prefix} Memory Latency Heatmap')
            plt.yscale('log', base=2)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            plt.savefig(plots_dir / f'heatmap_{title_prefix.lower()}_latency.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating 3D plot for {title_prefix} data: {e}")
            print("Creating fallback 2D visualization instead...")
            
            # Fallback to a more reliable 2D visualization if 3D fails
            try:
                plt.figure(figsize=(12, 8))
                
                # Create a latency matrix organized by ranks and sizes
                latency_data = {}
                for d in data_set:
                    rank = d['ranks']
                    latency_data[rank] = {}
                    for size in sorted(d['data_by_size'].keys()):
                        latency_data[rank][size] = d['data_by_size'][size]['mean']
                
                # Plot latency curves for each rank count
                for rank in sorted(latency_data.keys()):
                    sizes = sorted(latency_data[rank].keys())
                    latencies = [latency_data[rank][size] for size in sizes]
                    plt.plot(sizes, latencies, marker='o', linewidth=2, label=f'{rank} ranks')
                
                plt.xscale('log', base=2)
                plt.xlabel('Memory Size (MB)')
                plt.ylabel('Latency (ns)')
                plt.title(f'{title_prefix} Memory Latency by Rank Count')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                
                plt.savefig(plots_dir / f'fallback_{title_prefix.lower()}_latency.png', 
                           bbox_inches='tight', dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error creating fallback plot: {e}")

def create_size_comparison_plots(data, plots_dir):
    """Create plots comparing different rank counts for each memory size."""
    print("\nCreating size comparison plots...")
    
    # Separate interleaved and non-interleaved data
    interleaved_data = [d for d in data if d['is_interleaved']]
    sequential_data = [d for d in data if not d['is_interleaved']]
    
    # If we have both types, create a comparison plot
    if interleaved_data and sequential_data:
        # Get common memory sizes
        interleaved_sizes = set()
        for d in interleaved_data:
            interleaved_sizes.update(d['data_by_size'].keys())
            
        sequential_sizes = set()
        for d in sequential_data:
            sequential_sizes.update(d['data_by_size'].keys())
            
        common_sizes = sorted(list(interleaved_sizes.intersection(sequential_sizes)))
        
        # For each memory size, plot sequential vs interleaved
        for size in common_sizes:
            plt.figure(figsize=(12, 8))
            
            # Extract data for sequential
            seq_ranks = []
            seq_means = []
            seq_mins = []
            seq_maxs = []
            
            for d in sorted(sequential_data, key=lambda x: x['ranks']):
                if size in d['data_by_size']:
                    seq_ranks.append(d['ranks'])
                    seq_means.append(d['data_by_size'][size]['mean'])
                    seq_mins.append(d['data_by_size'][size]['min'])
                    seq_maxs.append(d['data_by_size'][size]['max'])
            
            # Extract data for interleaved
            int_ranks = []
            int_means = []
            int_mins = []
            int_maxs = []
            
            for d in sorted(interleaved_data, key=lambda x: x['ranks']):
                if size in d['data_by_size']:
                    int_ranks.append(d['ranks'])
                    int_means.append(d['data_by_size'][size]['mean'])
                    int_mins.append(d['data_by_size'][size]['min'])
                    int_maxs.append(d['data_by_size'][size]['max'])
            
            # Plot data
            plt.plot(seq_ranks, seq_means, 'b-', linewidth=2, label='Sequential Mean')
            plt.fill_between(seq_ranks, seq_mins, seq_maxs, color='blue', alpha=0.2)
            
            plt.plot(int_ranks, int_means, 'r-', linewidth=2, label='Interleaved Mean')
            plt.fill_between(int_ranks, int_mins, int_maxs, color='red', alpha=0.2)
            
            plt.xlabel('Number of MPI Ranks', fontsize=12)
            plt.ylabel('Latency (ns)', fontsize=12)
            plt.title(f'Memory Latency Comparison ({size} MB): Sequential vs Interleaved', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True)
            plt.savefig(plots_dir / f'comparison_{size}MB.png', bbox_inches='tight')
            plt.close()

def create_summary(data, job_dir):
    """Create a summary text file with statistical analysis for multiple memory sizes."""
    plots_dir = job_dir / 'plots'
    with open(plots_dir / 'analysis_summary.txt', 'w') as f:
        f.write("NUMA Benchmark Analysis Summary - Multiple Memory Sizes\n")
        f.write("===================================================\n\n")
        
        f.write("Statistical Analysis per Rank Count and Memory Size:\n")
        f.write("------------------------------------------------\n")
        
        for d in sorted(data, key=lambda x: x['ranks']):
            allocation_type = "Interleaved" if d['is_interleaved'] else "Sequential"
            f.write(f"\n{d['ranks']} ranks ({allocation_type}):\n")
            
            for size in sorted(d['data_by_size'].keys()):
                stats = d['data_by_size'][size]
                f.write(f"\n  Memory Size: {size} MB\n")
                f.write(f"    Mean latency: {stats['mean']:.2f} ns\n")
                f.write(f"    Std dev: {stats['std']:.2f} ns\n")
                f.write(f"    Min latency: {stats['min']:.2f} ns\n")
                f.write(f"    Max latency: {stats['max']:.2f} ns\n")
                f.write(f"    Range: {stats['max'] - stats['min']:.2f} ns\n")
                f.write(f"    Coefficient of variation: {(stats['std'] / stats['mean']) * 100:.2f}%\n")

def main():
    print("Starting multi-size analysis script...")
    if len(sys.argv) != 2:
        print("Usage: python analyze_multiple_sizes.py <job_directory>")
        sys.exit(1)
    
    job_dir = Path(sys.argv[1])
    plots_dir = job_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Parse data
    data = parse_csv_files(job_dir)
    
    # Create plots
    create_memory_size_plots(data, plots_dir)
    create_3d_surface_plot(data, plots_dir)
    create_size_comparison_plots(data, plots_dir)
    
    # Create summary
    create_summary(data, job_dir)
    
    print(f"\nAnalysis complete. Results saved in {job_dir}")
    print(f"Plots saved in {plots_dir}")
    print(f"Summary saved in {plots_dir}/analysis_summary.txt")

if __name__ == "__main__":
    main() 