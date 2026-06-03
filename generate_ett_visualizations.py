import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set up paths
base_path = "/Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main"
data_path = os.path.join(base_path, "COTN/ETT-small")
output_path = os.path.join(base_path, "Data Visualization")

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Dataset files
datasets = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]

# Parameters
days_per_segment = 10
hours_per_day = 24
rows_per_segment = days_per_segment * hours_per_day  # 240 rows
num_segments = 10

# Color palette for 10 segments
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
          '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#A8E6CF']

# Set random seed for reproducibility
np.random.seed(42)

# Process each dataset
for dataset_idx, dataset_name in enumerate(datasets):
    print(f"Processing {dataset_name}...")

    # Load data
    file_path = os.path.join(data_path, dataset_name)
    df = pd.read_csv(file_path)

    # Get total number of rows
    total_rows = len(df)
    max_start_idx = total_rows - rows_per_segment

    # Generate random non-consecutive start indices
    # We'll ensure they're spaced apart by at least 2 * rows_per_segment
    min_spacing = 2 * rows_per_segment
    start_indices = []

    while len(start_indices) < num_segments:
        candidate = np.random.randint(0, max_start_idx + 1)

        # Check if this candidate is far enough from existing indices
        is_valid = True
        for existing_idx in start_indices:
            if abs(candidate - existing_idx) < min_spacing:
                is_valid = False
                break

        if is_valid:
            start_indices.append(candidate)

    start_indices.sort()

    # Generate plots for each segment (OT column only)
    for seg_idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + rows_per_segment
        segment_data = df.iloc[start_idx:end_idx]

        # Create figure with white background and no grid
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(False)

        # Plot only OT column
        ax.plot(range(len(segment_data)), segment_data['OT'].values,
               color='orange', linewidth=2)

        # Set labels and title
        start_date = segment_data.iloc[0, 0]
        end_date = segment_data.iloc[-1, 0]
        ax.set_xlabel('Hours', fontsize=10)
        ax.set_ylabel('OT Value', fontsize=10)
        ax.set_title(f'{dataset_name} - Segment {seg_idx+1} ({start_date} to {end_date})',
                    fontsize=12, fontweight='bold')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save figure
        output_filename = f"{dataset_name.replace('.csv', '')}_segment_{seg_idx+1:02d}.png"
        output_filepath = os.path.join(output_path, output_filename)
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=100, facecolor='white', edgecolor='none')
        plt.close()

        print(f"  Saved: {output_filename}")

    # Generate comparison plot with all 10 segments in one figure
    print(f"  Creating comparison plot for {dataset_name}...")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)

    for seg_idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + rows_per_segment
        segment_data = df.iloc[start_idx:end_idx]

        ax.plot(range(len(segment_data)), segment_data['OT'].values,
               color=colors[seg_idx], linewidth=1.5, label=f'Segment {seg_idx+1}',
               alpha=0.8)

    ax.set_xlabel('Hours', fontsize=11)
    ax.set_ylabel('OT Value', fontsize=11)
    ax.set_title(f'{dataset_name} - All 10 Segments Comparison', fontsize=13, fontweight='bold')

    # Place legend outside the plot area to avoid overlapping with lines
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, frameon=True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save comparison figure
    comparison_filename = f"{dataset_name.replace('.csv', '')}_comparison_all_segments.png"
    comparison_filepath = os.path.join(output_path, comparison_filename)
    plt.tight_layout()
    plt.savefig(comparison_filepath, dpi=100, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"  Saved: {comparison_filename}")

print(f"\nAll visualizations saved to: {output_path}")
print(f"Individual segment images: {len(datasets) * num_segments}")
print(f"Comparison images: {len(datasets)}")
print(f"Total images generated: {len(datasets) * num_segments + len(datasets)}")
