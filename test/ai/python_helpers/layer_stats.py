import matplotlib.pyplot as plt
import numpy as np
import re

def parse_layer_diffs(filename):
    """
    Parse the layer file and extract Lower Diff and Upper Diff values
    for each AFFINE layer (skipping the first two AFFINE layers in each example).
    Consolidates results from multiple examples.
    
    Returns:
        dict: {layer_num: {'lower_diff': [...], 'upper_diff': [...]}}
    """
    layer_data = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # First pass: identify which layer numbers correspond to first two AFFINE layers
    # by looking at the first example
    first_two_affine_layers = []
    current_layer = None
    i = 0
    
    while i < len(lines) and len(first_two_affine_layers) < 2:
        line = lines[i].strip()
        
        if line.startswith('LAYER'):
            parts = line.split()
            current_layer = int(parts[1])
            i += 1
            continue
        
        if line.startswith('Type:') and current_layer is not None:
            current_type = line.split(':')[1].strip()
            if 'AFFINE' in current_type:
                first_two_affine_layers.append(current_layer)
            i += 1
            continue
        
        # Stop if we hit a new example (indicated by PREDICTED CLASS or example markers)
        if 'EXAMPLE' in line or 'PREDICTED CLASS' in line:
            if len(first_two_affine_layers) >= 2:
                break
        
        i += 1
    
    print(f"First two AFFINE layer numbers to skip: {first_two_affine_layers}")
    
    # Second pass: extract data from all layers NOT in first two
    current_layer = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for layer header
        if line.startswith('LAYER'):
            parts = line.split()
            current_layer = int(parts[1])
            i += 1
            continue
        
        # Check for layer type
        if line.startswith('Type:') and current_layer is not None:
            current_type = line.split(':')[1].strip()
            
            # Process AFFINE layers that are NOT in the first two
            if 'AFFINE' in current_type:
                if current_layer not in first_two_affine_layers:
                    if current_layer not in layer_data:
                        layer_data[current_layer] = {'lower_diff': [], 'upper_diff': []}
                        print(f"Will process AFFINE layer {current_layer}")
            i += 1
            continue
        
        # Extract Lower Diff - numbers are on the NEXT line
        if line.startswith('Lower Diff:') and current_layer in layer_data:
            i += 1  # Move to next line
            if i < len(lines):
                numbers_str = lines[i].strip()
                if numbers_str:  # Check if line is not empty
                    numbers = [float(x) for x in numbers_str.split()]
                    # Extend to consolidate across examples
                    layer_data[current_layer]['lower_diff'].extend(numbers)
            continue
        
        # Extract Upper Diff - numbers are on the NEXT line
        if line.startswith('Upper Diff:') and current_layer in layer_data:
            i += 1  # Move to next line
            if i < len(lines):
                numbers_str = lines[i].strip()
                if numbers_str:  # Check if line is not empty
                    numbers = [float(x) for x in numbers_str.split()]
                    # Extend to consolidate across examples
                    layer_data[current_layer]['upper_diff'].extend(numbers)
            continue
        
        i += 1
    
    # Print summary
    print(f"\nFound data for {len(layer_data)} AFFINE layers:")
    for layer_num in sorted(layer_data.keys()):
        lower_count = len(layer_data[layer_num]['lower_diff']) if layer_data[layer_num]['lower_diff'] else 0
        upper_count = len(layer_data[layer_num]['upper_diff']) if layer_data[layer_num]['upper_diff'] else 0
        print(f"  Layer {layer_num}: {lower_count} lower diff values, {upper_count} upper diff values")
    
    return layer_data

def plot_histograms(layer_data):
    """
    Create separate histogram figures for Lower Diff and Upper Diff for each layer.
    """
    # Get all layer numbers sorted
    layer_nums = sorted(layer_data.keys())
    
    if not layer_nums:
        print("No AFFINE layers found in the file!")
        return
    
    saved_files = []
    
    for layer_num in layer_nums:
        lower_diff = layer_data[layer_num]['lower_diff']
        upper_diff = layer_data[layer_num]['upper_diff']
        
        # Create separate figure for this layer (2 subplots side by side)
        fig, (ax_lower, ax_upper) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Lower Diff histogram
        if lower_diff:
            ax_lower.hist(lower_diff, bins=50, edgecolor='black', alpha=0.7, color='blue')
            ax_lower.set_title(f'Layer {layer_num} - Lower Diff', fontsize=14, fontweight='bold')
            ax_lower.set_xlabel('Value', fontsize=12)
            ax_lower.set_ylabel('Frequency', fontsize=12)
            ax_lower.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(lower_diff)
            std_val = np.std(lower_diff)
            median_val = np.median(lower_diff)
            ax_lower.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_val:.3f}')
            ax_lower.axvline(median_val, color='orange', linestyle=':', linewidth=2,
                           label=f'Median: {median_val:.3f}')
            ax_lower.legend(fontsize=10)
            
            # Add text box with stats
            stats_text = f'Count: {len(lower_diff)}\nStd: {std_val:.3f}\nMin: {np.min(lower_diff):.3f}\nMax: {np.max(lower_diff):.3f}'
            ax_lower.text(0.02, 0.98, stats_text, transform=ax_lower.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot Upper Diff histogram
        if upper_diff:
            ax_upper.hist(upper_diff, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax_upper.set_title(f'Layer {layer_num} - Upper Diff', fontsize=14, fontweight='bold')
            ax_upper.set_xlabel('Value', fontsize=12)
            ax_upper.set_ylabel('Frequency', fontsize=12)
            ax_upper.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(upper_diff)
            std_val = np.std(upper_diff)
            median_val = np.median(upper_diff)
            ax_upper.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_val:.3f}')
            ax_upper.axvline(median_val, color='orange', linestyle=':', linewidth=2,
                           label=f'Median: {median_val:.3f}')
            ax_upper.legend(fontsize=10)
            
            # Add text box with stats
            stats_text = f'Count: {len(upper_diff)}\nStd: {std_val:.3f}\nMin: {np.min(upper_diff):.3f}\nMax: {np.max(upper_diff):.3f}'
            ax_upper.text(0.02, 0.98, stats_text, transform=ax_upper.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save individual figure
        
        import os
        dirpath = f'test/ai/data/histograms/{model}/'
        try:
            os.makedirs(dirpath, exist_ok=False)
        except:
            pass
        
        filename = f'test/ai/data/histograms/{model}/layer_{layer_num}_diff_histogram.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        print(f"Saved: {filename}")
        
        plt.close()  # Close to free memory
    
    print(f"\nAll {len(saved_files)} histogram files saved successfully!")
    return saved_files

def print_statistics(layer_data):
    """
    Print summary statistics for each layer.
    """
    print("\n" + "="*80)
    print("LAYER STATISTICS")
    print("="*80)
    
    for layer_num in sorted(layer_data.keys()):
        print(f"\nLayer {layer_num}:")
        
        lower_diff = layer_data[layer_num]['lower_diff']
        upper_diff = layer_data[layer_num]['upper_diff']
        
        if lower_diff:
            print(f"  Lower Diff:")
            print(f"    Count: {len(lower_diff)}")
            print(f"    Mean:  {np.mean(lower_diff):.6f}")
            print(f"    Std:   {np.std(lower_diff):.6f}")
            print(f"    Min:   {np.min(lower_diff):.6f}")
            print(f"    Max:   {np.max(lower_diff):.6f}")
        
        if upper_diff:
            print(f"  Upper Diff:")
            print(f"    Count: {len(upper_diff)}")
            print(f"    Mean:  {np.mean(upper_diff):.6f}")
            print(f"    Std:   {np.std(upper_diff):.6f}")
            print(f"    Min:   {np.min(upper_diff):.6f}")
            print(f"    Max:   {np.max(upper_diff):.6f}")


model = 'mnist_relu_9_200'

def main():
    # Specify your input file
    filename = f'test/ai/data/logs/{model}/{model}_float_worker_1.txt'  # Change this to your file path
    
    print(f"Reading data from '{filename}'...")
    
    try:
        layer_data = parse_layer_diffs(filename)
        
        if not layer_data:
            print("No AFFINE layers with Lower/Upper Diff found!")
            return
        
        print(f"Found {len(layer_data)} AFFINE layers")
        
        # Print statistics
        print_statistics(layer_data)
        
        # Create histograms
        print("\nGenerating histograms...")
        plot_histograms(layer_data)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        print("Please make sure the file exists and update the filename in the script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
