import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from scipy.stats import shapiro, kurtosis
from sklearn.mixture import GaussianMixture

def quality_metrics(x):
    def histogram_entropy(x, bins=50):
        """
        Normalized Shannon entropy of histogram.
        Returns value in [0, 1].
        """
        hist, _ = np.histogram(x, bins=bins, density=False)
        hist = hist.astype(np.float64)
        hist /= hist.sum()

        # avoid log(0)
        hist = hist[hist > 0]

        H = -np.sum(hist * np.log(hist))
        H_max = np.log(bins)

        return H / H_max
        
    def histogram_tv(x, bins=50):
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist /= hist.sum()
        return np.sum(np.abs(np.diff(hist)))
    
    def bimodality_score(x):
        x = np.array(x)
        g1 = GaussianMixture(1).fit(x.reshape(-1,1)).bic(x.reshape(-1,1))
        g2 = GaussianMixture(2).fit(x.reshape(-1,1)).bic(x.reshape(-1,1))
        return g1 - g2  # positive → bimodal
    
    hnorm = histogram_entropy(x)
    tv = histogram_tv(x)
    bms = bimodality_score(x)

    cv = np.std(x) / np.mean(x)
    
    TV_norm = tv / (tv + 1)
    B_norm  = bms  / (bms  + 50)   # critical: scale BIC!

    score = (
        0.5 * hnorm +
        0.3 * (1 - TV_norm) +
        0.2 * (1 - B_norm)
    )
    
    return cv, hnorm, tv, bms, score

HEURISTICS_PATH = 'test/ai/data/heuristics'

def round_json(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)

    if isinstance(obj, dict):
        return {k: round_json(v, decimals) for k, v in obj.items()}

    if isinstance(obj, list):
        return [round_json(v, decimals) for v in obj]

    return obj

def parse_verification_file(filepath):
    """
    Parse the verification file and extract layer differences for verified examples.
    Only includes AFFINE layers (which have diff values).
    
    Returns:
        dict: layer_name -> list of all diff values across all verified examples
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split into examples
    examples = re.split(r'EXAMPLE \d+ :', content)
    examples = [ex for ex in examples if ex.strip()]
    
    layer_diffs = defaultdict(list)
    
    for example in examples:
        # Check if verified
        if 'Verified: YES' not in example:
            continue
        
        # Find all layers - now more carefully parsing the structure
        # Look for patterns like "LAYER X\nType: AFFINE[Y]\n...Lower Diff:\n..."
        layer_sections = re.split(r'LAYER (\d+)', example)
        
        # Process pairs of (layer_num, layer_content)
        for i in range(1, len(layer_sections), 2):
            if i + 1 >= len(layer_sections):
                break
                
            layer_num = layer_sections[i]
            layer_content = layer_sections[i + 1]
            
            # Extract layer type
            type_match = re.search(r'Type:\s*(\w+(?:\[\d+\])?)', layer_content)
            if not type_match:
                continue
            
            layer_type = type_match.group(1)
            
            # Only process AFFINE layers
            if not layer_type.startswith('AFFINE') and not layer_type.startswith('CONV2D'):
                continue
            
            # Look for Lower Diff or Upper Diff sections
            diff_match = re.search(r'(?:Lower Diff|Upper Diff):\s*\n((?:[\d.\s-]+\n?)+)', layer_content)
            if not diff_match:
                continue
            
            diffs_str = diff_match.group(1)
            
            # Extract layer name
            layer_name = f"Layer {layer_num} ({layer_type})"
            
            # Parse the diff values
            diff_values = []
            for line in diffs_str.strip().split('\n'):
                if line.strip():  # Skip empty lines
                    values = [float(x) for x in line.split() if x.strip()]
                    diff_values.extend(values)
            
            if diff_values:  # Only add if we found values
                layer_diffs[layer_name].extend(diff_values)
    
    return layer_diffs

def compute_percentiles(layer_diffs):
    """
    Compute 10th, 20th, ..., 90th percentiles for each layer.
    
    Returns:
        dict: layer_name -> dict of percentile -> value
    """
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    layer_stats = {}
    
    for layer_name, diffs in layer_diffs.items():
        if diffs:
            cv, hnorm, tv, bms, score = quality_metrics(diffs)
            
            layer_stats[layer_name] = {
                f'{p}th': np.percentile(diffs, p)
                for p in percentiles
            }
            layer_stats[layer_name]["min"] = np.min(diffs)
            layer_stats[layer_name]["mean"] = np.mean(diffs)
            layer_stats[layer_name]["max"] = np.max(diffs)
            layer_stats[layer_name]["score"] = score
            
    
    return layer_stats

def plot_histograms(layer_diffs, output_file='layer_histograms.png'):
    """
    Create histograms for all layers on a single image.
    """
    n_layers = len(layer_diffs)
    if n_layers == 0:
        print("No data to plot")
        return
    
    # Determine grid layout
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Sort layers by layer number for consistent ordering
    sorted_layers = sorted(layer_diffs.items(), 
                          key=lambda x: int(re.search(r'Layer (\d+)', x[0]).group(1)))
    
    for idx, (layer_name, diffs) in enumerate(sorted_layers):
        ax = axes[idx]
        ax.hist(diffs, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Difference Value')
        ax.set_ylabel('Frequency')
        ax.set_title(layer_name)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(diffs):.3f}\nStd: {np.std(diffs):.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(sorted_layers), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Histograms saved to {output_file}")
    plt.close()

def save_statistics(layer_stats, output_file='layer_statistics.txt'):
    """
    Save percentile statistics to a text file.
    """
    with open(output_file, 'w') as f:
        f.write("Layer-wise Percentile Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by layer number
        sorted_layers = sorted(layer_stats.items(),
                              key=lambda x: int(re.search(r'Layer (\d+)', x[0]).group(1)))
        
        for layer_name, stats in sorted_layers:
            f.write(f"{layer_name}:\n")
            for percentile, value in sorted(stats.items(), 
                                          key=lambda x: int(x[0].replace('th', ''))):
                f.write(f"  {percentile:>4s} percentile: {value:.6f}\n")
            f.write("\n")
    
    print(f"Statistics saved to {output_file}")

def save_statistics_json(layer_stats, output_file='layer_statistics.json'):
    """
    Save percentile statistics to a JSON file for easy reading.
    """
    rounded_stats = round_json(layer_stats)
    with open(output_file, 'w') as f:
        json.dump(rounded_stats, f, indent=2)
    print(f"Statistics saved to {output_file} (JSON format)")

def load_statistics_json(input_file='layer_statistics.json'):
    """
    Load percentile statistics from a JSON file.
    
    Returns:
        dict: layer_name -> dict of percentile -> value
    """
    with open(input_file, 'r') as f:
        return json.load(f)

def collect_and_write_statistics(input_file, hist_output='layer_histograms.png', 
         stats_output='layer_statistics.txt',
         json_output='layer_statistics.json'):
    """
    Main function to process verification file and generate outputs.
    """
    print(f"Processing file: {input_file}")
    
    # Parse the file
    layer_diffs = parse_verification_file(input_file)
    
    if not layer_diffs:
        print("No verified examples found in the file!")
        return
    
    print(f"Found {len(layer_diffs)} layers with difference data")
    for layer_name, diffs in sorted(layer_diffs.items(),
                                    key=lambda x: int(re.search(r'Layer (\d+)', x[0]).group(1))):
        print(f"  {layer_name}: {len(diffs)} values")
    
    # Compute statistics
    layer_stats = compute_percentiles(layer_diffs)
    
    # Generate outputs
    plot_histograms(layer_diffs, hist_output)
    save_statistics_json(layer_stats, json_output)
    
    print("\nProcessing complete!")
    print(f"  - Histograms: {hist_output}")
    print(f"  - Statistics (JSON): {json_output}")
    
    return layer_stats


if __name__ == "__main__":
    model_name = 'cifar_relu_conv_small'
    logfile = f'test/ai/data/logs/{model_name}/{model_name}_float_worker_1.txt'
    
    os.makedirs(f'{HEURISTICS_PATH}/{model_name}', exist_ok=True)
    
    histogram_path = f'{HEURISTICS_PATH}/{model_name}/histogram.png'
    stats_txt_path = f'{HEURISTICS_PATH}/{model_name}/stats.txt'
    percentiles_path = f'{HEURISTICS_PATH}/{model_name}/thresholds.json'
    
    # Process the verification file
    stats = collect_and_write_statistics(logfile, histogram_path, stats_txt_path, percentiles_path)
    
    # Example: Load statistics back as a map
    loaded_stats = load_statistics_json(percentiles_path)
    

