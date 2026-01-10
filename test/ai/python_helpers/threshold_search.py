import json
import subprocess
import re
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import jsbeautifier
from itertools import product

options = jsbeautifier.default_options()

def load_layer_stats(stats_file='layer_statistics.json'):
    """Load percentile statistics for each layer."""
    with open(stats_file, 'r') as f:
        return json.load(f)

def load_config(config_path):
    """Load the config file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    """Save the config file, preserving formatting."""
    with open(config_path, 'w') as f:
        f.write(jsbeautifier.beautify(json.dumps(config), options))

def run_verification(model_name, num_examples=100, epsilon1=0.1, epsilon2=0.15):
    """Run both verification commands and parse output from first command."""
    cmd1 = f"./bin/test_ai_run_verification 1 10000 {model_name} -1 {num_examples} {epsilon1} {epsilon2}"
    cmd2 = f"./bin/test_ai_run_verification 2 10000 {model_name} -1 {num_examples} {epsilon1} {epsilon2}"
    
    try:
        # Start both processes
        proc1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for both to complete (with timeout)
        output1, err1 = proc1.communicate(timeout=600)
        output2, _ = proc2.communicate(timeout=600)
        
        # Use output from first command only
        output = output1 + err1
        
        # Parse verification accuracy
        verified_match = re.search(r'Verified \s*(\d+)/(\d+)\s+examples', output)
        if verified_match:
            verified = int(verified_match.group(1))
            total = int(verified_match.group(2))
        else:
            return None
        
        # Parse coupled savings
        coupled_match = re.search(r'Avg\. savings = (\d+)\s+\[(\d+)\s*%\].*?\(Total (\d+) examples\)', output, re.DOTALL)
        if coupled_match:
            avg_savings = int(coupled_match.group(1))
            avg_savings_pct = int(coupled_match.group(2))
        else:
            avg_savings = 0
            avg_savings_pct = 0
        
        # Parse max coupled savings
        max_match = re.search(r'Max\. savings = (\d+)\s+\[(\d+)\s*%\]\s+\(Example (\d+)\)', output)
        if max_match:
            max_savings = int(max_match.group(1))
            max_savings_pct = int(max_match.group(2))
            max_example = int(max_match.group(3))
        else:
            max_savings = 0
            max_savings_pct = 0
            max_example = -1
            
        return {
            'verified': verified,
            'total': total,
            'avg_savings_pct': avg_savings_pct,
            'max_savings_pct': max_savings_pct,
            'max_example': max_example
        }
    except Exception as e:
        print(f"Error running verification: {e}")
        return None

def get_affine_layer_indices(config):
    """Get indices of AFFINE layers from config."""
    affine_indices = []
    for idx, layer_num in enumerate(config['bs_waiver_thresholds']):
        affine_indices.append(int(layer_num))
    return affine_indices

def update_thresholds(config, layer_stats, affine_indices, percentile, threshold_key='bs_waiver_thresholds'):
    """Update thresholds in config based on percentile."""
    for idx in affine_indices:
        layer_name = f"Layer {idx} (AFFINE[{idx}])"
        if layer_name in layer_stats:
            threshold_value = layer_stats[layer_name][f'{percentile}th']
            config[threshold_key][str(idx)] = round(threshold_value, 3)
    return config

def get_layer_thresholds(layer_stats):
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    layer_thresholds = {}
    
    for layer in layer_stats:
        if 'Layer 2' in layer or 'Layer 4 (AFFINE' in layer or 'Layer 18 (AFFINE' in layer:
            print(layer)
            continue
        
        layer_num = int(re.search(r'Layer\s+(\d+)', layer).group(1))
        
        stats = layer_stats[layer]
        layer_thresholds[layer_num] = [-1]
        
        thresholds = []
        for p in percentiles:
            thresholds.append(stats[f"{p}th"])
                    
        delta_threshold = 0
        last_used_threshold = thresholds[0]
        range_thresholds = thresholds[-1] - thresholds[0]
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            delta_threshold = thresholds[i] - last_used_threshold
            if delta_threshold >= 0.2 * range_thresholds:
                layer_thresholds[layer_num].append(thresholds[i])
                last_used_threshold = thresholds[i]
                
    
    return layer_thresholds

            

def search_phase1(config_path, model_name, layer_stats, affine_indices, 
                  min_verified, max_verified, num_examples=100):
    """
    Phase 1: Increase bs_waiver_thresholds from 10th to 90th percentile.
    """
    print("\n" + "="*80)
    print("PHASE 1: Increasing bs_waiver_thresholds")
    print("="*80)
    
    config = load_config(config_path)
    results = []
    best_result = {'verified': 0, 'avg_savings': 0, 'avg_savings_pct': 0}
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    layer_thresholds = get_layer_thresholds(layer_stats)
    keys = layer_thresholds.keys()
    values = layer_thresholds.values()

    combinations = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]
    print(len(combinations), ' combinations to try')
        
    percentile = 0
    last_percentile = 0
    flagged_combos = set()
    for combo in combinations:
        
        combo_key = frozenset(combo.items())
        if combo_key in flagged_combos:
            continue
        flagged_combos.add(combo_key)
        combinations.remove(combo)
        
        print(combo, end=' ==> ')
                                        
        # Update config
        # config = update_thresholds(config, layer_stats, affine_indices, percentile, 'bs_waiver_thresholds')
        config['bs_waiver_thresholds'] = combo
        save_config(config, config_path)
        
        # Run verification
        result = run_verification(model_name, num_examples)
        if result:
            # Stop if verification drops below minimum
            print(result['verified'], ' examples')
            if result['verified'] < min_verified:
            
                def matches_all_kv(m, flag):
                    return all(k in m and m[k] >= v for k, v in flag.items())

                flag_combo = {}
                for k in combo.keys():
                    if combo[k] != -1:
                        flag_combo[k] = combo[k]
                                        
                skip_combos = [m for m in combinations if matches_all_kv(m, flag_combo)]
                for sc in skip_combos:
                    sc_key = frozenset(sc.items())
                    flagged_combos.add(sc_key)
                    
                print(f"Verification dropped below minimum ({min_verified}).  Skipping all such combos which contain {flag_combo}. {len([c for c in combinations if frozenset(c.items()) not in flagged_combos])} combinations left")
                
                continue
                    
            result['combo'] = combo
            result['percentile'] = percentile
            result['phase'] = 1
            result['config'] = config['bs_waiver_thresholds'].copy()
            result['config2'] = config['bs_waiver_thresholds2'].copy()
            
            # print(f"Verified: {result['verified']}/{result['total']}")
            # print(f"Avg coupled savings: {result['avg_savings']} ({result['avg_savings_pct']}%)")
            # print(f"Max savings: {result['max_savings']} ({result['max_savings_pct']}%) at example {result['max_example']}")
            
            if result['avg_savings_pct'] > best_result['avg_savings_pct']:
                best_result = result.copy()
                print(f"*** NEW BEST: {best_result['verified']} verified, {best_result['avg_savings_pct']} % savings ***")
                results.append(result)
            else:
                existing_best_combo_for_this_accuracy = {}
                for r in results:
                    if r['verified'] == result['verified']:
                        existing_best_combo_for_this_accuracy = r
                        break
                        
                if existing_best_combo_for_this_accuracy == {}:
                    # this accuracy does not exist
                    results.append(result)
                else:
                    
                    if existing_best_combo_for_this_accuracy['avg_savings_pct'] >= result['avg_savings_pct']:
                        # this new result does not give any benefit: same accuracy but less savings; chuck all such combinations
                        flag_combo = combo
                        
                        def matches_all_kv_st_expected_savings_less(m, flag):
                            return all(k in m and m[k] <= v for k, v in flag.items())
                                    
                        skip_combos = [m for m in combinations if matches_all_kv_st_expected_savings_less(m, flag_combo)]
                        for sc in skip_combos:
                            sc_key = frozenset(sc.items())
                            flagged_combos.add(sc_key)
                            
                        print(f"Skipping all such combos which contain <= {flag_combo} since suboptimal for existing accuracy. {len([c for c in combinations if frozenset(c.items()) not in flagged_combos])} combinations left")

                            
                    else:
                        # this new result gives better savings with same accuracy
                        results.remove(existing_best_combo_for_this_accuracy)
                        results.append(result)
                
            
            last_percentile = percentile
            
    print("="*40, "Phase 1 completed", "="*40)
    
    return results, best_result, last_percentile

def h2_search(config_path, p1_results, model_name, layer_stats, affine_indices, min_verified, max_verified, num_examples=100):
    print("\n", "="*80)
    print("PHASE 2 SEARCH FOR HEURISTIC 2")
    print("="*80)
    
    accuracy_to_best = {}
    for result in p1_results:
        acc = result['verified']
        if acc not in accuracy_to_best or result['avg_savings_pct'] > accuracy_to_best[acc]['avg_savings_pct']:
            accuracy_to_best[acc] = result
    p1_results = accuracy_to_best.values()        
    
    
    results = []
    best_result = {'verified': 0, 'avg_savings': 0, 'avg_savings_pct': 0}
    
    percentile = 0
    
    for p1_result in p1_results:
        combo1 = p1_result['combo']
        p1_acc = p1_result['verified']
        
        config = load_config(config_path)
        config['bs_waiver_thresholds'] = combo1
        
        layerwise_remaining_thresholds = {}
        for layer in layer_stats:
            if 'Layer 2' in layer or 'Layer 4 (AFFINE' in layer or 'Layer 18 (AFFINE' in layer:
                continue
            
            stats = layer_stats[layer]
            layer_num = int(re.search(r'Layer\s+(\d+)', layer).group(1))
                        
            layerwise_remaining_thresholds[layer_num] = []
            for p in range(10, 100, 10):
                if combo1[(layer_num)] != -1 and stats[f"{p}th"] > combo1[(layer_num)]:
                    layerwise_remaining_thresholds[layer_num].append(stats[f"{p}th"])
            
            if len(layerwise_remaining_thresholds[layer_num]) == 0:
                layerwise_remaining_thresholds[layer_num].append(-1)
            
            
        keys = layerwise_remaining_thresholds.keys()
        values = layerwise_remaining_thresholds.values()
        combinations = [
            dict(zip(keys, combo))
            for combo in product(*values)
        ]
        
        
        print("\n\n", len(combinations), ' combinations to try')
        
        
        flagged_combos = set()
        for combo in combinations:
            combo_key = frozenset(combo.items())
            if combo_key in flagged_combos:
                continue
            flagged_combos.add(combo_key)
            combinations.remove(combo)
            
            print(combo, end=' ==> ')
            
            config['bs_waiver_thresholds2'] = combo 
            save_config(config, config_path)
            
            
            result = run_verification(model_name, num_examples)
            if result:
                # Stop if verification drops below minimum
                print(result['verified'], ' examples')
                if result['verified'] < min_verified:
                
                    def matches_all_kv(m, flag):
                        return all(k in m and m[k] >= v for k, v in flag.items())

                    flag_combo = {}
                    for k in combo.keys():
                        if combo[k] != -1:
                            flag_combo[k] = combo[k]
                                            
                    skip_combos = [m for m in combinations if matches_all_kv(m, flag_combo)]
                    for sc in skip_combos:
                        sc_key = frozenset(sc.items())
                        flagged_combos.add(sc_key)
                        
                    print(f"Verification dropped below minimum ({min_verified}).  Skipping all such combos which contain {flag_combo}. {len([c for c in combinations if frozenset(c.items()) not in flagged_combos])} combinations left")
                    
                    continue
                        
                result['combo'] = combo
                result['percentile'] = percentile
                result['phase'] = 2
                result['config'] = config['bs_waiver_thresholds'].copy()
                result['config2'] = config['bs_waiver_thresholds2'].copy()
            
                if result['avg_savings_pct'] > p1_acc:
                    best_result = result.copy()
                    print(f"*** PHASE 2 NEW BEST: {best_result['verified']} verified, {best_result['avg_savings_pct']} % savings ***")
                    results.append(result)
                    
    return results


def search_phase2(config_path, model_name, layer_stats, affine_indices,
                  min_verified, max_verified, num_examples=100, phase1_results=None, last_percentile = 90, best_config1 = None):
    """
    Phase 2: Fix bs_waiver_thresholds2 and relax bs_waiver_thresholds.
    """
    print("\n" + "="*80)
    print("PHASE 2: Relaxing with bs_waiver_thresholds2")
    print("="*80)
    
    if not phase1_results:
        return []
    
    # Find the last config before dropping below min_verified
    # valid_results = [r for r in phase1_results if r['verified'] >= min_verified]
    valid_results = phase1_results[::-1]
    if not valid_results:
        print("No valid Phase 1 results to continue from.")
        return []
    
    results = []
    # Start from the highest percentile from phase 1
    for start_config in valid_results:
        if start_config['verified'] >= min_verified:
            # no need to refine this further
            continue
        
        config = load_config(config_path)
        
        best_result = start_config.copy()
        
        # Copy thresholds from bs_waiver_thresholds to bs_waiver_thresholds2
        config['bs_waiver_thresholds2'] = start_config['config']
        
        # Try relaxing bs_waiver_thresholds
        percentiles = range(10, start_config['percentile'], 10)[::-1]
        
        for percentile in percentiles:
            print(f"\n--- Testing {percentile}th percentile (with threshold2 fixed to {start_config['percentile']}) ---")
            
            # Update only bs_waiver_thresholds
            config = update_thresholds(config, layer_stats, affine_indices, percentile, 'bs_waiver_thresholds')
            save_config(config, config_path)
            
            # Run verification
            result = run_verification(model_name, num_examples)
            
            if result:
                result['percentile'] = percentile
                result['phase'] = 2
                result['config'] = config['bs_waiver_thresholds'].copy()
                result['config2'] = config['bs_waiver_thresholds2'].copy()
                
                if result['verified'] >= min_verified:                
                    results.append(result)
                    
                    print(f"Verified: {result['verified']}/{result['total']}")
                    print(f"Avg coupled savings: {result['avg_savings_pct']}%")
                    print(f"Max savings: {result['max_savings_pct']}% at example {result['max_example']}")
                    
                    if result['avg_savings_pct'] > best_result['avg_savings_pct'] and result['verified'] >= min_verified:
                        best_result = result.copy()
                        print(f"*** NEW BEST: {best_result['verified']} verified, {best_result['avg_savings_pct']} % savings ***")
                        
                    # no need to be more pessimistic, we obtain good accuracy already
                    break
        
    return results

def plot_results(all_results, output_file='threshold_search_results.png'):
    """Plot verification accuracy vs savings."""

    accuracy_to_avg_savings = defaultdict(list)
    accuracy_to_max_savings = defaultdict(list)

    for result in all_results:
        acc = result['verified']
        accuracy_to_avg_savings[acc].append(result['avg_savings_pct'])
        accuracy_to_max_savings[acc].append(result['max_savings_pct'])

    accuracies = sorted(accuracy_to_avg_savings.keys())

    max_avg_savings = [max(accuracy_to_avg_savings[a]) for a in accuracies]
    max_max_savings = [max(accuracy_to_max_savings[a]) for a in accuracies]

    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, max_avg_savings, 'o-', linewidth=2, markersize=8,
             label='Max Avg Savings (%)')
    plt.plot(accuracies, max_max_savings, 's--', linewidth=2, markersize=8,
             label='Max Savings (%)')

    plt.xlabel('Verification Accuracy (# Verified)', fontsize=12)
    plt.ylabel('Savings (%)', fontsize=12)
    plt.title('Verification Accuracy vs Savings', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")
    plt.close()
    
    
def save_best_configs(all_results, output_file='best_configs.json'):
    """Save best configuration for each accuracy level."""
    accuracy_to_best = {}
    
    for result in all_results:
        acc = result['verified']
        if acc not in accuracy_to_best or result['avg_savings_pct'] > accuracy_to_best[acc]['avg_savings_pct']:
            accuracy_to_best[acc] = {
                'verified': result['verified'],
                'avg_savings_pct': result['avg_savings_pct'],
                'max_savings_pct': result['max_savings_pct'],
                'percentile': result.get('percentile', -1),
                'phase': result.get('phase', -1),
                'bs_waiver_thresholds': result['config'],
                'bs_waiver_thresholds2': result['config2']
            }
    
    # Create indexed format
    indexed_results = {
        f"{acc}-{accuracy_to_best[acc]['avg_savings_pct']}": accuracy_to_best[acc]
        for acc in sorted(accuracy_to_best.keys())
    }
    
    with open(output_file, 'w') as f:
        json.dump(indexed_results, f, indent=4)
    
    print(f"Best configurations saved to {output_file}")


MODEL_INFO = {
    'mnist_relu_6_100': {
        'min_v1': 80,
        'p2_filter': 85,
    },
    
    'mnist_relu_9_200': {
        'min_v1': 50,
        'p2_filter': 50,
    },
    
    'cifar_relu_conv_small': {
        'min_v1': 40,
        'p2_filter': 42,
    },
}


def main():
    # Configuration
    model_name = 'mnist_relu_6_100'
    config_path = f'test/ai/data/configs/{model_name}_1.json'
    stats_file = f'test/ai/data/heuristics/{model_name}/thresholds.json'
    
    min_verified_p1 = MODEL_INFO[model_name]['min_v1']  # Minimum acceptable verified examples
    p2_filter = MODEL_INFO[model_name]['p2_filter']  # Minimum acceptable verified examples

    max_verified = 65  # Maximum possible verified examples
    num_examples = 100
    
    print("="*80)
    print("THRESHOLD HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Min verified: {min_verified_p1}/{num_examples}")
    print(f"Max verified: {max_verified}/{num_examples}")
    
    # Load layer statistics
    print(f"\nLoading layer statistics from {stats_file}...")
    layer_stats = load_layer_stats(stats_file)
    
    # Get AFFINE layer indices
    config = load_config(config_path)
    affine_indices = get_affine_layer_indices(config)
    print(f"AFFINE layers found at indices: {affine_indices}")
    
    # reset config to all -1 before starting
    for k in config['bs_waiver_thresholds']:
        config['bs_waiver_thresholds'][k] = -1
        config['bs_waiver_thresholds2'][k] = -1

    save_config(config, config_path)

    start_time = time.time()
    
    # Phase 1: Increase thresholds
    # phase1_results, best_phase1, last_percentile = [{
    #     "verified": 46,
    #     "avg_savings_pct": 4,
    #     "max_savings_pct": 4,
    #     "percentile": 0,
    #     "phase": 1,
    #     "config": {
    #         "4": -1,
    #         "6": 3.275,
    #         "8": -1
    #     },
    #     "combo": {
    #         "4": -1,
    #         "6": 3.275,
    #         "8": -1
    #     },
    #     "config2": {
    #         "4": -1,
    #         "6": -1,
    #         "8": -1
    #     }
    # }], None, None
    
    phase1_results, best_phase1, last_percentile = search_phase1(
        config_path, model_name, layer_stats, affine_indices,
        min_verified_p1, max_verified, num_examples
    )
       

    # Phase 2: Relax with threshold2
    filtered_p1_results = phase1_results
    filtered_p1_results = [r for r in phase1_results if r['verified'] >= p2_filter]
    phase2_results = []
    phase2_results = h2_search(config_path, filtered_p1_results, model_name, layer_stats, affine_indices, min_verified_p1, max_verified, num_examples)
    
    # Combine results
    all_results = phase1_results
    if len(phase2_results) > 0: 
        all_results = all_results + phase2_results
    all_results = [result for result in all_results if result['verified'] >= min_verified_p1]
    
    # Find overall best
    best_overall = max(all_results, key=lambda x: (x['avg_savings_pct'], x['verified']))
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("SEARCH COMPLETED")
    print("="*80)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"\nBest result:")
    print(f"  Verified: {best_overall['verified']}/{best_overall['total']}")
    print(f"  Avg savings: {best_overall['avg_savings_pct']}%")
    print(f"  Max savings: {best_overall['max_savings_pct']}%")
    print(f"  Phase: {best_overall['phase']}, Percentile: {best_overall['percentile']}th")
    
    # Generate outputs
    plot_results(all_results, 'threshold_search_results.png')
    save_best_configs(all_results, 'best_configs.json')
    
    print("\nSearch complete!")

if __name__ == "__main__":
    main()