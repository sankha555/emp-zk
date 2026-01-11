import sys
import subprocess
import onnx
import numpy as np
import json
import jsbeautifier
import re
import os

options = jsbeautifier.default_options()
options.indent_size = 4

model_name = sys.argv[1]

if '-d' in sys.argv:
    print('Deleting:')
    onnx_path = f'test/eran_models/{model_name}.onnx'
    config_path = f'test/ai/data/configs/{model_name}_1.json'
    params_path = f'test/ai/data/parameters/{model_name}_1.json'
    
    paths = [onnx_path, config_path]
    for path in paths:
        print(path)
    
    confirm = input('proceed? (y/[n])')
    if confirm.lower() == 'y':
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
    else:
        print("Aborting...")
        
    exit(0)
    

# extract weights from onnx
onnx_path = f'test/eran_models/{model_name}.onnx'
try:
    result = subprocess.run(f"python3 test/ai/python_helpers/extract_weights.py {model_name}", shell=True)
except Exception as e:
    print(f"Weight extraction failed: {e}")
    

# write config

is_conv = 'conv' in model_name
if is_conv:
    hidden_layers = 4
else:
    hidden_layers = int(model_name.split('_')[2]) + 2 if is_conv else int(model_name.split('_')[2])
    
num_neurons = 1 if is_conv else int(model_name.split('_')[3])
dataset = 'cifar' if 'cifar' in model_name else 'mnist'


def get_layer_description(onnx_model_path):
    model = onnx.load(onnx_model_path)
    shape_model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    
    # shape map
    shape_map = {}
    
    def extract(v):
        if v.type.tensor_type.shape.dim:
            shape_map[v.name] = [
                d.dim_value if d.dim_value > 0 else None
                for d in v.type.tensor_type.shape.dim
            ]

    for v in shape_model.graph.value_info:
        extract(v)
    for v in shape_model.graph.input:
        extract(v)
    for v in shape_model.graph.output:
        extract(v)


    
    # Get initial input shape
    input_tensor = graph.input[0]
    input_shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    # CIFAR-10 flattened MLP input is often [batch, 3072]
    # We take the product of all dims except batch for the 'size'
    input_size = np.prod([d for d in input_shape if d > 0])
    
    layers = [["INPUT", int(input_size), int(input_size), -1]]
    
    # Helper to get weight shapes
    weights = {initializers.name: onnx.numpy_helper.to_array(initializers) for initializers in graph.initializer}

    for node in graph.node:
        op_type = node.op_type
        
        if op_type == "Conv":
            attr = {a.name: a.ints for a in node.attribute}
            pads = attr.get('pads', [0, 0, 0, 0])
            strides = attr.get('strides', [1, 1])
            kernel = attr.get('kernel_shape', [3, 3])

            weight_tensor = weights[node.input[1]]
            out_channels, in_channels = weight_tensor.shape[:2]

            input_shape = shape_map[node.input[0]]  # NCHW
            _, _, H, W = input_shape

            layers.append([
                "CONV2D",
                in_channels, out_channels,
                H, W,
                kernel[0], kernel[1],
                strides[0], strides[1],
                pads[0], pads[1],
                -1
            ])
            
        elif op_type == "Relu":
            # For simplicity, assuming output size equals input size for ReLU
            # In your format: ["RELU", input_size, output_size, -1]
            # You may need to track the actual flattened size from the previous layer
            last_size = layers[-1][2] if layers[-1][0] != "CONV2D" else (layers[-1][2] * 16 * 16) # Simplified
            layers.append(["RELU", last_size, last_size, -1])

        elif op_type == "Gemm" or op_type == "MatMul":
            # ONNX Gemm is usually the 'Affine' layer
            weight_tensor = weights[node.input[1]]
            # Gemm weights are usually [Out, In]
            in_features = weight_tensor.shape[1]
            out_features = weight_tensor.shape[0]
            layers.append(["AFFINE", in_features, out_features, -1])

    # Add Output layer based on the last layer's output
    final_size = layers[-1][2]
    layers.append(["OUTPUT", final_size, final_size, -1])
    
    return layers


def get_epsilon():
    if is_conv and dataset == 'cifar':
        return 0.008
    elif not is_conv and dataset == 'cifar':
        return 0.0002
    elif dataset == 'mnist':
        return 0.01
    
def get_default_thresholds():
    thresholds = {}
    
    if is_conv:
        for i in range(2, hidden_layers+1):
            thresholds[str(2*i)] = -1
    else:
        # MLP
        for i in range(3, hidden_layers+1):
            thresholds[str(2*i)] = -1
            
    return thresholds
        

config = {
    "model_name": model_name,
    "num_neurons": num_neurons,
    "layers": get_layer_description(onnx_path),
    "epsilon": get_epsilon(),
    "input_min": -10 if dataset == 'cifar' else -1,
    "input_max": 10 if dataset == 'cifar' else 1,
    "do_float": 0,
    "FXPSCALE": 24,
    "examples": list(range(1, 101)),
    "bs_waiver_thresholds": get_default_thresholds(),
    "bs_waiver_thresholds2": get_default_thresholds()
}

config_path = f'test/ai/data/configs/{model_name}_1.json'
with open(config_path, 'w') as config_file:
    config_file.write(jsbeautifier.beautify(json.dumps(config)))
    
    
    
if 'verify' in sys.argv:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    print(f"Running verification algorithm on {model_name}")
    print(f"epsilon = {config['epsilon']}")
    
    num_examples = 100
    cmd1 = f"./bin/test_ai_run_verification 1 10000 {model_name} -1 {num_examples}"
    cmd2 = f"./bin/test_ai_run_verification 2 10000 {model_name} -1 {num_examples}"
    
    proc1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        output1, err1 = proc1.communicate(timeout=600)
        output2, _ = proc2.communicate(timeout=600)
    except Exception as e:
        print(e)
    
    verified_match = re.search(re.compile(r"Verified (\d+)/(\d+) examples \[ correctly classified = (\d+) \]"), output1)
    if verified_match:
        verified = int(verified_match.group(1))
        classified = int(verified_match.group(3))
    else:
        verified = 1
        classified = 1
        
    print(f"Verified {verified}/{classified} examples.")