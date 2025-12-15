import onnx
import numpy as np
from onnx import numpy_helper

def read_onnx_net(net_file):
    onnx_model = onnx.load(net_file)
    onnx.checker.check_model(onnx_model)

    is_conv = any(node.op_type == "Conv" for node in onnx_model.graph.node)
    return onnx_model, is_conv


def get_onnx_parameters_as_arrays(onnx_model):
    params = {}

    for init in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(init)
        params[init.name] = arr

    return params


model_path = "test/eran_models/cifar_relu_6_100.onnx"
output_path = "test/ai/data/parameters/cifar_relu_6_100_1.txt"

model, is_conv = read_onnx_net(model_path)
params = get_onnx_parameters_as_arrays(model)

graph = model.graph

with open(output_path, "w") as file:
    for node in graph.node:

        # ---------- CONV LAYERS ----------
        if node.op_type == "Conv":
            W = None
            B = None

            # Find weights + bias tensors
            for init in graph.initializer:
                if init.name == node.input[1]:
                    W = numpy_helper.to_array(init)
                if len(node.input) > 2 and init.name == node.input[2]:
                    B = numpy_helper.to_array(init)

            if W is None:
                continue

            print(f"Layer: {node.name} (Conv)\n")

            # W shape: (out_channels, in_channels, kH, kW)
            out_ch, in_ch, kH, kW = W.shape

            # Dump weights in requested layout
            for oc in range(out_ch):
                print(f"OUT_CHANNEL {oc}\n")
                for ic in range(in_ch):
                    print(f" IN_CHANNEL {ic}\n")
                    for r in range(kH):
                        for c in range(kW):
                            file.write(str(W[oc][ic][r][c]) + " ")
                        file.write("\n")

            # Dump bias
            if B is not None:
                print("BIAS\n")
                for b in B:
                    file.write(str(b) + " ")
                file.write("\n")

            print("---------\n")


        # ---------- GEMM / LINEAR LAYERS ----------
        elif node.op_type == "Gemm":
            W = None
            B = None

            for init in graph.initializer:
                if init.name == node.input[1]:
                    W = numpy_helper.to_array(init)
                if len(node.input) > 2 and init.name == node.input[2]:
                    B = numpy_helper.to_array(init)

            if W is None:
                continue

            print(f"Layer: {node.name} (Gemm)\n")

            # Dump weights
            m, n = W.shape
            for i in range(m):
                for j in range(n):
                    file.write(str(W[i][j]) + " ")
                file.write("\n")

            # Dump bias
            if B is not None:
                for b in B:
                    file.write(str(b) + " ")
                file.write("\n")

            print("---------\n")