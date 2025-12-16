import onnx
import numpy as np
from onnx import numpy_helper

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            if a.i is not None:
                return a.i
            if a.f is not None:
                return a.f
    return default

def describe_conv(node):
    print(f"Layer: {node.name}")

    kernel = get_attr(node, "kernel_shape")
    pads   = get_attr(node, "pads")
    stride = get_attr(node, "strides")
    dil    = get_attr(node, "dilations")
    groups = get_attr(node, "group", 1)

    print(f"  Kernel size : {kernel}")
    print(f"  Stride      : {stride}")
    print(f"  Padding     : {pads}")
    print(f"  Dilation    : {dil}")
    print(f"  Groups      : {groups}")

    for init in graph.initializer:
        if init.name == node.input[1]:
            W = numpy_helper.to_array(init)
            print(f"  Weight shape: {W.shape}")
    print()

def describe_gemm(node):
    print(f"Layer: {node.name}")

    alpha  = get_attr(node, "alpha", 1.0)
    beta   = get_attr(node, "beta", 1.0)
    transA = get_attr(node, "transA", 0)
    transB = get_attr(node, "transB", 0)

    print(f"  Alpha   : {alpha}")
    print(f"  Beta    : {beta}")
    print(f"  TransA  : {transA}")
    print(f"  TransB  : {transB}")

    # Gemm inputs: A (input), B (weights), C (bias)
    if len(node.input) >= 2:
        for init in graph.initializer:
            if init.name == node.input[1]:
                W = numpy_helper.to_array(init)
                print(f"  Weight shape: {W.shape}")

    if len(node.input) >= 3:
        for init in graph.initializer:
            if init.name == node.input[2]:
                B = numpy_helper.to_array(init)
                print(f"  Bias shape  : {B.shape}")

    print()


model_path = "test/eran_models/mnist_relu_conv_small.onnx"

model = onnx.load(model_path)
graph = model.graph

for node in graph.node:
    print(f"{node.op_type:15}  {node.name}")

    if node.op_type == "Conv":
        describe_conv(node)

    if node.op_type == "Gemm":
        describe_gemm(node)