import onnx
from onnx import numpy_helper

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints: return list(a.ints)
            if a.i: return a.i
    return default

def describe_conv(node):
    if node.op_type == "Conv":
        print(f"Layer: {node.name}")

        # Extract attributes
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

        # Optional: print weight shape
        for init in graph.initializer:
            if init.name == node.input[1]:
                W = numpy_helper.to_array(init)
                print(f"  Weight shape: {W.shape}")
        print()
        

model_path = "test/eran_models/cifar_conv_relu_small.onnx"

model = onnx.load(model_path)
graph = model.graph

for node in graph.node:
    print(f"{node.op_type:15}  {node.name}")
    if node.op_type == "Conv":
        describe_conv(node)