# import onnx
# import numpy as np
# from onnx import numpy_helper

# def read_onnx_net(net_file):
#     onnx_model = onnx.load(net_file)
#     onnx.checker.check_model(onnx_model)

#     is_conv = any(node.op_type == "Conv" for node in onnx_model.graph.node)
#     return onnx_model, is_conv


# def get_onnx_parameters_as_arrays(onnx_model):
#     params = {}

#     for init in onnx_model.graph.initializer:
#         arr = numpy_helper.to_array(init)
#         params[init.name] = arr

#     return params


# model_path = "test/eran_models/mnist_relu_3_50.onnx"
# output_path = "test/ai/data/parameters/mnist_relu_3_50_2.txt"

# model, is_conv = read_onnx_net(model_path)
# params = get_onnx_parameters_as_arrays(model)

# num_layers = len(params.items())//2
# params_array = []
# for i in range(num_layers):
#     params_array.append((f"{2*(i+1)}.weight", params[f"{2*(i+1)}.weight"]))
#     params_array.append((f"{2*(i+1)}.bias", params[f"{2*(i+1)}.bias"]))


# with open(output_path, "w") as file:
#     for name, array in params_array:
#         if len(array.shape) == 4:   # Conv weights: (out_ch, in_ch, kH, kW)
#             out_ch, in_ch, kH, kW = array.shape

#             arr_str = ""
#             for oc in range(out_ch):
#                 for ic in range(in_ch):
#                     for r in range(kH):
#                         for c in range(kW):
#                             arr_str += str(array[oc][ic][r][c]) + " "
#                         arr_str += "\n"

#             file.writelines(arr_str)

#         elif len(array.shape) == 2 and "weight" in name:
#             print("Shape:", array.shape)
        
#             arr_str = ""
#             m, n = array.shape
#             for i in range(m):
#                 for j in range(n):
#                     arr_str = arr_str + str(array[i][j])
#                     arr_str = arr_str + " " 
#                 arr_str = arr_str + "\n"    
#             # print(arr_str)    
            
#             file.writelines(arr_str)
             

#         elif len(array.shape) == 1 and "bias" in name:
#             print("Shape:", array.shape)
            
#             m = array.shape[0]
#             for i in array:
#                 arr_str += str(i)
#                 arr_str += " "
#             # print(arr_str)
#             arr_str += "\n"
        
#             file.writelines(arr_str)
            
        
#         # print("Shape:", array.shape)
        
#         # arr_str = ""
#         # if "weight" in name:
#         #     m, n = array.shape
#         #     for i in range(m):
#         #         for j in range(n):
#         #             arr_str = arr_str + str(array[i][j])
#         #             arr_str = arr_str + " " 
#         #         arr_str = arr_str + "\n"    
#         #     # print(arr_str)        
#         # elif "bias" in name:
#         #     m = array.shape[0]
#         #     for i in array:
#         #         arr_str += str(i)
#         #         arr_str += " "
#         #     # print(arr_str)
#         #     arr_str += "\n"
        
#         # file.writelines(arr_str)
        
#         print("---------")
        
        
        
import onnx

# Load the ONNX model
model = onnx.load("test/eran_models/cifar_conv_relu_med.onnx")

print("=== Searching for Pad nodes ===")
pad_found = False

for node in model.graph.node:
    if node.op_type == 'Pad':
        pad_found = True
        print(f"\nNode: {node.name}")
        print(f"Op Type: {node.op_type}")
        print(f"All attributes:")
        
        for attr in node.attribute:
            print(f"  - {attr.name}: ", end="")
            if attr.type == onnx.AttributeProto.STRING:
                print(attr.s.decode('utf-8'))
            elif attr.type == onnx.AttributeProto.FLOAT:
                print(attr.f)
            elif attr.type == onnx.AttributeProto.INTS:
                print(list(attr.ints))
            else:
                print(attr)

if not pad_found:
    print("\nNo explicit Pad nodes found in the model.")
    print("\n=== This means: ===")
    print("The Conv/Pool layers with pads=[1,1,1,1] use DEFAULT ZERO PADDING")
    print("This is the standard behavior in ONNX and most frameworks.")