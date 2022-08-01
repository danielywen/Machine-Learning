import onnx
import onnxruntime as ort
import argparse
from onnx import TensorProto
from typing import Dict
import numpy as np
import pdb
from onnx import helper

# Dictionary to map all TensorProto types to string labels
TENSOR_PROTO_TO_STRING: Dict[int, str] = {
    TensorProto.UNDEFINED: 'UNDEFINED',
    TensorProto.FLOAT: 'FLOAT32',
    TensorProto.UINT8: 'UINT8',
    TensorProto.INT8: 'INT8',
    TensorProto.UINT16: 'UINT16',
    TensorProto.INT16: 'INT16',
    TensorProto.INT32: 'INT32',
    TensorProto.INT64: 'INT64',
    TensorProto.STRING: 'STRING',
    TensorProto.BOOL: 'BOOL',
    TensorProto.FLOAT16: 'FLOAT16',
    TensorProto.DOUBLE: 'DOUBLE',
    TensorProto.UINT32: 'UINT32',
    TensorProto.UINT64: 'UINT64',
    TensorProto.COMPLEX64: 'COMPLEX64',
    TensorProto.COMPLEX128: 'COMPLEX128',
}

# Print parameters for convolution layers
def print_attr_conv(node, input_channels, output_channels):
    print("\n")
    for attributes in range(len(node.attribute)):
        if node.attribute[attributes].name == 'group':
            if (node.attribute[attributes].i == input_channels) and (node.attribute[attributes].i == output_channels):
                print('DEPTHWISE: YES')
            else:
                print('DEPTHWISE: NO')
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].i)
        elif node.attribute[attributes].name == 'dilations':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints)
        elif node.attribute[attributes].name == 'kernel_shape':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints)
        elif node.attribute[attributes].name == 'pads':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints)
        elif node.attribute[attributes].name == 'strides':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints, end="")

# Returns True if conv layer
def if_Conv(node):
    if node.op_type == 'Conv':
        return True
    elif node.op_type == 'ConvTranspose':
        return True
    elif node.op_type == 'QLinearConv':
        return True
    return False

# Returns True if pooling layer
def if_PoolCheck(node):
    if node.op_type == 'MaxPool':
        return True
    elif node.op_type == 'AveragePool':
        return True
    elif node.op_type == 'QLinearAveragePool':
        return True
    return False

# Prints parameters for pooling layers
def print_attr_pool(node):
    print('\n')
    for attributes in range(len(node.attribute)):
        if node.attribute[attributes].name == 'kernel_shape':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints)
        elif node.attribute[attributes].name == 'pads':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints)
        elif node.attribute[attributes].name == 'strides':
            print(f"{node.attribute[attributes].name}:", end=" ")
            print(node.attribute[attributes].ints, end="")

# Print tensor shape
def print_shape(model, node_count, found, check_depthwise, channels):
    for dimensions in range(len(model.graph.value_info[node_count].type.tensor_type.shape.dim)):
        if dimensions == ((len(model.graph.value_info[node_count].type.tensor_type.shape.dim)) - 1):
            print(f"{model.graph.value_info[node_count].type.tensor_type.shape.dim[dimensions].dim_value})", end="  ")
            # Found tensor shape   
            return True, channels
        else:
            if (dimensions == 1) and (check_depthwise == True):
                channels = model.graph.value_info[node_count].type.tensor_type.shape.dim[dimensions].dim_value
            print(f"{model.graph.value_info[node_count].type.tensor_type.shape.dim[dimensions].dim_value},", end=" ")
    # FOR SPECIAL CASE WHERE NODE INPUT SHAPE = []
    if (len(model.graph.value_info[node_count].type.tensor_type.shape.dim) == 0):
        print(")", end="  ")
        return True, channels
    return found, channels

# head = model.graph.input OR model.graph.output OR model.graph.initializer
# node_name = model.graph.input[iter] OR model.graph.output[iter]
# input_or_initializer = 0 if searching initializers, = 1 if searching graph inputs
# found = TRUE if shape is found
def parse_search(head, node_name, input_or_initializer, found, check_depthwise, channels):
    for count in range(len(head)):
        if input_or_initializer == 0:
            j = head[count].dims   
            p = TENSOR_PROTO_TO_STRING[head[count].data_type]     
        elif input_or_initializer == 1:
            j = head[count].type.tensor_type.shape.dim
            p = TENSOR_PROTO_TO_STRING[head[count].type.tensor_type.elem_type]
        if (node_name == head[count].name) and (found == False):
            print(f"{node_name}:{p}->(", end="")
            # iterate through the tensor dimensions
            for dimensions in range(len(j)):
                if input_or_initializer == 0:
                    k = j[dimensions]
                elif input_or_initializer == 1:
                    k = j[dimensions].dim_value
                if dimensions == ((len(j)) - 1):
                    print(f"{k})", end="  ")
                    return True, channels
                else:
                    if (dimensions == 1) and (check_depthwise == True):
                        channels = k
                    print(f"{k},", end=" ")

            # FOR SPECIAL CASE WHERE INITIALIZER SHAPE = []
            if (len(j) == 0):
                print(")", end="  ")
                return True, channels

    return found, channels

# x = node.input OR node.output
# y = model.graph.input OR model.graph.output
# input_or_output = True if input, False if output
def parse_node(model, x, y, channels, input_or_output, check_depthwise_conv):
    # Check that the layer has inputs/outputs
    if len(x) >= 1:
        # iterate through this particular node's input/output list.  e.g. ['input_node0', 'input_node1', 'input_node2']
        for iter in range(len(x)):   
            found = False
            check_depthwise = False
            # iter == 0 will be the input for conv layers
            if (iter == 0) and (check_depthwise_conv == True):
                check_depthwise = True
            # NODE INPUTS
            # Search through nodes in graph in order of tree (graph.value_info)
            for node_count in range(len(model.graph.value_info)):      
                if (x[iter] == model.graph.value_info[node_count].name) and (found == False):                       
                    print(f"{x[iter]}:", end="")
                    print(f"{TENSOR_PROTO_TO_STRING[model.graph.value_info[node_count].type.tensor_type.elem_type]}->(", end="")
                    if (iter == 1) and (input_or_output == True):
                        found, channels = print_shape(model, node_count, found, check_depthwise, channels)
                    else:
                        found, channels = print_shape(model, node_count, found, check_depthwise, channels)
                # if the input/output node is a graph input or initializer
                elif (found == False):
                    # GRAPH INPUTS: Search through GRAPH INPUTS (not node inputs) to print NODE INPUT shape.  e.g. ['input0', 'input1', 'input2']
                    found, channels = parse_search(y, x[iter], 1, found, check_depthwise, channels)

                    # INITIALIZERS: Search through INITIALIZERS to print NODE OUTPUT shape
                    found, channels = parse_search(model.graph.initializer, x[iter], 0, found, check_depthwise, channels)                    
                    continue
    else:
        print("None", end="")
    return channels

# main read file function
def read_onnx(onnx_file):
    print(f"Loading model {onnx_file}...")
    model = onnx.load(onnx_file)
    print("Checking model...")
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    print("Printing metadata...\n")

    nodes = model.graph.node
    count = 0
    for node in nodes:
        count = count +1
        channels = 0
        print(f"{count}        {node.op_type}")
        print(f"input_name:        {node.input}")
        print(f"input_shape:       ", end="")

        check_depthwise_conv = if_Conv(node)
        check_pool = if_PoolCheck(node)

        # INPUT SHAPE
        input_channels = parse_node(model, node.input, model.graph.input, channels, True, check_depthwise_conv)

        print(f"\noutput_name:       {node.output}")
        print(f"output_shape:      ", end="")

        # OUTPUT SHAPE
        output_channels = parse_node(model, node.output, model.graph.output, channels,  False, check_depthwise_conv)
        
        if (check_depthwise_conv == True):
            print_attr_conv(node, input_channels, output_channels)
        elif (check_pool == True):
            print_attr_pool(node)

        print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args included to enable adding commands
    args, arg_fname = parser.parse_known_args()

    for onnx_file in arg_fname:
        read_onnx(onnx_file)