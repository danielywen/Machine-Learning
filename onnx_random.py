import onnx
import random
import operator
import numpy as np
import argparse
import pdb

from onnx_local.onnx_utils import (
    Parameter,
    BiasParameter,
    WeightParameter,
    ZeroPointParameter,
    ScaleParameter,
    InputParameter,
    OutputParameter,
    ParameterInitializer,
    RandomIntegersBetweenMinMaxInitializer,
    RandomlySampleDefinedValuesInitializer,
    OnesInitializer,
    ZerosInitializer,
    RandomFloatBetweenMinMaxInitializer,
    DefinedValuesInitializer,
)
import onnx
from onnx_local.onnx_helper_classes import OnnxAutoLayer, AutoGraph
from typing import List, Dict, Type
from onnx_local.onnx_auto_layers import (
    QLinearMatMulAuto,
    QLinearConvAuto,
    QuantizeLinearAuto,
    DequantizeLinearAuto,
    QLinearLeakyReluAuto,
    QLinearConcatAuto,
    QLinearAddAuto,
    QLinearAveragePoolAuto,
    QLinearGlobalAveragePoolAuto,
    QLinearMulAuto,
    ConvAuto,
    GlobalAveragePoolAuto,
    LeakyReluAuto,
    MaxPoolAuto,
    ReluAuto,
    SigmoidAuto,
    TanhAuto,
    ReshapeAuto,
    LRNAuto,
    PadAuto,
    TransposeAuto,
    ResizeAuto,
    GemmAuto,
    ConvTransposeAuto,
    AddAuto,
    MulAuto,
    ConcatAuto,
)
from onnx import TensorProto

layer_def: Dict[int, str] = {
    1: "Add",
    2: "Conv",
    3: "Gemm",
    4: "LeakyRelu",
    5: "MaxPool",
    6: "Mul",
    7: "Relu",
    8: "Pad",
    9: "Sigmoid",
    10: "Tanh",
    11: "Transpose",
    12: "Resize",
    13: "Concat",
    14: "GlobalAveragePool",
}

# map of graph inputs
inputConnections: Dict[str, List[str]] = {
    # Example inputs:
    # "input0" : ["conv0"],
    # "input1" : ["maxpool3"]
}

# map of all the layers' connections
connectionMap: Dict[str, List[str]] = {
    # Example inputs:
    # "conv1": ["conv2"],
    # "conv2": [],
}

# map of actual layers (must always be updated with connectionMap)
layers: List[OnnxAutoLayer] = [
    # Example inputs:
    # ConvAuto("conv1", output_channels=128),
    # ConvAuto("conv2", output_channels=32)
]

# tracks the number of each layer for naming convention
layerCounters: Dict[str, int] = {
    "conv_num": 0,
    "maxpool_num": 0,
    "relu_num": 0,
    "globalavgpool_num": 0,
    "leakyrelu_num": 0,
    "sigmoid_num": 0,
    "tanh_num": 0,
    "resize_num": 0,
    "transpose_num": 0,
    "pad_num": 0,
    "convtranspose_num": 0,
    "add_num": 0,
    "mul_num": 0,
    "concat_num": 0,
    "gemm_num": 0,
    "reshape_num": 0,
}

# stores the name and the shape of each layer
layerInfo: List[List] = [
    # * Note: all layers should appear in this format except Reshape layers
    # * which are 2D, but they should only be created for Gemm operations
    # Example inputs:
    # ["conv0", 1, 64, 28, 24],
    # ["maxpool0", 1, 64, 24, 28]
]

# inp_width = 64
# inp_height = 64
# inp_channels = 64
# kernel_size = 1
# strides = 1
# padding = 'same'
# layers = 1
# ch_mult = 1

# function for all auto layers. Can add more cases or alter existing ones as needed
def create_auto_layer(name, number, current_height, current_width):
    if name == "conv":
        k, s, p = random_kernel_stride_pad()
        f = random_channels()
        return ConvAuto(
            name + str(number),
            output_channels=f,
            kernel_shape=[k, k],
            pads=[p, p, p, p],
            strides=[s, s],
        )
    elif name == "maxpool":
        k, s, p = random_kernel_stride_pad()
        k = 3
        return MaxPoolAuto(
            name + str(number), kernel_shape=[k, k], pads=[p, p, p, p], strides=[s, s]
        )
    elif name == "relu":
        return ReluAuto(name + str(number))
    elif name == "globalavgpool":
        return GlobalAveragePoolAuto(name + str(number))
    elif name == "leakyrelu":
        return LeakyReluAuto(name + str(number))
    elif name == "sigmoid":
        return SigmoidAuto(name + str(number))
    elif name == "tanh":
        return TanhAuto(name + str(number))
    elif name == "resize":
        resize_height_ratio = multiply_shape(current_height)
        resize_width_ratio = multiply_shape(current_width)
        return ResizeAuto(
            name + str(number),
            resize_to=[1, 1, resize_height_ratio, resize_width_ratio],
            ratios=True,
        )
    elif name == "transpose":
        return TransposeAuto(name + str(number), perm=[0, 1, 3, 2])
    elif name == "pad":
        height_pad = 3
        width_pad = 3
        return PadAuto(
            name + str(number),
            pads=[0, 0, height_pad, width_pad, 0, 0, height_pad, width_pad],
        )
    elif name == "convtranspose":
        k, s, p = random_kernel_stride_pad()
        f = random_channels()
        return ConvTransposeAuto(
            name + str(number),
            output_channels=f,
            kernel_shape=[k, k],
            pads=[p, p, p, p],
            strides=[s, s],
        )
    elif name == "add":
        return AddAuto(name + str(number))
    elif name == "mul":
        return MulAuto(name + str(number))
    elif name == "concat":
        return ConcatAuto(name + str(number))
    elif name == "gemm":
        # * Should not ever reach here, but keeping here for modifiability
        # * Currently, there is a separate create_gemm_layer()
        pass


# Calls setGraphInput() to extract the batch, channels, height, and width of current layer to record in layerInfo
def record_cur_chan_height_width(name, number, graph_inputs):
    temp_graph = AutoGraph(layers, connectionMap=connectionMap)
    assert len(graph_inputs) == len(set([k.name for k in graph_inputs]))
    temp_graph.setGraphInput(
        graph_inputs, graph_input_str_to_node_strs=inputConnections
    )
    output_batch = (
        temp_graph.calculate_value_infos()[-1].type.tensor_type.shape.dim[0].dim_value
    )
    output_channels = (
        temp_graph.calculate_value_infos()[-1].type.tensor_type.shape.dim[1].dim_value
    )
    output_height = (
        temp_graph.calculate_value_infos()[-1].type.tensor_type.shape.dim[2].dim_value
    )
    output_width = (
        temp_graph.calculate_value_infos()[-1].type.tensor_type.shape.dim[3].dim_value
    )
    layerInfo.append(
        [name + str(number), output_batch, output_channels, output_height, output_width]
    )
    return


# Same as record_cur_chan_height_width() except no need to call setGraphInput since we have the metadata already
def gemm_layer_record_cur_chan_height_width(
    name,
    number,
    graph_inputs,
    prev_layer_info,
    channel_height_width_mul,
    out_channel,
    out_dim,
    gemm_channel,
):
    layerInfo.append(
        [
            "reshape" + str(layerCounters["reshape_num"]),
            prev_layer_info[1],
            channel_height_width_mul,
        ]
    )
    layerInfo.append([name + str(number), channel_height_width_mul, gemm_channel])
    layerInfo.append(
        [
            "reshape" + str(layerCounters["reshape_num"] + 1),
            prev_layer_info[1],
            out_channel,
            out_dim,
            out_dim,
        ]
    )
    return


# Remove the layer
def layer_undo(prev_layer_names):
    connectionMap.popitem()
    layers.pop()
    prev_layer_names.pop()
    # * If the layer is the first layer, no previous layer in connectionMap to update
    # * Instead, remove the input connection
    if len(prev_layer_names) > 0:
        connectionMap.update({str(prev_layer_names[-1]): []})
    else:
        inputConnections.popitem()
        input_counter = input_counter - 1


#  Add Reshape, Gemm, Reshape to layers[] and prev_layers_names[]
def add_reshape_gemm_layers(
    name,
    number,
    prev_layer_names,
    channel_height_width_mul,
    gemm_channel,
    prev_layer_info,
    out_channel,
    out_dim,
):
    layers.append(
        ReshapeAuto(
            "reshape" + str(layerCounters["reshape_num"]),
            new_shape=(prev_layer_info[1], channel_height_width_mul),
        )
    )
    layers.append(GemmAuto(name + str(number), channel_height_width_mul, gemm_channel))
    layers.append(
        ReshapeAuto(
            "reshape" + str(layerCounters["reshape_num"] + 1),
            new_shape=(prev_layer_info[1], out_channel, out_dim, out_dim),
        )
    )
    prev_layer_names.append("reshape" + str(layerCounters["reshape_num"]))
    prev_layer_names.append(name + str(number))
    prev_layer_names.append("reshape" + str(layerCounters["reshape_num"] + 1))


# Matrix multiplcation calculation
def gemm_layer_calculate(previous_layer_name, first_layer):
    # * Parse through the list of layers and get the metadata of the previous layer
    if first_layer == False:
        channel_height_width_mul = 0
        prev_layer_info = layerInfo[0]
        for curr_layer in layerInfo:
            # * This check is so we don't include the 2D layers from Reshape + Gemm connections
            if len(curr_layer) == 5:
                if curr_layer[0] == previous_layer_name:
                    prev_layer_info = curr_layer
                    channel_height_width_mul = (
                        curr_layer[2] * curr_layer[3] * curr_layer[4]
                    )
    # * If the current layer is the first layer, get the input layer instead
    else:
        channel_height_width_mul = 0
        prev_layer_info = []
        for curr_layer in graph_inputs:
            if curr_layer.name == previous_layer_name:
                prev_layer_info.append(curr_layer.name)
                for dim in curr_layer.shape:
                    prev_layer_info.append(dim)
                channel_height_width_mul = (
                    prev_layer_info[2] * prev_layer_info[3] * prev_layer_info[4]
                )
    # * Randomly choose the input channels, output channels, and output dimensions for matrix multiplcation calculation
    works = False
    while works == False:
        gemm_channel = random.randint(1, 512)
        out_dim = random.randint(1, 16)
        out_channel = random.randint(1, 128)
        if gemm_channel == out_channel + (out_dim * out_dim):
            works = True
            # print(f"{gemm_channel} = {out_channel} + {out_dim} * {out_dim}")
    return channel_height_width_mul, gemm_channel, prev_layer_info, out_channel, out_dim


# Same as layer_undo() except has to remove three layer instead of one
# Gemm operation actually uses Reshape, Gemm, Reshape
def gemm_layer_undo(prev_layer_names):
    global input_counter
    # * This is undoing the layer
    for i in range(3):
        connectionMap.popitem()
        layers.pop()
        prev_layer_names.pop()
    # * If it is the first layer
    if len(prev_layer_names) > 0:
        connectionMap.update({str(prev_layer_names[-1]): []})
    else:
        inputConnections.popitem()
        input_counter = input_counter - 1


# Create Gemm layer, which needs a Reshape layer beforehand to convert previous layer from 4D to 2D
# and another Reshape layer afterwards to convert Gemm from 2D back to 4D
def create_gemm_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    number = layerCounters[name + "_num"]
    global input_counter
    global flag
    if undo == 0:
        # * If not the first layer
        if len(prev_layer_names) > 0:
            previous_layer_name = prev_layer_names[-1]
            # * Connect prev layer with reshape1, reshape1 with gemm, then gemm with reshape2
            connectionMap.update(
                {
                    str(prev_layer_names[-1]): [
                        "reshape" + str(layerCounters["reshape_num"])
                    ]
                }
            )
            connectionMap.update(
                {"reshape" + str(layerCounters["reshape_num"]): [name + str(number)]}
            )
            connectionMap.update(
                {
                    name
                    + str(number): ["reshape" + str(layerCounters["reshape_num"] + 1)]
                }
            )
            connectionMap.update(
                {"reshape" + str(layerCounters["reshape_num"] + 1): []}
            )

            (
                channel_height_width_mul,
                gemm_channel,
                prev_layer_info,
                out_channel,
                out_dim,
            ) = gemm_layer_calculate(previous_layer_name, False)

            add_reshape_gemm_layers(
                name,
                number,
                prev_layer_names,
                channel_height_width_mul,
                gemm_channel,
                prev_layer_info,
                out_channel,
                out_dim,
            )
            # * This is the maximum channels. Can be changed at the cost of performance time
            if channel_height_width_mul > 1024:
                # print("Deleting GEMM layer")
                flag = True
            assert channel_height_width_mul <= 1024
            gemm_layer_record_cur_chan_height_width(
                name,
                number,
                graph_inputs,
                prev_layer_info,
                channel_height_width_mul,
                out_channel,
                out_dim,
                gemm_channel,
            )
        # * If this is the first layer
        else:
            previous_layer_name = "input" + str(input_counter)

            inputConnections.update(
                {
                    "input"
                    + str(input_counter): [
                        "reshape" + str(layerCounters["reshape_num"])
                    ]
                }
            )
            input_counter = input_counter + 1
            connectionMap.update(
                {"reshape" + str(layerCounters["reshape_num"]): [name + str(number)]}
            )
            connectionMap.update(
                {
                    name
                    + str(number): ["reshape" + str(layerCounters["reshape_num"] + 1)]
                }
            )
            connectionMap.update(
                {"reshape" + str(layerCounters["reshape_num"] + 1): []}
            )

            (
                channel_height_width_mul,
                gemm_channel,
                prev_layer_info,
                out_channel,
                out_dim,
            ) = gemm_layer_calculate(previous_layer_name, True)

            # * This is the maximum channels. Can be changed at the cost of performance time
            if channel_height_width_mul > 1024:
                # print("Deleting GEMM layer")
                flag = True
            add_reshape_gemm_layers(
                name,
                number,
                prev_layer_names,
                channel_height_width_mul,
                gemm_channel,
                prev_layer_info,
                out_channel,
                out_dim,
            )
            # * Call assert() to error back out to build_model()
            assert channel_height_width_mul <= 1024
            gemm_layer_record_cur_chan_height_width(
                name,
                number,
                graph_inputs,
                prev_layer_info,
                channel_height_width_mul,
                out_channel,
                out_dim,
                gemm_channel,
            )
    # * Remove layer, but it errored out before appending to layerInfo, so no need to remove from it
    elif undo == 1:
        gemm_layer_undo(prev_layer_names)
    # * Remove layer and layerInfo
    else:
        for i in range(2):
            layerInfo.pop()
        gemm_layer_undo(prev_layer_names)
    return


# Create normal layer (not a binary or Gemm layer)
def create_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    number = layerCounters[name + "_num"]
    global input_counter
    if undo == 0:
        # * If not the first layer
        if len(prev_layer_names) > 0:
            # * Make this layer the last layer (has no output yet)
            connectionMap.update({name + str(number): []})
            # * Connect previous layer to this layer
            connectionMap.update({str(prev_layer_names[-1]): [name + str(number)]})
            layers.append(
                create_auto_layer(name, number, current_height, current_width)
            )
            prev_layer_names.append(name + str(number))
            record_cur_chan_height_width(name, number, graph_inputs)
        # * If it is the first layer
        else:
            inputConnections.update(
                {"input" + str(input_counter): [name + str(number)]}
            )
            input_counter = input_counter + 1
            connectionMap.update({name + str(number): []})
            layers.append(
                create_auto_layer(name, number, current_height, current_width)
            )
            prev_layer_names.append(name + str(number))
            record_cur_chan_height_width(name, number, graph_inputs)
    # * Remove layer, but it errored out before appending to layerInfo, so no need to remove from it
    elif undo == 1:
        layer_undo(prev_layer_names)
    # * Remove layer and layerInfo
    else:
        layerInfo.pop()
        layer_undo(prev_layer_names)
    return


# Similar to layer_undo() except pops inputConnections[] twice and has a case for len(matching_hw_layer_names) == 1
def binary_layer_undo(prev_layer_names, matching_hw_layer_names):
    # * This is undoing the layer
    connectionMap.popitem()
    layers.pop()
    prev_layer_names.pop()
    if len(prev_layer_names) > 0:
        connectionMap.update({str(prev_layer_names[-1]): []})
    # * if is the first layer
    if len(prev_layer_names) == 0:
        inputConnections.popitem()
        inputConnections.popitem()
        input_counter = input_counter - 2
    # * If there was no layer with the same shape found so a new graph input had to be made.
    # * In this case, when create_add_layer() is called again to undo the layer, matching_hw_layer_names
    # * should now have 1 layer1 of the same shape: the layer that was just created
    elif len(matching_hw_layer_names) == 1:
        inputConnections.popitem()
        input_counter = input_counter - 1


# Create binary layer for skip connection (add, mul)
def create_binary_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    number = layerCounters[name + "_num"]
    matching_hw_layer_names = check_hw(
        current_channels, current_height, current_width, False
    )
    global input_counter
    if undo == 0:
        # * if this layer isn't the first layer of the model
        if len(prev_layer_names) > 0:
            # * If there are more than one layers with the same shape (including current layer)
            if len(matching_hw_layer_names) > 1:
                previous_layer_name = matching_hw_layer_names[-1]
                # print(f"The previous layer is: {previous_layer_name}")
                # * Removing current layer from the possible matched layers, since the current layer is one of the layers being added already
                matching_hw_layer_names.pop()
                random_matching_layer_name = random.choice(matching_hw_layer_names)
                # print(
                #     f"The randomly chosen matching layer is: {random_matching_layer_name}"
                # )
                # print(matching_hw_layer_names)

                connectionMap.update({str(prev_layer_names[-1]): [name + str(number)]})
                connectionMap.update({name + str(number): []})
                multiple_outputs = [
                    i for i in connectionMap[random_matching_layer_name]
                ]
                multiple_outputs.append(name + str(number))
                connectionMap.update({random_matching_layer_name: multiple_outputs})
                layers.append(
                    create_auto_layer(name, number, current_height, current_width)
                )
                prev_layer_names.append(name + str(number))
                record_cur_chan_height_width(name, number, graph_inputs)
            else:
                # * This means that there are no layers of the same shape found. In this case, create new graph input to add to the current layer
                # print(
                #     f"NO MATCHINGS SHAPES OF ({current_channels}, {current_height}, {current_width}) found, creating new graph input..."
                # )
                graph_input = InputParameter(
                    "input" + str(input_counter),
                    TensorProto.FLOAT,
                    [1, current_channels, current_height, current_width],
                )
                graph_inputs.append(graph_input)
                inputConnections.update(
                    {"input" + str(input_counter): [name + str(number)]}
                )
                input_counter = input_counter + 1
                connectionMap.update({name + str(number): []})
                connectionMap.update({str(prev_layer_names[-1]): [name + str(number)]})
                layers.append(
                    create_auto_layer(name, number, current_height, current_width)
                )
                prev_layer_names.append(name + str(number))
                record_cur_chan_height_width(name, number, graph_inputs)
        else:
            # * This means the layer is the first layer. In this case, create new graph input be 2nd input of this layer
            # print(f"{name + str(number)} IS THE FIRST LAYER")
            inputConnections.update(
                {"input" + str(input_counter): [name + str(number)]}
            )
            input_counter = input_counter + 1
            graph_input = InputParameter(
                "input" + str(input_counter),
                TensorProto.FLOAT,
                [1, current_channels, current_height, current_width],
            )
            graph_inputs.append(graph_input)
            inputConnections.update(
                {"input" + str(input_counter): [name + str(number)]}
            )
            input_counter = input_counter + 1
            connectionMap.update({name + str(number): []})
            layers.append(AddAuto(name + str(number)))
            prev_layer_names.append(name + str(number))
            record_cur_chan_height_width(name, number, graph_inputs)
    # * Remove layer, but it errored out before appending to layerInfo, so no need to remove from it
    elif undo == 1:
        binary_layer_undo(prev_layer_names, matching_hw_layer_names)
    # * Remove layer and layerInfo
    else:
        layerInfo.pop()
        binary_layer_undo(prev_layer_names, matching_hw_layer_names)
    return


# Create concat layer, which needs to randomly choose between concatenating the previous layer with a random
# existing layer, or concatenating the previous layer with itself.
# If the former is chosen and there are no layers with the same shape, create a new graph input to concatenate with
def create_concat_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    number = layerCounters[name + "_num"]
    matching_hw_layer_names = check_hw(
        current_channels, current_height, current_width, True
    )
    # * if 0, concat prev_layer on itself n amount of times
    # * if 1, concat prev_layer on random layer (creating a skip connection)
    skip_connection_or_prev_layer = random.randint(0, 1)
    global input_counter
    if undo == 0:
        # * If this layer isn't the first layer of the model
        if len(prev_layer_names) > 0:
            # * If there are more than one layers with the same shape (including current layer)
            if len(matching_hw_layer_names) > 1:
                previous_layer_name = matching_hw_layer_names[-1]
                # print(f"The previous layer is: {previous_layer_name}")
                # * Removing current layer from the possible matched layers, since the current layer is one of the layers being added already
                matching_hw_layer_names.pop()
                random_matching_layer_name = random.choice(matching_hw_layer_names)
                # print(
                #     f"The randomly chosen matching layer is: {random_matching_layer_name}"
                # )
                # print(matching_hw_layer_names)

                # * if 1, concat prev_layer on random layer (creating a skip connection)
                if skip_connection_or_prev_layer == 1:
                    # print("CONCAT PREVIOUS LAYER WITH RANDOM LAYER")
                    connectionMap.update(
                        {str(prev_layer_names[-1]): [name + str(number)]}
                    )
                    connectionMap.update({name + str(number): []})
                    multiple_outputs = [
                        i for i in connectionMap[random_matching_layer_name]
                    ]
                    for i in range(random.randint(1, 5)):
                        multiple_outputs.append(name + str(number))
                    connectionMap.update({random_matching_layer_name: multiple_outputs})
                    layers.append(
                        create_auto_layer(name, number, current_height, current_width)
                    )
                    prev_layer_names.append(name + str(number))
                    record_cur_chan_height_width(name, number, graph_inputs)
                # * concat prev_layer on itself n amount of times. Set to 5 but can be changed
                else:
                    # print("CONCAT PREVIOUS LAYER WITH ITSELF")
                    multiple_outputs = [
                        i for i in connectionMap[str(prev_layer_names[-1])]
                    ]
                    for i in range(random.randint(2, 5)):
                        multiple_outputs.append(name + str(number))
                    connectionMap.update({str(prev_layer_names[-1]): multiple_outputs})
                    connectionMap.update({name + str(number): []})
                    layers.append(
                        create_auto_layer(name, number, current_height, current_width)
                    )
                    prev_layer_names.append(name + str(number))
                    record_cur_chan_height_width(name, number, graph_inputs)
            else:
                # * This means that there are no layers of the same shape found. In this case, just concat with previous layer repeatedly
                # print("NO MATCHING SHAPES FOUND, CONCAT PREVIOUS LAYER WITH ITSELF")
                multiple_outputs = [i for i in connectionMap[str(prev_layer_names[-1])]]
                for i in range(random.randint(2, 5)):
                    multiple_outputs.append(name + str(number))
                connectionMap.update({str(prev_layer_names[-1]): multiple_outputs})
                connectionMap.update({name + str(number): []})
                layers.append(
                    create_auto_layer(name, number, current_height, current_width)
                )
                prev_layer_names.append(name + str(number))
                record_cur_chan_height_width(name, number, graph_inputs)
        else:
            # * This means the layer is the first layer. In this case, concat graph input with itself.
            # print("CONCAT IS FIRST LAYER, CONCAT INPUT LAYER WITH ITSELF")
            multiple_outputs = []
            if len(inputConnections) > 0:
                append_input = random.randint(0, len(inputConnections))
                multiple_outputs = [i for i in inputConnections[append_input]]
            for i in range(random.randint(2, 5)):
                multiple_outputs.append(name + str(number))
            if len(inputConnections) > 0:
                inputConnections.update({"input" + str(append_input): multiple_outputs})
            else:
                inputConnections.update(
                    {"input" + str(input_counter): multiple_outputs}
                )
                input_counter = input_counter + 1
            connectionMap.update({name + str(number): []})
            layers.append(
                create_auto_layer(name, number, current_height, current_width)
            )
            prev_layer_names.append(name + str(number))
            record_cur_chan_height_width(name, number, graph_inputs)
    # * Remove layer, but it errored out before appending to layerInfo, so no need to remove from it
    elif undo == 1:
        # * undoing the layer
        connectionMap.popitem()
        layers.pop()
        prev_layer_names.pop()
        connectionMap.update({str(prev_layer_names[-1]): []})
        # * if is the first layer
        if len(prev_layer_names) == 0:
            inputConnections.popitem()
            input_counter = input_counter - 1
    # * Remove layer and layerInfo
    else:
        # * undoing the layer
        connectionMap.popitem()
        layers.pop()
        prev_layer_names.pop()
        layerInfo.pop()
        connectionMap.update({str(prev_layer_names[-1]): []})
        # * if is the first layer
        if len(prev_layer_names) == 0:
            inputConnections.popitem()
            input_counter = input_counter - 1
    return


# ? NORMAL LAYERS
# All these layers call create_layer().
# They are separated in different functions in case someone wants to modify a specific layer

def create_conv_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_maxpool_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_relu_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_globalavgpool_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_leakyrelu_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_sigmoid_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_tanh_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_resize_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_transpose_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_pad_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_convtranspose_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def multiply_shape(dim):
    ratio_choices = [0.25, 0.5, 2, 3, 4]
    resize_ratio = random.choice(ratio_choices)
    new_shape = int(dim * resize_ratio)
    unacceptable_shape = True
    while unacceptable_shape:
        if (new_shape <= 512) and (new_shape >= 1):
            unacceptable_shape = False
        else:
            new_shape = dim
            resize_ratio = random.choice(ratio_choices)
            new_shape = int(dim * resize_ratio)
    return resize_ratio


# ? BINARY LAYERS
# These layers all call create_binary_layer()
# They are separated in different functions in case someone wants to modify a specific layer

def create_add_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_binary_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def create_mul_layer(
    prev_layer_names,
    layerCounters,
    current_channels,
    current_height,
    current_width,
    graph_inputs,
    name,
    undo,
):
    create_binary_layer(
        prev_layer_names,
        layerCounters,
        current_channels,
        current_height,
        current_width,
        graph_inputs,
        name,
        undo,
    )
    return


def random_kernel_stride_pad():
    kernel = 1 if random.randint(0, 1) else 3
    stride = random.randint(1, 2)
    stride = stride if stride else 1
    # * same = 1, valid = 0
    pad = 1 if random.randint(0, 1) else 0
    return kernel, stride, pad


def random_channels():
    inp_channel = random.randint(8, 16)
    ch_mult = random.randint(mult[0], mult[1])
    return inp_channel * ch_mult


# In order to add two layers, the tensor shapes must be the same
# The channels must also either be the same or one of the tensor has only 1 channel
def check_hw(channels, height, width, concat):
    matching_hw_layer_names = []
    if concat == False:
        for curr_test in layerInfo:
            # * This check is so we don't include the 2D layers from Reshape + Gemm connections
            if len(curr_test) == 5:
                if (curr_test[2] == 1) or (channels == 1) or (curr_test[2] == channels):
                    if curr_test[3] == height:
                        if curr_test[4] == width:
                            # print(
                            #     f"{curr_test[0]} has the same shape of ({curr_test[1]},{channels},{height},{width})!"
                            # )
                            matching_hw_layer_names.append(curr_test[0])
    # * we don't need to check channels for concat, just shape
    else:
        for curr_test in layerInfo:
            if len(curr_test) == 5:
                if curr_test[3] == height:
                    if curr_test[4] == width:
                        # print(
                        #     f"{curr_test[0]} has the same shape of ({height},{width})!"
                        # )
                        matching_hw_layer_names.append(curr_test[0])
    return matching_hw_layer_names


# Increments/Decrements the counters of each layer depending on whether the layer is being added or removed
def increment_layer_counter(chosen_layer, undo):
    if undo == 0:
        op = operator.add
    else:
        op = operator.sub

    if chosen_layer == create_conv_layer:
        layerCounters["conv_num"] = op(layerCounters["conv_num"], 1)
    elif chosen_layer == create_maxpool_layer:
        layerCounters["maxpool_num"] = op(layerCounters["maxpool_num"], 1)
    elif chosen_layer == create_relu_layer:
        layerCounters["relu_num"] = op(layerCounters["relu_num"], 1)
    elif chosen_layer == create_globalavgpool_layer:
        layerCounters["globalavgpool_num"] = op(layerCounters["globalavgpool_num"], 1)
    elif chosen_layer == create_leakyrelu_layer:
        layerCounters["leakyrelu_num"] = op(layerCounters["leakyrelu_num"], 1)
    elif chosen_layer == create_sigmoid_layer:
        layerCounters["sigmoid_num"] = op(layerCounters["sigmoid_num"], 1)
    elif chosen_layer == create_tanh_layer:
        layerCounters["tanh_num"] = op(layerCounters["tanh_num"], 1)
    elif chosen_layer == create_resize_layer:
        layerCounters["resize_num"] = op(layerCounters["resize_num"], 1)
    elif chosen_layer == create_transpose_layer:
        layerCounters["transpose_num"] = op(layerCounters["transpose_num"], 1)
    elif chosen_layer == create_pad_layer:
        layerCounters["pad_num"] = op(layerCounters["pad_num"], 1)
    elif chosen_layer == create_convtranspose_layer:
        layerCounters["convtranspose_num"] = op(layerCounters["convtranspose_num"], 1)
    elif chosen_layer == create_add_layer:
        layerCounters["add_num"] = op(layerCounters["add_num"], 1)
    elif chosen_layer == create_mul_layer:
        layerCounters["mul_num"] = op(layerCounters["mul_num"], 1)
    elif chosen_layer == create_concat_layer:
        layerCounters["concat_num"] = op(layerCounters["concat_num"], 1)
    elif chosen_layer == create_gemm_layer:
        layerCounters["gemm_num"] = op(layerCounters["gemm_num"], 1)
        layerCounters["reshape_num"] = op(layerCounters["reshape_num"], 2)
    return layerCounters


# Separated from increment_layer_counter() due to execution order in build_model()
def current_layer_name(chosen_layer):
    if chosen_layer == create_conv_layer:
        name = "conv"
    elif chosen_layer == create_maxpool_layer:
        name = "maxpool"
    elif chosen_layer == create_relu_layer:
        name = "relu"
    elif chosen_layer == create_globalavgpool_layer:
        name = "globalavgpool"
    elif chosen_layer == create_leakyrelu_layer:
        name = "leakyrelu"
    elif chosen_layer == create_sigmoid_layer:
        name = "sigmoid"
    elif chosen_layer == create_tanh_layer:
        name = "tanh"
    elif chosen_layer == create_resize_layer:
        name = "resize"
    elif chosen_layer == create_transpose_layer:
        name = "transpose"
    elif chosen_layer == create_pad_layer:
        name = "pad"
    elif chosen_layer == create_convtranspose_layer:
        name = "convtranspose"
    elif chosen_layer == create_add_layer:
        name = "add"
    elif chosen_layer == create_mul_layer:
        name = "mul"
    elif chosen_layer == create_concat_layer:
        name = "concat"
    elif chosen_layer == create_gemm_layer:
        name = "gemm"
    return name


def include_layers(all_layers, layers):
    layer_excludes = []
    for i in range(len(layers)):
        if layers[i] == "conv":
            layer_excludes.append(create_conv_layer)
        elif layers[i] == "maxpool":
            layer_excludes.append(create_maxpool_layer)
        elif layers[i] == "relu":
            layer_excludes.append(create_relu_layer)
        elif layers[i] == "globalavgpool":
            layer_excludes.append(create_globalavgpool_layer)
        elif layers[i] == "leakyrelu":
            layer_excludes.append(create_leakyrelu_layer)
        elif layers[i] == "sigmoid":
            layer_excludes.append(create_sigmoid_layer)
        elif layers[i] == "tanh":
            layer_excludes.append(create_tanh_layer)
        elif layers[i] == "resize":
            layer_excludes.append(create_resize_layer)
        elif layers[i] == "transpose":
            layer_excludes.append(create_transpose_layer)
        elif layers[i] == "pad":
            layer_excludes.append(create_pad_layer)
        elif layers[i] == "convtranspose":
            layer_excludes.append(create_convtranspose_layer)
        elif layers[i] == "add":
            layer_excludes.append(create_add_layer)
        elif layers[i] == "mul":
            layer_excludes.append(create_mul_layer)
        elif layers[i] == "concat":
            layer_excludes.append(create_concat_layer)
        elif layers[i] == "gemm":
            layer_excludes.append(create_gemm_layer)
    layer_choices = []
    for i in all_layers:
        if i not in layer_excludes:
            layer_choices.append(i)
    return layer_choices


graph_inputs = []
input_counter = 0
flag = False
# min_num_layers, max_num_layers, inp_channels, inp_height, inp_width, layerCounters
def build_model(args, layerCounters):
    print("Randomizing model...")
    rand_num_layers = random.randint(args.minmax[0], args.minmax[1])
    prev_layer_names = []
    current_channels = args.channels[0]
    current_height = args.height[0]
    current_width = args.width[0]
    global input_counter
    global flag
    graph_input1 = InputParameter(
        "input" + str(input_counter),
        TensorProto.FLOAT,
        [1, args.channels[0], args.height[0], args.width[0]],
    )
    graph_inputs.append(graph_input1)
    # * Choices of layers to randomize
    all_layers = [
        create_conv_layer,
        create_maxpool_layer,
        create_relu_layer,
        create_globalavgpool_layer,
        create_leakyrelu_layer,
        create_sigmoid_layer,
        create_tanh_layer,
        create_resize_layer,
        create_transpose_layer,
        create_pad_layer,
        create_convtranspose_layer,
        create_add_layer,
        create_mul_layer,
        create_concat_layer,
        create_gemm_layer
    ]
    layer_choices = include_layers(all_layers, args.exclude)

    i = 0
    while i < rand_num_layers:
        try:
            i = i + 1
            flag = False
            chosen_layer = random.choice(layer_choices)
            # * Need to add the two reshape layers
            if chosen_layer == create_gemm_layer:
                i = i + 2
            # print(f"Trying {chosen_layer}")
            name = current_layer_name(chosen_layer)
            chosen_layer(
                prev_layer_names,
                layerCounters,
                current_channels,
                current_height,
                current_width,
                graph_inputs,
                name,
                0,
            )
        except:
            undo = 1
        try:
            layerCounters = increment_layer_counter(chosen_layer, 0)
            if flag == True:
                # print("Reshape + Gemm layers' channels are too large")
                i = i - 3
                chosen_layer(
                    prev_layer_names,
                    layerCounters,
                    current_channels,
                    current_height,
                    current_width,
                    graph_inputs,
                    name,
                    1,
                )
                layerCounters = increment_layer_counter(chosen_layer, 1)
            else:
                test_graph = AutoGraph(layers, connectionMap=connectionMap)
                assert len(graph_inputs) == len(set([k.name for k in graph_inputs]))
                test_graph.setGraphInput(
                    graph_inputs, graph_input_str_to_node_strs=inputConnections
                )
                current_channels = (
                    test_graph.calculate_value_infos()[-1]
                    .type.tensor_type.shape.dim[1]
                    .dim_value
                )
                current_height = (
                    test_graph.calculate_value_infos()[-1]
                    .type.tensor_type.shape.dim[2]
                    .dim_value
                )
                current_width = (
                    test_graph.calculate_value_infos()[-1]
                    .type.tensor_type.shape.dim[3]
                    .dim_value
                )
        except:
            # print(
            #     f"Model does not work using {chosen_layer}, currently at shape: ({current_height}, {current_width}). Choosing new layer..."
            # )
            i = i - 1
            if undo == 1:
                chosen_layer(
                    prev_layer_names,
                    layerCounters,
                    current_channels,
                    current_height,
                    current_width,
                    graph_inputs,
                    name,
                    1,
                )
            else:
                chosen_layer(
                    prev_layer_names,
                    layerCounters,
                    current_channels,
                    current_height,
                    current_width,
                    graph_inputs,
                    name,
                    2,
                )
            layerCounters = increment_layer_counter(chosen_layer, 1)

    # print(f"CONNECTION MAP: {connectionMap}")
    # print(f"layerInfo: {layerInfo}")
    # print(f"INPUTCONNECTIONS: {inputConnections}")
    # print("Input shapes in order:")
    # for i in graph_inputs:
    #     print(i.shape)
    # print(f"layerCounters: {layerCounters}")
    graph = AutoGraph(layers, connectionMap=connectionMap)
    assert len(graph_inputs) == len(set([k.name for k in graph_inputs]))
    graph.setGraphInput(graph_inputs, graph_input_str_to_node_strs=inputConnections)
    
    print("Finished!")
    print("Creating model...")
    model = graph.createModel(output_every_layer=False)
    onnx.save(model, args.name[0])
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minmax",
        "-m",
        action="store",
        type=int,
        nargs=2,
        default=[10, 20],
        help="Specify the minimum and maximum size of the model in format: min max. Default = 10 20",
    )
    parser.add_argument(
        "--channels",
        "-c",
        action="store",
        type=int,
        nargs=1,
        default=[128],
        help="Specify the input tensor's number of channels. Default = 128",
    )
    parser.add_argument(
        "--height",
        "-he",
        action="store",
        type=int,
        nargs=1,
        default=[28],
        help="Specify the input tensor's height. Default = 28",
    )
    parser.add_argument(
        "--width",
        "-w",
        action="store",
        type=int,
        nargs=1,
        default=[28],
        help="Specify the input tensor's width. Default = 28",
    )
    parser.add_argument(
        "--chmult",
        "-chm",
        action="store",
        type=int,
        nargs=2,
        default=[1, 5],
        help="Specify a range for the channel multiplier for randomly generated channels in format: min max. Default = 1 5",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        action="store",
        type=str,
        nargs="*",
        default=["all"],
        help="Specify the different types of layers you want excluded from the randomizer.  Default = No layers excluded. "
        "Arguments must be written as followed: "
        "conv maxpool relu globalavgpool leakyrelu sigmoid tanh resize transpose pad convtranspose add mul concat gemm "
        "Example: -e conv maxpool",
    )
    parser.add_argument(
        "--name",
        "-n",
        action="store",
        type=str,
        nargs=1,
        default=["randomized_model.onnx"],
        help="Specify the output model's file name. Default = randomized_model.onnx",
    )

    args = parser.parse_args()
    global mult
    mult = []
    mult.append(args.chmult[0])
    mult.append(args.chmult[1])
    # args = vars(args)

    # build_model(30, 50, 128, 28, 28, layerCounters)
    build_model(args, layerCounters)