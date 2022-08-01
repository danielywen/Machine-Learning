import onnx
import argparse
import pdb
import os
from onnx import helper

# x = node.input OR node.output
# j = model.graph.value_info OR model.graph.input OR model.graph.output OR model.graph.initializer
# k = newValueInfo OR newGraphInputs OR newGraphOutputs OR newInitializers
def add_to_new_graph(x, j, k, iter, exists):
    # Search through j and see if we need to add to new model k
    for count in range(len(j)):
        if (x[iter] == j[count].name):
            # Check that x[iter] does not already exist in the new model k
            for count_check in range(len(k)):
                if x[iter] == k[count_check].name:
                    exists = True
            if exists == False:
                k.append(j[count])

            exists = False
    return exists


# x = node.input OR node.output
# y = model.graph.input OR model.graph.output
# input_or_output = True if input, False if output
def parse_node(model, x, y, input_or_output, newInitializers, newValueInfo, newGraphInputs, newGraphOutputs):
    # Check that the layer has inputs/outputs
    if len(x) >= 1:
        # iterate through this particular node's input/output list.  e.g. ['input_node0', 'input_node1', 'input_node2']
        for iter in range(len(x)):   
            exists = False
            # Search through value info
            exists = add_to_new_graph(x, model.graph.value_info, newValueInfo, iter, exists)
            # Search through graph inputs or outputs
            if input_or_output == True:
                exists = add_to_new_graph(x, model.graph.input, newGraphInputs, iter, exists)
            elif input_or_output == False:
                exists = add_to_new_graph(x, model.graph.output, newGraphOutputs, iter, exists)
            # Search through initializers
            exists = add_to_new_graph(x, model.graph.initializer, newInitializers, iter, exists)
    return 

# Due to version inconsistencies, the "axes" attribute is moved to the node input in ONNX versions > 11
def version_fix(model, node):
    if (node.op_type == 'ReduceSum') or (node.op_type == 'Unsqueeze') or (node.op_type == 'Squeeze'):
        if node.attribute[0].name == 'axes':
            node.input.append(node.attribute[0].name)
            tensor = helper.make_tensor(node.attribute[0].name, node.attribute[0].type, [], node.attribute[0].ints)
            model.graph.initializer.append(tensor)
            del node.attribute[0]

# Stores all inputs or outputs of each node in new model
# x = node.input OR node.output
# y = checkGraphInputs OR checkGraphOutputs
def all_node_in_outs(x, y, model):
    checkOldGraph = []
    for in_node in x:
        if in_node not in y:
            check = False
            # check that the node input isn't already a graph input
            for graph_inp_node in model.graph.input:
                if graph_inp_node.name == in_node:
                    check = True
            # check that the node input isn't already a graph initializer
            for graph_init_node in model.graph.initializer:
                if graph_init_node.name == in_node:
                    check = True
            if check == False:
                checkOldGraph.append(in_node)
    for node2 in checkOldGraph:
        y.append(node2)

# Finds the new graph inputs and outputs of the new model
# x = checkGraphOutputs OR checkGraphInputs
# y = whichever x isn't
# z = finalGraphOutputs or finalGraphInputs
def find_new_in_outs(x, y, z):
    # any node output that isn't in the node input list is a new graph output
    # any node input that isn't in the node output list is a new graph input
    for node in x:
        if node not in y:
            z.append(node)

# Adds the new graph inputs and outputs to the new model
# x = finalGraphInputs OR finalGraphOutputs
# y = newGraphInputs OR newGraphOutputs
def add_in_out_to_graph(x, y, model):
    for add_input in x:
        for value_info in model.graph.value_info:
            repeat = False
            for check_repeat in y:
                if check_repeat.name == add_input:
                    repeat = True
            if repeat == False:
                if add_input == value_info.name:
                    y.append(value_info)

# Main split model function
def split_onnx(onnx_file, args):
    print(f"Loading model {onnx_file}...")
    model = onnx.load(onnx_file)
    print("Checking model...")
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    print('Splitting model...')
    nodes = model.graph.node

    # Error checks
    if args.split[1] > len(nodes) + 1:
        print('ERROR: End argument is larger than graph size')
        return
    elif args.split[0] < 1:
        print('ERROR: Start argument is less than 1')
        return
    elif args.split[0] > args.split[1]:
        print('ERROR: Start argument is larger than end argument')
        return
    elif args.split[0] == args.split[1]:
        print('ERROR: Start argument is equal to the end argument')
        return

    count = 0
    start =  args.split[0]
    end = args.split[1]
    newNodes = []
    newInitializers = []
    newValueInfo = []
    newGraphInputs = []
    newGraphOutputs = []
    checkGraphInputs = []
    checkGraphOutputs = []

    for node in nodes:
        count = count + 1
        if (count == start) and (start != end + 1):
            start = start + 1
            version_fix(model, node)
            all_node_in_outs(node.input, checkGraphInputs, model)
            all_node_in_outs(node.output, checkGraphOutputs, model)
            parse_node(model, node.input, model.graph.input, True, newInitializers, newValueInfo, newGraphInputs, newGraphOutputs)
            parse_node(model, node.output, model.graph.output, False, newInitializers, newValueInfo, newGraphInputs, newGraphOutputs)
            newNodes.append(node)
    
    finalGraphOutputs = []
    finalGraphInputs = []
    find_new_in_outs(checkGraphOutputs, checkGraphInputs, finalGraphOutputs)
    find_new_in_outs(checkGraphInputs, checkGraphOutputs, finalGraphInputs)

    add_in_out_to_graph(finalGraphInputs, newGraphInputs, model)
    add_in_out_to_graph(finalGraphOutputs, newGraphOutputs, model)

    # get current path
    path = os.getcwd()
    # create new path
    new_path = path + '/' + args.name[0]

    new_graph = helper.make_graph(newNodes, 'New Graph', newGraphInputs, newGraphOutputs, initializer=newInitializers, value_info=newValueInfo)
    new_model = helper.make_model(new_graph, 
                opset_imports=[
                helper.make_opsetid("", version=15),
                helper.make_opsetid("com.microsoft.experimental", version=1),
                helper.make_opsetid("ai.onnx.ml", version=2),
                helper.make_opsetid("ai.onnx.training", version=1),
                helper.make_opsetid("ai.onnx.preview.training", version=1),
                helper.make_opsetid("com.microsoft", version=1),
                helper.make_opsetid("com.microsoft.nchwc", version=1),
            ])

    print('Checking new model...')
    onnx.checker.check_model(new_model)
    onnx.save(new_model, new_path)
    print('Finished!')
    print(f'Saved to: {new_path}')
    
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', action='store', type=int, nargs=2, help='Specify where to split in format: start end')
    parser.add_argument('--name', '-n', action='store', type=str, nargs=1, default=['new_model.onnx'], help='Specify name of new file. Default: new_model.onnx')
    args, arg_fname = parser.parse_known_args()

    for onnx_file in arg_fname:
        split_onnx(onnx_file, args)
