# Usage
```python
python3 onnx_split.py <ONNX file> -s arg1 arg2
```
    arg1: start of split
    arg2: end of split
    -split can be used instead of -s

# Optional Flags
```
-n <str>  
    str: name of new file  
    DEFAULT: new_model.onnx  
    Model indexing starts at 1 not 0  
    -name can be used instead of -n
```

# Description
Takes in an ONNX file as input and two integers as positional arguments, and outputs new split model.
Pair with onnx_read.py to visualize new model.  
Created by Daniel Wen for use by Expedera, Inc.

# Additional Concerns
Due to ONNX version mismatching,  certain layers such as ReduceSum, Unsqueeze, and Squeeze had to be modified so that their 'axes'
attribute was moved to the node's input.  These layers in ONNX versions <= 11 have 'axes' as an attribute, while in
ONNX versions > 11, 'axes' is a node input.
If you run into this error, compare the versions of the particular layer and add a case in version_fix().
Most if not all of them should already be accounted for though- supported layers include:  

    Add  
    Conv  
    ConvTranspose  
    Gemm  
    LeakyRelu  
    MaxPool  
    Mul  
    QLinearAdd  
    QLinearAveragePool  
    QLinearConv  
    QLinearMatMul  
    QLinearMul  
    QuantizeLinear  
    DequantizeLinear  
    Relu  
    QLinearLeakyRelu  
    QLinearConcat  
    QLinearGlobalAveragePool  
    Reshape  
    Pad  
    Sigmoid  
    Tanh  
    Transpose  
    Resize  
    Concat  
    GlobalAveragePool
