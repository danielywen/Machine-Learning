# Usage
```python
python3 onnx_read.py <ONNX file>
```

# Description
Takes in an ONNX file as input and outputs visualization of model metadata.  This aids developers in quickly visualizing the model they built with one quick command without having to import their new model into an external model visualizer, such as Netron.
Created by Daniel Wen for use by Expedera, Inc.

# Supported Layers
Any layers not listed below will still be supported; however, their attributes may not be printed.

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
