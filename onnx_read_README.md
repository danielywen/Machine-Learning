# Usage
```python
python3 onnx_read.py <ONNX file>
```

# Description
Takes in an ONNX file as input and outputs visualization of model metadata.

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
