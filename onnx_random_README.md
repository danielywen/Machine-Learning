# Usage
```python
python3 onnx_random.py
```

# Optional Flags
```
--minmax, -m <MIN> <MAX>  
  Description: Specify the minimum and maximum size of the randomized model.  
  Default: 10 20  
 
--channels, -c <CHANNELS>  
  Description: Specify the input tensor's number of channels. 
  Default: 128
  
--height, -he <HEIGHT>  
  Description: Specify the input tensor's height. 
  Default: 28
  
--width, -w <WIDTH>  
  Description: Specify the input tensor's width
  Default: 28
  
--chmult, -chm <MIN> <MAX>  
  Description: Specify a range for the channel multiplier for randomly generated channels. 
  Default: 1 5  
  
--exclude, -e <LAYER>
  Description: Specify the different types of layers you want excluded from the randomizer.  
  Arguments must be written as followed:  
  conv maxpool relu globalavgpool leakyrelu sigmoid tanh resize transpose pad convtranspose add mul concat gemm  
  Example: -e conv maxpool  
  Default: No layers excluded

--name, -n <NAME>  
  Description: Specify the output model's file name. 
  Default: randomized_model.onnx  
```

# Description
Creates a randomized model given specified (or default) parameters.  

# Supported Layers
Conv  
MaxPool  
Relu  
GlobalAveragePool  
LeakyRelu  
Sigmoid  
Tanh  
Resize  
Transpose  
Pad  
ConvTranspose  
Add  
Mul  
Concat  
Gemm  

# Future Modifications
Uncomment the print statements for debugging.  
If support for Sub and Div is needed:  
    1. Add a case for them in create_auto_layer() in the same way as Add and Mul.  
    2. Add "sub_num" and "div_num" to layerCounters.  
    3. Create a create_sub_layer() and create_div_layer() that calls create_binary_layer() the same way as create_add_layer() and create_mul_layer().  
    4. Add a case for them in increment_layer_counter() in the same way as Add and Mul.  
    5. Add a case for them in current_layer_name() in the same way as Add and Mul.  
    6. Add create_sub_layer() and create_div_layer() in all_layers the same was as Add and Mul.  
    7. Add a case for them in include_layers() in the same way as Add and Mul.

# Concerns
If the model you created is extremely large and you try to visualize it with Netron, sometimes it gives an error "File format is no onnx.ModelProto".  
This means the file is corrupted somehow and I fixed this by just rerunning the randomizer.
