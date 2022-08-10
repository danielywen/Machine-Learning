# Usage
```python
python3 CIFAR100TRAINING.py
```

# Description
Takes in CIFAR100 as input image dataset to train on from scratch.  Time it takes to finish can be made faster by changing epochs and learning rate,
but that may affect performance.  This custom model outperforms most pre-trained models, such as ResNet50, Inception, MobileNet.  However, the model 
was made specifically to trian on CIFAR100 and will not outperform pre-trained models on other training sets.  This model serves as a way to get
comfortable with TensorFlow and convolutional neural networks.

Some noticeable features of the custom model include: data pre-processing, data augmentation, and a learning rate scheduler. 

# Comments
Commented code serves as past modifications made to the model that were eventually cut out.  The commented code under 'Sequential model' is the same
as the model above it in a different format.

