from warnings import filters
import tensorflow as tf
import tensorflow.keras.datasets as datasets
from tensorflow.keras.regularizers import l2
from keras.initializers import glorot_uniform


# GPU configs
# Working without memory growth setting cause memory error in my PC
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data() # m=60000, h,w=28

# Pre-processing data
x_train = x_train / 255.0
x_test = x_test / 255.0

data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
  brightness_range=(0.2,1.0),
  rotation_range=20,
  width_shift_range=0.4,
  height_shift_range=0.4,
  zoom_range=0.2,
  horizontal_flip=True,
  featurewise_center=True,
  featurewise_std_normalization=True,
  validation_split=0.1
)

data_aug.fit(x_train)

def identity_block(input_tensor, k_size, filters, stage, block):

  # Defining name basis
  conv_name_base = "res" + str(stage) + block + "_branch"
  bn_name_base = "bn" + str(stage) + block + "_branch"

  # Retrieve filters (channels)
  F1, F2, F3 = filters

  # Save the input value
  input_save = input_tensor

  # First block of main path
  input_tensor = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(input_tensor)
  input_tensor = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(input_tensor)
  input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)

  # Second block of main path
  input_tensor = tf.keras.layers.Conv2D(filters=F2, kernel_size=(k_size,k_size), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(input_tensor)
  input_tensor = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(input_tensor)
  input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)

  # Third block of main path
  input_tensor = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(input_tensor)
  input_tensor = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(input_tensor)

  # Add input_save to the main path and pass it through a RELU activation
  input_tensor = tf.keras.layers.Add()([input_tensor, input_save])
  input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)

  return input_tensor

def convolutional_block(input_tensor, k_size, filters, stage, block, stride):
  
   # defining name basis
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  # Retrive filters
  F1, F2 = filters

  # Save the input tensor
  input_save = input_tensor

  # MAIN PATH
  input_tensor = tf.keras.layers.Conv2D(F1, 1, strides=stride, padding='same', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(input_tensor)
  input_tensor = tf.keras.layers.BatchNormalization(axis=3, name = bn_name_base + '2a')(input_tensor)
  input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
  input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor)

  input_tensor = tf.keras.layers.Conv2D(F1, 1, strides=stride, padding='same', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(input_tensor)
  input_tensor = tf.keras.layers.BatchNormalization(axis=3, name = bn_name_base + '2a')(input_tensor)
  input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
  input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor)  

  # SHORTCUT PATH
  input_save = tf.keras.layers.Conv2D(F2, 1, strides=stride, padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(input_save)
  input_save = tf.keras.layers.BatchNormalization(axis=3, name = bn_name_base + '1')(input_save)
  input_save = tf.keras.layers.Activation(tf.keras.activations.relu)(input_save)

  # ADD LAYER
  input_tensor = tf.keras.layers.Add()([input_tensor, input_save])
  input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)

  return input_tensor


# Functional API model

initial_input_tensor = tf.keras.Input(shape=(32,32, 3))

# STAGE 1
input_tensor = tf.keras.layers.ZeroPadding2D(padding=1)(initial_input_tensor) # 34
input_tensor = tf.keras.layers.Conv2D(64,3,strides=1)(input_tensor) # 32
input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)

# STAGE 2
input_tensor = tf.keras.layers.ZeroPadding2D(padding=1)(input_tensor) # 34
input_tensor = tf.keras.layers.Conv2D(128,3,strides=1)(input_tensor) # 32
input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor) # 16
# input_tensor = convolutional_block(input_tensor, 3, [128, 128], stage=2, block='a', stride=1)
# input_tensor = identity_block(input_tensor, 3, [128, 128, 128], stage=2, block='b')
# input_tensor = identity_block(input_tensor, 3, [128, 128, 128], stage=2, block='c')

# STAGE 3
input_tensor = tf.keras.layers.ZeroPadding2D(padding=1)(input_tensor) # 18
input_tensor = tf.keras.layers.Conv2D(256,3,strides=1)(input_tensor) # 16
input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor) # 8
# input_tensor = convolutional_block(input_tensor, 3, [256, 256], stage=3, block='a', stride=1)
# input_tensor = identity_block(input_tensor, 3, [256, 256, 256], stage=3, block='b')
# input_tensor = identity_block(input_tensor, 3, [256, 256, 256], stage=3, block='c')


# STAGE 4
# input_tensor = tf.keras.layers.ZeroPadding2D(padding=1)(input_tensor) # 10
# input_tensor = tf.keras.layers.Conv2D(512,3,strides=1)(input_tensor) # 8
# input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
# input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
# input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor) # 4
# input_tensor = convolutional_block(input_tensor, 3, [512, 512], stage=4, block='a', stride=1)
# input_tensor = identity_block(input_tensor, 3, [512, 512, 512], stage=4, block='b')
# input_tensor = identity_block(input_tensor, 3, [512, 512, 512], stage=4, block='c')


# STAGE 5
# input_tensor = tf.keras.layers.ZeroPadding2D(padding=1)(input_tensor) # 6
# input_tensor = tf.keras.layers.Conv2D(1028,3,strides=1)(input_tensor) # 4
# input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
# input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
# input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor) # 2
# input_tensor = convolutional_block(input_tensor, 3, [1028, 1028], stage=5, block='a', stride=1)
# input_tensor = identity_block(input_tensor, 3, [1028, 1028, 1028], stage=5, block='b')
# input_tensor = identity_block(input_tensor, 3, [1028, 1028, 1028], stage=5, block='c')

# STAGE 6
# input_tensor = tf.keras.layers.ZeroPadding2D(padding=1)(input_tensor) # 4
# input_tensor = tf.keras.layers.Conv2D(2056,3,strides=1)(input_tensor) # 2
# input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
# input_tensor = tf.keras.layers.Activation(tf.keras.activations.relu)(input_tensor)
# input_tensor = tf.keras.layers.MaxPooling2D()(input_tensor) # 1

# OUTPUT
input_tensor = tf.keras.layers.Flatten()(input_tensor)
input_tensor = tf.keras.layers.Dropout(0.2)(input_tensor)
input_tensor = tf.keras.layers.Dense(256, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation="relu")(input_tensor)
input_tensor = tf.keras.layers.Dropout(0.2)(input_tensor)
input_tensor = tf.keras.layers.Dense(128, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation="relu")(input_tensor)
output = tf.keras.layers.Dense(100, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation="softmax")(input_tensor)

# Sequential model
# model = tf.keras.Sequential()
# model.add(tf.keras.Input(shape=(32,32, 3)))
# # model.add(tf.keras.layers.Reshape((32,32,3)))
# model.add(tf.keras.layers.ZeroPadding2D(padding=1)) # 34
# model.add(tf.keras.layers.Conv2D(64,3,strides=1)) # 32
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

# model.add(tf.keras.layers.ZeroPadding2D(padding=1)) # 33
# model.add(tf.keras.layers.Conv2D(128,3,strides=1)) # 31
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
# model.add(tf.keras.layers.MaxPooling2D()) # 15

# model.add(tf.keras.layers.ZeroPadding2D(padding=1)) # 16
# model.add(tf.keras.layers.Conv2D(256,3,strides=1)) # 14
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
# model.add(tf.keras.layers.MaxPooling2D()) # 7

# model.add(tf.keras.layers.ZeroPadding2D(padding=1)) # 8
# model.add(tf.keras.layers.Conv2D(512,3,strides=1)) # 6
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
# model.add(tf.keras.layers.MaxPooling2D()) # 3

# model.add(tf.keras.layers.ZeroPadding2D(padding=1)) # 4
# model.add(tf.keras.layers.Conv2D(1028,3,strides=1)) # 2
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
# model.add(tf.keras.layers.MaxPooling2D()) # 1

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(256, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation="relu"))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(128, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation="relu"))
# model.add(tf.keras.layers.Dense(100, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation="softmax"))



# model = tf.keras.applications.MobileNet(weights=None, input_shape=(32,32,3), classes=100)

model = tf.keras.Model(inputs=initial_input_tensor, outputs=output, name='functionalAPI_model')

def scheduler(epoch, lr):
  if epoch < 25:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

opt = tf.keras.optimizers.Adam(learning_rate=0.003)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# print(x_train.shape)
# print(y_train[:100])

model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=[callback])

eval = model.evaluate(x_test,y_test)


print("Accuracy:{}, Loss:{}".format(eval[1],eval[0]))
