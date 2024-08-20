# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import add
from tensorflow.keras import backend as K
import numpy as np
from inputparameters import *

# taken from https://keras.io/examples/vision/autoencoder/
class UNet:
	@staticmethod
	def build(width, height, depth, filters=(32, 64), latentDim=16, eta=0.5):

		padd = "same"
		#padd = "valid"
         
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs

		# loop over the number of filters
		index = 0
		for f in filters:
                        print("layer ",f)
                        if index == 0:
                          x = Conv2D(f, (3, 3), activation="relu", strides=2, padding=padd)(x) #half the input size
                          x = BatchNormalization(axis=chanDim)(x)
                          previous_block_activation = x                          
                        else:
                          # apply a CONV => RELU => POOLING => BN operation
                          # first convolutional layer
                          x = Conv2D(f, (3, 3), activation="relu", strides=1, padding=padd)(x)
                          x = BatchNormalization(axis=chanDim)(x)
                          # second convolutional layer
                          x = Conv2D(f, (3, 3), activation="relu", strides=1, padding=padd)(x)
                          x = BatchNormalization(axis=chanDim)(x)
                          # Pooling
                          x = MaxPooling2D((2, 2), padding=padd)(x) #half the input size
                          x = Dropout(eta)(x)
                          # Project residuals
                          residual = Conv2D(f, 1, strides=2, padding=padd)(previous_block_activation) #half the input size
                          x = add([x, residual])  
                          previous_block_activation = x  

                        index = index+1

		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
                        # apply a CONV_TRANSPOSE => RELU => BN operation
                        # first convolutional layer
                        x = Conv2DTranspose(f, (3, 3), activation="relu", strides=1, padding=padd)(x) #same size as input
                        x = BatchNormalization(axis=chanDim)(x)
                        # second convolutional layer
                        x = Conv2DTranspose(f, (3, 3), activation="relu", strides=1, padding=padd)(x)
                        x = BatchNormalization(axis=chanDim)(x)
                        x = UpSampling2D(2)(x)                                #two times the input size
                        # Project residuals
                        residual = UpSampling2D(2)(previous_block_activation) #two times the input size
                        residual = Conv2D(f, 1, padding=padd)(residual)       #same size as input
                        x = add([x, residual]) 
                        previous_block_activation = x

		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		#x = Conv2DTranspose(depth, (3, 3), padding=padd)(x)
		#x = Conv2DTranspose(depth, (3, 3), padding=padd)(x)
                # this was used in the first tests: outputs = Conv2D(depth, (3, 3), activation="sigmoid", padding=padd)(x)
		if buildsky == 0:
		   outputs = Conv2D(2, (1, 1), activation="softmax", padding=padd)(x)
		if buildsky == 1:
		   outputs = Conv2D(1, (1, 1), activation="sigmoid", padding=padd)(x)
		#outputs = Activation("sigmoid")(x)

		# build the decoder model
		##decoder = Model(latentInputs, outputs, name="decoder")
		###decoder = Model(outputs, name="decoder")

		# autoencoder is the encoder + decoder
		autoencoder = Model(inputs, outputs, name="autoencoder")

		# return the autoencoder
		return (autoencoder)
