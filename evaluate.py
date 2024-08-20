# set the matplotlib backend so figures can be saved in the background
#import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
from UNet import UNet
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import numpy as np
import argparse
import glob
import math
import sys
import time
#sys.path.insert(0, '/marconi/home/userexternal/cgheller/pyfits/lib/python3.6/site-packages')
#import pyfits
from inputparameters import *
from mpi_routines import *
from load_data import *
from load_data_lofar import *

tf.get_logger().setLevel('WARNING') #DEBUG, INFO, WARNING, ERROR
# construct the argument parse and parse the arguments
# parse command line arguments
time1 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Read the model file')
args = parser.parse_args()

# model setup
UNet = tf.keras.models.load_model(args.model)
UNet.summary()

# read data
# for simulations use this 
seed = 0
trainXNoisy, trainX, skydata, tilesize = tiling_data(seed)

# for lofar use this
#trainXNoisy, imagename_list, imageheader_list = tiling_data_lofar('/leonardo/home/userexternal/cstuardi/LOTSSdr2psz2/group3/',0)

print(trainXNoisy.shape)
trainXNoisy = np.expand_dims(trainXNoisy, axis=-1)

# apply the network
predicted = UNet.predict(trainXNoisy)
print(predicted.shape)

#number_of_inputs = end_eval - start_eval
number_of_inputs = len(imagename_list)
tilesize = tile_size
tilesh = 2*tiles
t0 = int(tilesize/4)
t1 = t0+int(tilesize/2)
imageid = 0
valX_fits = np.zeros((number_of_inputs,fullimagex,fullimagey))
print(valX_fits.shape)

for ifits in range(number_of_inputs):
   for j in range(tilesh):
      jstart = j*int(tilesize/2)
      jend = jstart+tilesize
      j0 = jstart+t0
      j1 = j0+int(tilesize/2)
      if jend <= fullimagex:
        for k in range(tilesh):
          kstart = k*int(tilesize/2)
          kend = kstart+tilesize
          if kend <= fullimagey:
             k0 = kstart+t0
             k1 = k0+int(tilesize/2)
             valX_fits[ifits,j0:j1,k0:k1] = predicted[imageid,t0:t1,t0:t1,1]
             imageid = imageid + 1

outputfolder = '/leonardo/home/userexternal/cstuardi/radioUNET_output/simulazioni/'
#for simulation use this
save_data(valX_fits,outputfolder)
#for lofar use this
#save_data_lofar(valX_fits,outputfolder,imagename_list,imageheader_list)
