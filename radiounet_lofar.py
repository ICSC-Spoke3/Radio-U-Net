# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from UNet import UNet
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import argparse
import glob
import math
import sys
import time
import random
#sys.path.insert(0, '/marconi/home/userexternal/cgheller/pyfits/lib/python3.6/site-packages')
#import pyfits
import fileinput
from inputparameters import *
from mpi_routines import *
from load_data import *
from load_data_lofar import *

dotraining = 1

tf.get_logger().setLevel('WARNING') #DEBUG, INFO, WARNING, ERROR
# construct the argument parse and parse the arguments
# parse command line arguments
time1 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--hyper_file', type=str, help='Read the hyperparameter file')
parser.add_argument('--ngpus', type=int, help='Number of gpus per node', default=1)
parser.add_argument('--rand', type=int, help='Random seed')
#parser.add_argument('--model', type=str, help='cnn, dda', default="none")
args = parser.parse_args()
seed = args.rand
if USE_MPI == 1:
   comm = MPI.COMM_WORLD
   rank = comm.rank
   comm.bcast(seed,0)
else:
   comm = 0
   rank = 0

# load json file and distribute parameters
hyper_json = read_json(args.hyper_file)
if USE_MPI == 1:
   hyper_dict = distribute_parameters(hyper_json, comm)
else:
   hyper_dict = distribute_parameters(hyper_json)

# set GPU environment
if args.ngpus > 0:
    config_proto = set_gpus()
    if USE_MPI == 1:
       mydevice = '/device:GPU:'+str(divmod(comm.rank, args.ngpus)[1])
    else:
       mydevice = '/device:GPU:'+str(divmod(0, args.ngpus)[1])
    config_proto.gpu_options.allow_growth = True
else:
    config_proto = tf.compat.v1.ConfigProto()
    if USE_MPI == 1:
       mydevice = '/device:CPU:'+str(comm.rank)
    else:
       mydevice = '/device:CPU:'+str(0)
    #config_proto.gpu_options.allow_growth = True


# Read Data
time2 = time.time()
tt1 = np.zeros(1,dtype=int)
tt2 = np.zeros(1,dtype=int)
tt3 = np.zeros(1,dtype=int)
if rank == 0:
   trainXNoisy, trainX, imagenameNoisy_list, imageheaderNoisy_list = tiling_data_lofar(retrainingpath,seed)
   print(trainXNoisy.shape)
   print(trainX.shape)
   #trainsize = len(imagenameNoisy_list)
   #tt1[0] = trainsize
else:
   trainsize = 0
if USE_MPI == 1:
   comm.Bcast(tt1,root=0)
   comm.barrier() #remove this at the end!
   #trainsize = tt1[0]
   #tilesize = trainsize

def discard_emptytiles(array1,array2):
    # Calculate the sum over axes 1 and 2 of the masks
    sums = np.sum(array2, axis=(1, 2))
    # Identify indices of elements with zero sum
    zero_sum_indices = np.where(sums == 0)[0]
    # Calculate the number of elements to discard
    num_elements_to_discard = len(zero_sum_indices)-(len(array2[:,0,0])-len(zero_sum_indices))
    # Generate random indices to discard elements with zero sum
    indices_to_discard = random.sample(zero_sum_indices.tolist(), num_elements_to_discard)
    # Remove elements at the generated indices
    array1 = np.delete(array1, indices_to_discard, axis=0)
    array2 = np.delete(array2, indices_to_discard, axis=0)
    return array1,array2

trainXNoisy,trainX = discard_emptytiles(trainXNoisy,trainX)

print(trainXNoisy.shape)
print(trainX.shape)

if rank > 0:
   trainX = np.empty((tilesize,tile_size,tile_size),dtype=np.int64)
   trainXNoisy = np.empty((tilesize,tile_size,tile_size),dtype=np.float64)


## Create input arrays
#if buildsky == 0:
#   trainX = np.zeros((trainsize,tile_resolution_y,tile_resolution_x),dtype=np.uint8)
#if buildsky == 1:
#   trainX = np.zeros((trainsize,tile_resolution_y,tile_resolution_x))
#trainXNoisy = np.zeros((trainsize,tile_resolution_y,tile_resolution_x))
#
## Load inputs
#if rank == 0:
#   if buildsky == 0:
#      trainX[:,:,:] = maskdata[:,:,:].astype(int)
#   if buildsky == 1:
#      trainX[:,:,:] = maskdata[:,:,:]
#   trainXNoisy[:,:,:] = noisedata[:,:,:]

# resize arrays
trainX = np.expand_dims(trainX, axis=-1)
trainXNoisy = np.expand_dims(trainXNoisy, axis=-1)
print(trainXNoisy.shape)
print(trainX.shape)


# Broadcast data (in case of MPI)
if USE_MPI == 1:
   print("================== ",rank)
   comm.Bcast(trainX, root=0)
   comm.Bcast(trainXNoisy, root=0)

# construct convolutional autoencoder
print("[INFO] building UNet...")

time3 = time.time()
index = 0
hyperlist = []

#with tf.device(mydevice):
if dotraining == 1:
  with tf.compat.v1.Session(config=config_proto) as sess:
     for imodel in hyper_dict['hyperparam']:
         #with tf.device(mydevice):
              #for i in range(1):
              #with tf.compat.v1.Session(config=config_proto) as sess:
              #init = tf.compat.v1.global_variables_initializer()
              #sess.run(init)
              num_epoch = int(imodel['num_epoch'])
              nbatch = int(imodel['nbatch'])
              eta_input = float(imodel['eta_input'])
              dense_layer = 0#int(imodel['dense_layer'])
              deep_model = int(imodel['deepmodel'])
              print(rank," running model ",index," with ",num_epoch," epochs, ",nbatch, " as batch size, "\
                    ,eta_input, " as learning rate")

              #(encoder, decoder, autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (32, 64, 64, 64, 32), dense_layer)
              #if deep_model == 5:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (32, 64, 128, 256, 512), dense_layer)
              #if deep_model == 6:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (64, 128, 256, 512), dense_layer)
              #if deep_model == 4:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (32, 64, 128, 256), dense_layer)
              #if deep_model == 7:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (32, 64, 64, 128), dense_layer)
              #if deep_model == 3:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (64, 128, 256), dense_layer)
              #if deep_model == 8:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (32, 64, 128), dense_layer)
              #if deep_model == 2:
              #   (autoencoder) = UNet.build(tile_resolution, tile_resolution, 1, (32, 64), dense_layer)

	      #start the training from pre-trained weights
              autoencoder=tf.keras.models.load_model(trained_network)

              #add data augmentation layers

             # data_augmentation = tf.keras.Sequential([
             #   RandomFlip("horizontal_and_vertical",input_shape=(tile_resolution,tile_resolution,1)),
             #   RandomRotation(0.2),
             # ])

             # augment_autoencoder = tf.keras.Sequential([
             #   data_augmentation,
             #   autoencoder,
             # ])

              #fine tuning
              fine_tune_at = 66

              for layer in autoencoder.layers[:fine_tune_at]:
                 layer.trainable = False


              if buildsky == 1:
                 opt = Adam(lr=eta_input)
                 #autoencoder.compile(loss="mse", optimizer=opt,metrics=["accuracy"])
                 autoencoder.compile(loss="mse", optimizer=opt)
              if buildsky == 0:
                 opt = RMSprop(learning_rate=eta_input)
                 ########autoencoder.compile(loss="mse", optimizer=opt)
                 #loss_fn = keras.losses.SparseCategoricalCrossentropy()
                 #autoencoder.compile(optimizer=opt, loss="sparse_categorical_crossentropy",metrics=["accuracy"])
                 autoencoder.compile(optimizer=opt, loss="sparse_categorical_crossentropy")
                 #autoencoder.compile(optimizer="rmsprop", loss=loss_fn)

              # train the convolutional autoencoder
              H = autoencoder.fit(trainXNoisy, trainX,
                                  validation_split=0.1,
                                  epochs=num_epoch,
                                  batch_size=nbatch)

              # construct a plot that plots and saves the training history
              N = np.arange(0, num_epoch)
              plt.style.use("ggplot")
              plt.figure()
              loss_aux = H.history["loss"]
              log_loss_aux = np.log10(loss_aux)
              val_loss_aux = H.history["val_loss"]
              log_val_loss_aux = np.log10(val_loss_aux)
              if logoutput == 1:
                 plt.plot(N, log_loss_aux, label="log(train_loss)")
                 plt.plot(N, log_val_loss_aux, label="log(val_loss)")
              else:
                 plt.plot(N, H.history["loss"], label="train_loss")
                 plt.plot(N, H.history["val_loss"], label="val_loss")
              plt.title("Training Loss")
              plt.xlabel("Epoch number")
              plt.ylabel("Loss")
              plt.legend(loc="lower left")
              plotfilename = images+"/plot_rank"+str(rank)+"_"+str(index)+".png"
              print("plot filename: ",plotfilename)
              plt.savefig(plotfilename)
              plt.close()
 
              #hyperstring  = str(rank)+" "+str(deep_model)+" "+str(num_epoch)+" "+str(nbatch)+" "+str(dense_layer)+" "+\
              #               str(eta_input)+" "+str(loss_aux[num_epoch-1])+" "+str(val_loss_aux[num_epoch-1])+"\n"
              #hyperlist.append(hyperstring)

		  # Saving the model
              nouts = 9
              outXNoisy = trainXNoisy[0:nouts,:,:]
              #outXNoisy = np.expand_dims(outXNoisy, axis=-1)
              print("====================== ",outXNoisy.shape)
              prediction = autoencoder.predict(outXNoisy)

              if checkpoint == 1:
                 restartfile = 'augmented-retrained-model-'+str(deep_model)+'-'+str(nbatch)+'-'+str(num_epoch)+'-'+str(eta_input)+'-'+str(fine_tune_at)+'.ckpt'
                 autoencoder.save(restartfile)

     ### write results
     ##if USE_MPI == 1:
     ##   resultfile = "results"+str(comm.rank)+".txt"
     ##else:
     ##   resultfile = "results"+str(0)+".txt"
     ##file2 = open(resultfile,"w")
     ##for i in range (index):
     ##    file2.write(hyperlist[i])
     ##file2.close()

     if USE_MPI == 1:
        print("MPI rank {} waiting".format(comm.rank))
        comm.Barrier()
        print("MPI rank {} passed".format(comm.rank))
     ##if rank == 0:
     ##   outfilename = "fullresults.txt"
     ##   file_list = glob.glob("results*.txt")
     ##   with open(outfilename, 'w') as file:
     ##        input_lines = fileinput.input(file_list)
     ##        file.writelines(input_lines)
     ##   #	file.close()


time4 = time.time()

if rank == 0:
 if plotdata == 1:
  #for tiletoshow in range(trainsize):
  for tiletoshow in range(nouts):

   outfile = images+'test'+str(tiletoshow)+'.png'
   print(outfile)
   plt.figure(1)#,figsize=(20,12))
   plt.subplot(221)
   plt.imshow(trainXNoisy[tiletoshow,:,:],cmap='gray')
   #plt.colorbar(label=r"$\rm S_{x, \ 0.3-2.0 \ keV} \ [cnt]$", orientation="horizontal", fraction=0.04)
   plt.title("Source map")
   plt.subplot(222)
   plt.imshow(trainX[tiletoshow,:,:],cmap='magma')
   #plt.colorbar(label=r"$\rm S_{x, \ 0.3-2.0 \ keV} \ [cnt]$", orientation="horizontal", fraction=0.04)
   plt.title("Mask map")
   plt.subplot(223)
   plt.imshow(prediction[tiletoshow,:,:,0],cmap='magma')
   #plt.colorbar(label=r"$\rm S_{x, \ 0.3-2.0 \ keV} \ [cnt]$", orientation="horizontal", fraction=0.04)
   plt.title("Predicted map 0")
   plt.subplot(224)
   # class 1 is "probability that a pixel is signal"
   plt.imshow(prediction[tiletoshow,:,:,1],cmap='magma')
   #plt.colorbar(label=r"$\rm S_{x, \ 0.3-2.0 \ keV} \ [cnt]$", orientation="horizontal", fraction=0.04)
   plt.title("Predicted map 1")
   
   plt.savefig(outfile)

if rank == 0:
	print("Set-up time     = ",time2-time1)
	print("Data load time  = ",time3-time2)
	print("Processing time = ",time4-time3)
	print("Total time      = ",time4-time1)
