# save trained model
checkpoint = 1
# paths to input images
cleanpath = '/leonardo/home/userexternal/cstuardi/simulations'
inputpath = '/leonardo/home/userexternal/cstuardi/simulations'
validationpath = '/leonardo/home/userexternal/cstuardi/simulations/'
retrainingpath = '/leonardo/home/userexternal/cstuardi/radioUNET_output/training_DE78/'
trained_network = '/leonardo/home/userexternal/cstuardi/radioUNET_output/original_model/model-5-50-200-0.0001.ckpt'
fileroot_clean = "RADIO30MHz"
fileroot_input = "RADIO30MHz"
fileend_clean = "-image-pb"
fileend_input = ""
# Output parameters
images = "/leonardo/home/userexternal/cstuardi/radioUNET_output/images/"

# actual number of files read for the training set in the range below
numberoffiles_train = 100
# training files are selected between startfile and endfile+1
start_train = 1
end_train = 100
end_train = end_train+1
start_eval = 101
end_eval = 110
end_eval = end_eval+1

# full resolution of the input images
fullimagex = 960
fullimagey = 960
# number of tiles in both x and y directions
tile_size = 192
tile_boundary = 0

# threshold in Jansky for the mask
sigma_th = 1e-8
# fraction of data above threshold to set the maks not 0
dirtypixels = 0.0
# flux max for normalization (Jansky)
fluxmax = 1e-5
fluxmax_lofar = 1e-2 #Jy/beam
# flux min for normalization (Jansky)
fluxmin = 1e-10
fluxmin_lofar = 1e-7
# small number for the flux (< fluxmin)
dth = 1e-12
dth_lofar = 1e-9
# if buildsky != 0 mask is not created
buildsky = 0
# sigma of additional gaussian random noise to add
sigmanoise = 0

maskth = 0
logoutput = 1
plotdata = 1
USE_MPI = 1 #NB no needed for evaluation

# end of parameters setting: do not change below ================================
tiles = int(fullimagex/tile_size)
resolution = fullimagex
tile_resolution = tile_size
#tile_resolution = tile_boundary + tile_resolution + tile_boundary
tile_resolution0 = tile_resolution
tile_resolution = tile_resolution+2*tile_boundary
imagex = tile_resolution
imagey = tile_resolution
#imagesize = imagex*imagey
imagesize = tile_resolution*tile_resolution
numberofimages0 = numberoffiles_train*tiles*tiles

