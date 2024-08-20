# save trained model
checkpoint = 1
# paths to input images
cleanpath = '/m100_scratch/userexternal/cgheller/radiounet/data/'
inputpath = '/m100_scratch/userexternal/cgheller/radiounet/data/'
validationpath = '/m100_scratch/userexternal/cgheller/radiounet/images/'
fileroot_clean = "RADIO30MHz"
fileroot_input = "RADIO30MHz"
fileend_clean = "-image-pb"
fileend_input = ""
# Output parameters
images = "images/"

# actual number of files read for the training set in the range below
numberoffiles_train = 100
# training files are selected between startfile and endfile+1
start_train = 1
end_train = 100
end_train = end_train+1
# evaluation files
start_eval = 101
end_eval = 110
end_eval = end_eval+1

# full resolution of the input images
fullimagex = 2000
fullimagey = 2000
# number of tiles in both x and y directions
tile_size = 192
tile_boundary = 0

# threshold in Jansky/pixel for the mask
sigma_th = 1e-8
# fraction of data above threshold to set the maks not 0
dirtypixels = 0.0
# flux max for normalization (Jansky/pixel)
fluxmax = 1e-5
# flux min for normalization (Jansky/pixel))
fluxmin = 1e-10
# small number for the flux (< fluxmin)
dth = 1e-12
# if buildsky != 0 mask is not created
buildsky = 0
# sigma of additional gaussian random noise to add
sigmanoise = 0

maskth = 0
logoutput = 1
plotdata = 1
USE_MPI = 0

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

