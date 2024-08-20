import numpy as np
import math
import sys
sys.path.insert(0, '/marconi/home/userexternal/cgheller/pyfits/lib/python3.6/site-packages')
import pyfits
from inputparameters import *
from mpi_routines import *

def read_fits(infile_clean, infile_input):

# This function reads fits couples of fits images.
# cleanfile contains the data to perform the segmentation 
# inputfile contains the ground truth data
    cleanfile = infile_clean
    inputfile = infile_input
    print("Reading:")
    print(cleanfile)
    print(inputfile)

    cleanimg = pyfits.getdata(cleanfile,0,memmap=False)
    inputimg = pyfits.getdata(inputfile,0,memmap=False)

    return [np.array(cleanimg,dtype=np.float32), np.array(inputimg,dtype=np.float32)]


def process_data(image):
# pre-process the input data to be more suitable to the analysis

    dmax = np.log10(fluxmax)
    dmin = np.log10(fluxmin)

    # calculate logarithms and clip
    dimage = np.log10(np.clip(image,dth,None))

    # normalize
    dimageaux = (dimage - dmin)/(dmax-dmin)
    dimage=np.clip(dimageaux, 0.0, 1.0)

    xmin = np.amin(dimage)
    xmean = np.mean(dimage)
    xstd = np.std(dimage)
    xmax = np.amax(dimage)
    print("avg = ",xmean," +/- ",xstd," MIN ",xmin," MAX ", xmax)

    #return [np.array(dimage,dtype=np.float32)]
    return [dimage]


def tiling_data(seed):

# Build the training set

    if seed > 0:
        startfile = start_train
        endfile = end_train
        numberoffiles = numberoffiles_train
        cleantile = np.zeros((numberofimages0,tile_resolution,tile_resolution))
        skytile = np.zeros((numberofimages0,tile_resolution,tile_resolution))
        originalskytile = np.zeros((numberofimages0,tile_resolution,tile_resolution))
        tile_resolutionh = tile_resolution0
        tiles2 = tiles
    else:
        startfile = start_eval
        endfile = end_eval
        numberoffiles = endfile - startfile
        tiles2 = 2*tiles
        numberofimages_eval = numberoffiles*tiles2*tiles2
        cleantile = np.zeros((numberofimages_eval,tile_resolution,tile_resolution))
        skytile = np.zeros((numberofimages_eval,tile_resolution,tile_resolution))
        originalskytile = np.zeros((numberofimages_eval,tile_resolution,tile_resolution))
        tile_resolutionh = int(tile_resolution0/2)
    print("Reading ",tiles2,"x",tiles2," tiles images")
    print("Total number of tiles = ",numberofimages0)

    tile = np.zeros((tile_resolution,tile_resolution),dtype=np.float32)
    tile1 = np.zeros((tile_resolution,tile_resolution),dtype=np.float32)

    idy = np.arange(startfile,endfile)
    if seed > 0:
       rdmy = np.random.RandomState(seed)
       rdmy.shuffle(idy)

    dmax = np.log10(fluxmax)
    dmin = np.log10(fluxmin)
    sigma = (np.log10(sigma_th)-dmin)/(dmax-dmin)
    print("THRESHOLD = ", sigma)

    ifile = 0
    tileindex = 0
    print("Reading training dataset")

    for i in range(numberoffiles):

        print("Reading image N. ",ifile)
        ifile = ifile+1
        infile_clean = cleanpath+'/'+fileroot_clean+str(idy[i])+fileend_clean+".fits"
        infile_input = inputpath+'/'+fileroot_input+str(idy[i])+fileend_input+".fits"
        # Load data
        # cleanimge contains the data to perform the segmentation 
        # inputimage contains the ground truth data
        cleanimage, inputimage = read_fits(infile_clean, infile_input)
        # pre-process data

        if sigmanoise > 0:
            dirtyimg = np.empty_like(cleanimage)
            dirtyimg = np.random.normal(loc=0.0, scale=sigmanoise, size=dirtyimg.shape)
            cleanimage = np.clip(dirtyimg+cleanimage,0,None)

        cleanimage = np.squeeze(process_data(cleanimage))
        inputimage = np.squeeze(process_data(inputimage))

        # Create tiles
        for j in range(tiles2):
            # indices for inputimage
            jstart = j*tile_resolutionh-tile_boundary
            jend = jstart+tile_resolution
            if jend > fullimagex:
                break
            jstart = max(jstart,0)
            # indices for tile
            tjstart = 0
            tjend = tile_resolution
            if jstart == 0:
                tjstart = tile_boundary
            if jend == fullimagex:
                tjend = tile_resolution-tile_boundary
            for k in range(tiles2):
                # indices for inputimage
                kstart = k*tile_resolutionh-tile_boundary
                kend = kstart+tile_resolution
                if kend > fullimagex:
                    break
                kstart = max(kstart,0)
                # indices for tile
                tkstart = 0
                tkend = tile_resolution
                if kstart == 0:
                    tkstart = tile_boundary
                if kend == fullimagex:
                    tkend = tile_resolution-tile_boundary

                tile[:,:] = 0.0
                tile1[:,:] = 0.0
                tile[tjstart:tjend, tkstart:tkend] = inputimage[jstart:jend,kstart:kend]
                tile1[tjstart:tjend, tkstart:tkend] = cleanimage[jstart:jend,kstart:kend]

                # create binary mask
                originalskytile[tileindex,:,:] = tile
                if buildsky == 0:
                    tile[tile < sigma] = 0.0
                    tile[tile > 0.0] = 1.0
                    ncount = float(np.count_nonzero(tile > 0.0))
                    xncount = ncount/float(tile_resolution*tile_resolution)
                    if xncount < dirtypixels:
                        tile[:,:] = 0.0
                skytile[tileindex,:,:] = tile
                cleantile[tileindex,:,:] = tile1
                tileindex = tileindex + 1   

    if buildsky == 0:
       skyaux = skytile.astype(int)
       return [cleantile, skyaux, originalskytile, tileindex]
    else:
       return [cleantile, skytile, originalskytile, tileindex]

def save_data(images2save,fileprefix):

    num_of_images = images2save.shape[0]
    print("Number of saved images: ",num_of_images)

    for i in range(num_of_images):
        hdu = pyfits.PrimaryHDU(images2save[i,:,:])
        hdulist = pyfits.HDUList([hdu])

        ilabel = i + start_eval
        outfile = fileprefix+str(ilabel)+'.fits'
        hdulist.writeto(outfile,clobber=True)
        #hdulist = pyfits.open(outfile)
        #hdulist[0].header['CTYPE1'] = 'RA---TAN'
        #hdulist[0].header['CTYPE2'] = 'DEC--TAN'
        #hdulist[0].header['CDELT1'] =          -0.000555556
        #hdulist[0].header['CDELT2'] =           0.000555556
        #hdulist[0].header['HISTORY'] ='  '
        hdulist.close()

    return


