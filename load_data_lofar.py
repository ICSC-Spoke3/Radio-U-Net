import numpy as np
import math
import os
import sys
#sys.path.insert(0, '/marconi/home/userexternal/cgheller/pyfits/lib/python3.6/site-packages')
import astropy.io.fits as pyfits
from inputparameters import *
from mpi_routines import *

def read_fits_lofar(infile_clean):

# This function reads fits couples of fits images.
# cleanfile contains the data to perform the segmentation 
# inputfile contains the ground truth data
    cleanfile = infile_clean
    print("Reading:")
    print(cleanfile)

    cleanimg = pyfits.getdata(cleanfile,0,memmap=False)
    header = pyfits.getheader(cleanfile,0)
    return [np.array(cleanimg,dtype=np.float32),header]


def process_data_lofar(image):
# pre-process the input data to be more suitable to the analysis

    dmax = np.log10(fluxmax_lofar)
    dmin = np.log10(fluxmin_lofar)

    # calculate logarithms
    image[np.isnan(image)]=dth_lofar
    dimage = np.log10(np.clip(image,dth_lofar,None))

    # normalize
    dimageaux = (dimage - dmin)/(dmax-dmin)
    dimage=np.clip(dimageaux, 0.0, 1.0)

    xmin = np.amin(dimage)
    xmean = np.mean(dimage)
    xstd = np.std(dimage)
    xmax = np.amax(dimage)
    print("avg = ",xmean," +/- ",xstd," MIN ",xmin," MAX ", xmax)

    #return [np.array(dimage,dtype=np.float32)]
    return dimage


def tiling_data_lofar(path_data,seed):

	img_list=os.listdir(path_data)
	if 'masks' in img_list: img_list.remove('masks')
	#startfile = start_eval
	#endfile = end_eval
	numberoffiles = len(img_list)
	#seed>0 for training
	if seed > 0:
		tiles2 = tiles
		numberofimages = numberoffiles*tiles2*tiles2
		cleantile = np.zeros((numberofimages,tile_resolution,tile_resolution))
		masktile = np.zeros((numberofimages,tile_resolution,tile_resolution))
		tile_resolutionh = tile_resolution0

		tile = np.zeros((tile_resolution,tile_resolution),dtype=np.float32)
	#seed=0 for validation/using the network (and re-composing images)
	else:
		tiles2 = 2*tiles
		numberofimages = numberoffiles*tiles2*tiles2
		cleantile = np.zeros((numberofimages,tile_resolution,tile_resolution))
		tile_resolutionh = int(tile_resolution0/2)

	print("Reading ",tiles2,"x",tiles2," tiles images")
	print("Total number of tiles = ",numberofimages)

	tile1 = np.zeros((tile_resolution,tile_resolution),dtype=np.float32)

	header_list = []

	ifile = 0
	tileindex = 0
	count=0
	print("Reading dataset")

	for i in range(numberoffiles):

		print("Reading image N. ",ifile)
		ifile = ifile+1
		infile_clean = path_data+img_list[i]
		if seed>0: infile_mask = path_data+'masks/'+img_list[i].replace('576','sub.mask.out')
		# Load data
		# cleanimge contains the data to perform the segmentation
		cleanimage, header = read_fits_lofar(infile_clean)
		#the mask is optimized for diffuse emission (the header is the same)
		if seed>0: maskimage, header = read_fits_lofar(infile_mask)
		header_list.append(header)
		# pre-process data
		cleanimage = np.squeeze(process_data_lofar(cleanimage)) #for LOFAR images squeeze is not necessary
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

				tile1[:,:] = 0.0
				tile1[tjstart:tjend, tkstart:tkend] = cleanimage[jstart:jend,kstart:kend]
				cleantile[tileindex,:,:] = tile1
				if seed>0:
                                	tile[:,:] = 0.0
                                	tile[tjstart:tjend, tkstart:tkend] = maskimage[jstart:jend,kstart:kend]
                                	if np.sum(tile)!=0: count=count+1
                                	masktile[tileindex,:,:] = tile

				tileindex = tileindex + 1
	if seed>0: print('The number of non empty masks is '+str(count))
	if seed>0: return [cleantile,masktile,img_list,header_list]

	else: return [cleantile,img_list,header_list]

def save_data_lofar(images2save,fileprefix,name_list,header_list):

    num_of_images = images2save.shape[0]
    print("Number of saved images: ",num_of_images)

    for i in range(num_of_images):
        hdu = pyfits.PrimaryHDU(images2save[i,:,:],header=header_list[i])
        hdulist = pyfits.HDUList([hdu])
        outfile = fileprefix+name_list[i]
        hdulist.writeto(outfile)
        #hdulist = pyfits.open(outfile)
        #hdulist[0].header['CTYPE1'] = 'RA---TAN'
        #hdulist[0].header['CTYPE2'] = 'DEC--TAN'
        #hdulist[0].header['CDELT1'] =          -0.000555556
        #hdulist[0].header['CDELT2'] =           0.000555556
        #hdulist[0].header['HISTORY'] ='  '
        hdulist.close()

    return
