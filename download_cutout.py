#!usr/bin/python3
import os
from astropy.table import Table
import numpy as np

#table of galaxy cluster with ra,dec,z and r500
table = Table.read('/m100_scratch/userexternal/cstuardi/LOTSSdr2psz2/LOTSSdr2psz2_validation.fits', format='fits')

name = table['Name']
ra = table['RAJ2000']
dec = table['DEJ2000']

for i in range(len(name)):

	cutout_url = 'https://lofar-surveys.org/dr2-low-cutout.fits?pos='+str(ra[i])+','+str(dec[i])+'&size=116'

	image_path = '/m100_scratch/userexternal/cstuardi/LOTSSdr2psz2/'+name[i].replace(' ','')+'_1546.fits'

	os.system("wget -O "+image_path+" '"+cutout_url+"'")

	

