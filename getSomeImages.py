import os
import numpy as np 
from PIL import Image
import h5py
import random

Dir = "/home/admin/Desktop/aerogel_preprocess"
datafile_name = "first_iteration.hdf5"
SaveDir = "/home/admin/Desktop/aerogel_repo/fromHDF"

DF = h5py.File(os.path.join(Dir, datafile_name), "r")
TrainYes = DF["TrainYes"]
TrainNo = DF["TrainNo"]

if not os.path.exists(SaveDir):
	os.mkdir(SaveDir)

for_seed = 139
np.random.seed(for_seed)

numIms1 = TrainYes.shape[0]
numIms2 = TrainNo.shape[0]

yes_inds = np.random.randint(0, high = numIms1, size = 3)
no_inds = np.random.randint(0, high = numIms2, size = 3)

for j in [0, 1, 2]:
	if not os.path.exists(os.path.join(SaveDir, "yes" + str(j))):
		os.mkdir(os.path.join(SaveDir, "yes" + str(j)))
	i = 0
	while True:
		try:
			im_slice = TrainYes[yes_inds[j], i, :, :, :]
			im = Image.fromarray(im_slice)
			im.save(os.path.join(SaveDir, "yes" + str(j), str(i) + ".png"))
			i += 1
		except ValueError:
			f = open(os.path.join(SaveDir, "yes" + str(j), "info.txt"), "w")
			f.write("j=" + str(j) + "\n")
			f.write("yes index: " + str(yes_inds[j]) + "\n")
			f.write("seed: " + str(for_seed) + "\n")
			f.write("Dir: " + Dir + "\n")
			f.write("datafile_name: " + datafile_name + "\n")
			f.write("SaveDir: " + SaveDir + "\n")
			f.close()
			break

for j in [0, 1, 2]:
	if not os.path.exists(os.path.join(SaveDir, "no" + str(j))):
		os.mkdir(os.path.join(SaveDir, "no" + str(j)))
	i = 0
	while True:
		try:
			im_slice = TrainYes[yes_inds[j], i, :, :, :]
			im = Image.fromarray(im_slice)
			im.save(os.path.join(SaveDir, "no" + str(j), str(i) + ".png"))
			i += 1
		except ValueError:
			f = open(os.path.join(SaveDir, "no" + str(j), "info.txt"), "w")
			f.write("j=" + str(j) + "\n")
			f.write("no index: " + str(no_inds[j]) + "\n")
			f.write("seed: " + str(for_seed) + "\n")
			f.write("Dir: " + Dir + "\n")
			f.write("datafile_name: " + datafile_name + "\n")
			f.write("SaveDir: " + SaveDir + "\n")
			f.close()
			break