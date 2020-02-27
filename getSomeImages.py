import os
import numpy as np 
from PIL import Image
import h5py
import random

Dir = "/home/admin/Desktop/aerogel_preprocess"
datafile_name = "FOV100.hdf5"
SaveDir = "/home/admin/Desktop/aerogel_repo/fromHDF"

print("If you don't input a datafile name, we will use the file {}".format(datafile_name))
new_file_name = input("What file name? ")
if new_file_name != "":
	datafile_name = new_file_name
print("Using {}".format(datafile_name))

DF = h5py.File(os.path.join(Dir, datafile_name), "r")
TrainYes = DF["TrainYes"]
TrainNo = DF["TrainNo"]

if not os.path.exists(SaveDir):
	os.mkdir(SaveDir)

num_ims = 7
for_seed = 4673
np.random.seed(for_seed)

numIms1 = TrainYes.shape[0]
numIms2 = TrainNo.shape[0]

yes_inds = np.random.randint(0, high = numIms1, size = num_ims)
no_inds = np.random.randint(0, high = numIms2, size = num_ims)

for j in range(num_ims):
	if not os.path.exists(os.path.join(SaveDir, "yes" + str(j))):
		os.mkdir(os.path.join(SaveDir, "yes" + str(j)))
	i = 0
	while True:
		try:
			im_slice = TrainYes[yes_inds[j], i, :, :, :]
			#print(im_slice.dtype)
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

for j in range(num_ims):
	if not os.path.exists(os.path.join(SaveDir, "no" + str(j))):
		os.mkdir(os.path.join(SaveDir, "no" + str(j)))
	i = 0
	while True:
		try:
			im_slice = TrainNo[no_inds[j], i, :, :, :]
			#print("no: ", im_slice.dtype)
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




