"""
For creating a new hdf5 file with the name DATAFILE_NAME, and filling it with the appropriate images.
"""
from datetime import datetime
import construct
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
import random
#from skimage.exposure import match_histograms


save_dir = "/home/admin/Desktop/aerogel_preprocess"
datafile_name = "FOV100.hdf5"
train_test_val = {"train":1/3, "test":1/3, "val":1/3}
max_per = None # The max number of movies in a single dataset. If no max_per, make max_per = None
last = 13 # This is the number of slices we keep in the movie.
size = (100, 100) # If full-size images, make size = None, else make it a tuple with your desired shape of the images. For example, size = (100, 100)
min_number_in_dataset = 600 # If not 0, then we will sample more than once to make the size of the datasets larger

if max_per:
	min_number_in_dataset = 0 #If you specify a max number, then make sure this is 0

#Put everything all together.
def make_hdf():
	ty_codes, tn_codes, tey_codes, ten_codes, vy_codes, vn_codes = split_codes("/home/admin/Desktop/aerogel_preprocess/blanks", ttv_split = train_test_val, max_per = max_per)
	datafile = h5py.File(os.path.join(save_dir, datafile_name), "w")

	create_t(datafile, ty_codes, "TrainYes", size = size)
	print("Done with TrainYes")
	create_b(datafile, tn_codes, "TrainNo", size = size)
	print("Done with TrainNo")
	create_t(datafile, tey_codes, "TestYes", size = size)
	print("Done with TestYes")
	create_b(datafile, ten_codes, "TestNo", size = size)
	print("Done with TestNo")
	create_t(datafile, vy_codes, "ValYes", size = size)
	print("Done with ValYes")
	create_b(datafile, vn_codes, "ValNo", size = size)
	print("Done with ValNo")

	create_attrs(datafile)

	datafile.close()

#Create a dataset with movies with tracks.
def create_t(datafile, codes, name, size = None):
	arr = create_big_array_track(codes, size = size)
	print("total number of movies in {}: {}".format(name, arr.shape[0]))
	datafile.create_dataset(name, arr.shape, data = arr)
	datafile.flush()

#Create a dataset without tracks.
def create_b(datafile, codes, name, size = None):
	arr = create_big_array_blank(codes, size = size)
	print("total number of movies in {}: {}".format(name, arr.shape[0]))
	datafile.create_dataset(name, arr.shape, data = arr)
	datafile.flush()

#Shuffle the codes and divide them throughout the 6 datasets.
def split_codes(directory, ttv_split = {"train":1/3, "test":1/3, "val":1/3}, max_per = None):
	names = [t[0] for t in os.walk(directory)][1:]
	trainYes = []
	trainNo = []
	testYes = []
	testNo = []
	valYes = []
	valNo = []
	while len(trainYes) < min_number_in_dataset:
		np.random.shuffle(names)
		i = 0
		k = length * ttv_split["train"] * .5
		while i < k:
			trainYes.append(names[i])
			i += 1
		k = k * 2
		while i < k:
			trainNo.append(names[i])
			i += 1
		k += length * ttv_split["test"] * .5
		while i < k:
			testYes.append(names[i])
			i += 1
		k += length * ttv_split['test'] * .5
		while i < k:
			testNo.append(names[i])
			i += 1
		k += length * ttv_split['val'] * .5
		while i < k:
			valYes.append(names[i])
			i += 1
		k = length
		while i < k:
			valNo.append(names[i])
			i += 1
	print("codes in TrainYes: ", len(trainYes))
	print("codes in TrainNo: ", len(trainNo))
	print("codes in TestYes: ", len(testYes))
	print("codes in TestNo: ", len(testNo))
	print("codes in ValYes: ", len(valYes))
	print("codes in ValNo: ", len(valNo))
	print("Total number of codes we have to pick from: ", len(names))
	if max_per:
		print("max per: {}".format(max_per))
		return trainYes[:max_per], trainNo[:max_per], testYes[:max_per], testNo[:max_per], valYes[:max_per], valNo[:max_per]
	else:
		return trainYes, trainNo, testYes, testNo, valYes, valNo

#Create an array of size N x LAST x SIZE[0] x SIZE[1] x 3 of blank movies
def create_big_array_blank(code_list, size = None):
	big_array = []
	for path in code_list:
		insert_mask = np.random.choice([0, 1], 1, p = [2/3, 1/3])
		if insert_mask:
			key = random.choice(list(construct.id_to_surface.keys()))
			arr = construct.insert_blank_mask(key, path, shape = size)[-last:]
		else:
			arr, surf = construct.load_and_getDelSq(path)
			##
			if size:
				try:
					arr = construct.get_subimages(arr, size)
				except IndexError:
					print(path)
					arr = construct.get_subimages(arr, size)
			##
			arr = (arr * 255).astype("uint8")
			x = random.randint(-1, 3)
			if surf + x + last > arr.shape[0]:
				arr = arr[-last:]
			else:
				arr = arr[surf + x : surf + x + last]
		if arr.shape[0] != last:
			print(arr.shape)
			continue
		big_array.append(arr)
	big_array = np.array(big_array)
	return big_array

#Create an array of size N x LAST x SIZE[0] x SIZE[1] x 3 of movies with tracks
def create_big_array_track(code_list, size = None):
	big_array = []
	for path in code_list:
		key = random.choice(list(construct.id_to_surface.keys()))
		try:
			arr = construct.insert(key, path, shape = size)[-last:]
		except OSError:
			continue
		if arr.shape[0] != last:
			print(arr.shape)
			continue
		big_array.append(arr)
	return np.array(big_array)

#Create a bunch of miscellaneous attributes for the datafile
def create_attrs(datafile):
	datafile.attrs["datetime"] = datetime.now()
	datafile.attrs["size"] = size
	datafile.attrs["depth"] = last
	datafile.attrs["train/test/val"] = ttv_split











