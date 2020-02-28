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
max_per = None #If no max_per, make max_per = None
last = 13 #This is the number of slices we keep in the movie
size = (100, 100) #If full images, make size = None, else make it a tuple with your desired shape of the images. For example, size = (100, 100)
min_number_in_trainYes = 600 #Make this 0 if you only want to do one loop thru the names list

if max_per:
	min_number_in_trainYes = 0

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

	datafile.close()

def create_t(datafile, codes, name, size = None):
	arr = create_big_array_track(codes, size = size)
	datafile.create_dataset(name, arr.shape, data = arr)
	datafile.flush()

def create_b(datafile, codes, name, size = None):
	arr = create_big_array_blank(codes, size = size)
	datafile.create_dataset(name, arr.shape, data = arr)
	datafile.flush()

def split_codes(directory, ttv_split = {"train":1/3, "test":1/3, "val":1/3}, max_per = None, shuffle = False):
	names = [t[0] for t in os.walk(directory)][1:]
	if shuffle:
		np.random.shuffle(names)
	trainYes = []
	trainNo = []
	testYes = []
	testNo = []
	valYes = []
	valNo = []
	length = len(names)
	while len(trainYes) < min_number_in_trainYes:
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
	print("TrainYes: ", len(trainYes))
	print("TrainNo: ", len(trainNo))
	print("TestYes: ", len(testYes))
	print("TestNo: ", len(testNo))
	print("ValYes: ", len(valYes))
	print("ValNo: ", len(valNo))
	print(length)
	if max_per:
		print("max per: {}".format(max_per))
		return trainYes[:max_per], trainNo[:max_per], testYes[:max_per], testNo[:max_per], valYes[:max_per], valNo[:max_per]
	else:
		return trainYes, trainNo, testYes, testNo, valYes, valNo


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


#split_codes(os.path.join(save_dir, "blanks"))
make_hdf()









