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
datafile_name = "first_iteration.hdf5"
train_test_val = {"train":1/3, "test":1/3, "val":1/3}
last = 13

def make_hdf():
	ty_codes, tn_codes, tey_codes, ten_codes, vy_codes, vn_codes = split_codes("/home/admin/Desktop/aerogel_preprocess/blanks", train_test_val)
	datafile = h5py.File(os.path.join(save_dir, datafile_name), "w")

	trainY = create_big_array_track(ty_codes).astype("f")
	datafile.create_dataset("TrainYes", trainY.shape, data = trainY)
	datafile.flush()
	print("done with TrainYes")

	trainN = create_big_array_blank(tn_codes).astype("f")
	datafile.create_dataset("TrainNo", trainN.shape, data = trainN)
	datafile.flush()
	print("done with TrainNo")

	testY = create_big_array_track(tey_codes).astype("f")
	datafile.create_dataset("TestYes", testY.shape, data = testY)
	datafile.flush()
	print("done with TestYes")

	testN = create_big_array_blank(ten_codes).astype("f")
	datafile.create_dataset("TestNo", testN.shape, data = testN)
	datafile.flush()
	print("done with TestNo")

	valY = create_big_array_track(vy_codes).astype("f")
	datafile.create_dataset("ValYes", valY.shape, data = valY)
	datafile.flush()
	print("done with ValYes")

	valN = create_big_array_blank(vn_codes).astype("f")
	datafile.create_dataset("ValNo", valN.shape, data = valN)
	datafile.flush()
	print("done with ValNo")

	datafile.close()


def split_codes(directory, ttv_split = {"train":1/3, "test":1/3, "val":1/3}):
	names = [t[0] for t in os.walk(directory)][1:]
	trainYes = []
	trainNo = []
	testYes = []
	testNo = []
	valYes = []
	valNo = []
	length = len(names)
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
	print(len(trainYes))
	print(len(trainNo))
	print(len(testYes))
	print(len(testNo))
	print(len(valYes))
	print(len(valNo))
	print(length)
	return trainYes, trainNo, testYes, testNo, valYes, valNo


def create_big_array_blank(code_list):
	big_array = []
	for path in code_list:
		arr, surf = construct.load_and_getDelSq(path)
		x = random.randint(-1, 3)
		if surf + x + last > arr.shape[0]:
			arr = arr[-last]
		else:
			arr = arr[surf + x : surf + x + last]
		big_array.append(arr)
	return np.array(big_array)

def create_big_array_track(code_list):
	big_array = []
	for path in code_list:
		key = random.choice(list(construct.id_to_surface.keys()))
		arr = construct.insert(key, path)[:-last]
		big_array.append(arr)
	return np.array(big_array)


#split_codes(os.path.join(save_dir, "blanks"))
make_hdf()









