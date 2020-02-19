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
datafile_name = "second_iteration.hdf5"
train_test_val = {"train":1/3, "test":1/3, "val":1/3}
last = 13

def make_hdf():
	ty_codes, tn_codes, tey_codes, ten_codes, vy_codes, vn_codes = split_codes("/home/admin/Desktop/aerogel_preprocess/blanks", train_test_val)
	datafile = h5py.File(os.path.join(save_dir, datafile_name), "w")

	create(datafile, ty_codes, "TrainYes")
	print("Done with TrainYes")
	create(datafile, tn_codes, "TrainNo")
	print("Done with TrainNo")
	create(datafile, tey_codes, "TestYes")
	print("Done with TestYes")
	create(datafile, ten_codes, "TestNo")
	print("Done with TestNo")
	create(datafile, vy_codes, "ValYes")
	print("Done with ValYes")
	create(datafile, vn_codes, "ValNo")
	print("Done with ValNo")
	# trainY = create_big_array_track(ty_codes)
	# datafile.create_dataset("TrainYes", trainY.shape, data = trainY)
	# datafile.flush()
	# trainY = 0
	# print("done with TrainYes")

	# trainN = create_big_array_blank(tn_codes)
	# datafile.create_dataset("TrainNo", trainN.shape, data = trainN)
	# datafile.flush()
	# trainN = 0
	# print("done with TrainNo")

	# testY = create_big_array_track(tey_codes)
	# datafile.create_dataset("TestYes", testY.shape, data = testY)
	# datafile.flush()
	# testY = 0
	# print("done with TestYes")

	# testN = create_big_array_blank(ten_codes)
	# datafile.create_dataset("TestNo", testN.shape, data = testN)
	# datafile.flush()
	# testN = 0
	# print("done with TestNo")

	# valY = create_big_array_track(vy_codes)
	# datafile.create_dataset("ValYes", valY.shape, data = valY)
	# datafile.flush()
	# valY = 0
	# print("done with ValYes")

	# valN = create_big_array_blank(vn_codes)
	# datafile.create_dataset("ValNo", valN.shape, data = valN)
	# datafile.flush()
	# valN = 0
	# print("done with ValNo")

	datafile.close()

def create(datafile, codes, name):
	arr = create_big_array_track(codes)
	datafile.create_dataset(name, arr.shape, data = arr)
	datafile.flush()


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
	print("TrainYes: ", len(trainYes))
	print("TrainNo: ", len(trainNo))
	print("TestYes: ", len(testYes))
	print("TestNo: ", len(testNo))
	print("ValYes: ", len(valYes))
	print("ValNo: ", len(valNo))
	print(length)
	return trainYes, trainNo, testYes, testNo, valYes, valNo


def create_big_array_blank(code_list):
	"""
	something I still need to put into this is to
	make it so that a mask will sometimes be put into these
	maybe 1/3 mask, 2/3 regular blank?
	What I want to do is have the same masks on blank backgrounds.
	"""
	big_array = []
	for path in code_list:
		arr, surf = construct.load_and_getDelSq(path)
		x = random.randint(-1, 3)
		if surf + x + last > arr.shape[0]:
			arr = arr[-last:]
		else:
			arr = arr[surf + x : surf + x + last]
		if arr.shape[0] != last:
			print(arr.shape)
			continue
		big_array.append(arr)
	return (np.array(big_array) * 255).astype("uint8")

def create_big_array_track(code_list):
	big_array = []
	for path in code_list:
		key = random.choice(list(construct.id_to_surface.keys()))
		try:
			arr = construct.insert(key, path)[-last:]
		except OSError:
			continue
		if arr.shape[0] != last:
			print(arr.shape)
			continue
		big_array.append(arr)
	return np.array(big_array)


#split_codes(os.path.join(save_dir, "blanks"))
make_hdf()









