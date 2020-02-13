#import construct
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
def make_hdf():
	datafile = h5py.File(os.path.join(save_dir, datafile_name), "w")
	datafile.create_dataset("TrainYes", )


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
	
split_codes(os.path.join(save_dir, "blanks"))