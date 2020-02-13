import construct
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
import random
from skimage.exposure import match_histograms


save_dir = "/home/admin/Desktop/aerogel_preprocess"
datafile_name = "first_iteration.hdf5"
def make_hdf():
	datafile = h5py.File(os.join(save_dir, datafile_name), "w")
	datafile.create_dataset("TrainYes", )


def split_codes(directory, ttv_split = {"train":1/3, "test":1/3, "val":1/3}):
	names = os.walk(directory)
	for n in names:
		print(n)
split_codes(os.join(save_dir, "blanks"))