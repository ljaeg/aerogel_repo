"""
This is a more efficient algorithm to stack an entire dataset than applying the functions in better_stack.py in a for loop
Whereas better_stack.py is helpful bc you can easily test out different algorithms and see results immediately, this is
Used for full-scale deployment
"""
import numpy as np 
import mahotas as mh 
from PIL import Image, ImageFilter 
import os
import h5py

directory = "/home/admin/Desktop/aerogel_preprocess"
dataset_file = os.path.join(directory, "FOV100.hdf5")
save_file = os.path.join(directory, "stacked_2.hdf5")

#Do a whole stack in a given direction (0-->Z, 1-->Y, 2-->X, in which the given axis is the one that's collapsed)
def stack(movie, direction):
	t_shape = (0, 1, 2, 3)
	if direction == 1:
		t_shape = (1, 0, 2, 3)
	elif direction == 2:
		t_shape = (2, 1, 0, 3)
	new_movie = np.transpose(movie, t_shape)[:, :, :, 0:3]
	stacked = np.zeros(new_movie.shape[1:4])
	for i in range(3):
		stacked[..., i] = stack_per_channel(new_movie[..., i])
	stacked = stacked.astype("uint8")
	return stacked

#Helper function bc sobel needs to be done channel-wise. Stacks a single channel at a time, and they're recombined in the stack function
def stack_per_channel(movie):
	stack, h, w = movie.shape
	focus = np.array([mh.sobel(t, just_filter=True) for t in movie]) #Using sobel to determine "infocusness of each pixel". This measure is somewhat arbitrary, and I tried it using laplacian filter and entropy as well
	best = np.argmax(focus, 0)
	image = movie
	image = image.reshape((stack,-1)) # image is now (stack, nr_pixels)
	image = image.transpose() # image is now (nr_pixels, stack)
	r = image[np.arange(len(image)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((h,w)) # reshape to get final result
	return r * 255

#ARR is a N x D x Z x Y x 4 array, in which N is the number of movies, D is the depth that I kept from the movie, ZxY is the FOV size, and 4 is the # of channels
#This function stacks each movie in the array to returns 3 N x Z x Y x 3 array.
def stack_dataset(arr):
	Z = None
	Y = None
	X = None
	for direc in range(3):
		stacked_arr = []
		for mov in arr:
			stacked_arr.append(stack(mov, direc))
		stacked_arr = np.array(stacked_arr)
		if direc == 0:
			Z = stacked_arr
		elif direc == 1:
			Y = stacked_arr
		elif direc == 2:
			X = stacked_arr
	return Z, Y, X 

def do(from_path, to_path):
	DF = h5py.File(from_path, "r")
	New_DF = h5py.File(to_path, "w")
	for name in ["TrainYes", "TrainNo", "TestYes", "TestNo", "ValYes", "ValNo"]:
		Z, Y, X = stack_dataset(DF[name])
		New_DF.create_dataset(name + "_Z", Z.shape, data = Z)
		New_DF.create_dataset(name + "_Y", Y.shape, data = Y)
		New_DF.create_dataset(name + "_X", X.shape, data = X)
		New_DF.flush()
		print("done with {}".format(name))

do(dataset_file, save_file)

