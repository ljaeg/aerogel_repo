#This file is not to be used in a full-scale pipeline. It is more useful for testing different algorithms for stacking and such, and seeing the results immediately.

import numpy as np 
import mahotas as mh 
from PIL import Image, ImageFilter 
import os

from skimage.filters.rank import entropy
from skimage.morphology import disk

from scipy.ndimage.filters import laplace

#Grab each slice of the movie from the local directory
def load_in_movie(path):
	big_arr = []
	i = 1
	while True:
		from_path = os.path.join(path, str(i) + ".png")
		try:
			img = np.array(Image.open(from_path))
			big_arr.append(img)
			i += 1
		except FileNotFoundError:
			break
	return np.array(big_arr)

#Do a whole stack in a given direction (0-->Z, 1-->Y, 2-->X, in which the given axis is the one that's collapsed)
def stack(movie, direction, save_dir):
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
	stacked = Image.fromarray(stacked)
	stacked.save(save_dir)

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

#Stack in all 3 directions and save each resulting image
def stack_all_directions(image_dir, save_dir_stem):
	movie = load_in_movie(image_dir)
	for i in range(3):
		print(i)
		save_to = os.path.join(save_dir_stem, "sobel_blur_{}.png".format(str(i)))
		stack(movie, i, save_to)



code = "fm_-60712_-54519"
path_to_movie = "/Users/loganjaeger/Desktop/aerogel/track ims/" + code
save_path = "/Users/loganjaeger/Desktop/aerogel/stacking/"

stack_all_directions(path_to_movie, save_path)








