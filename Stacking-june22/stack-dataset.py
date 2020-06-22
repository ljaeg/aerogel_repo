import numpy as np 
import mahotas as mh 
from PIL import Image, ImageFilter 
import os
from scipy.ndimage.filters import laplace
from scipy.ndimage import gaussian_filter
import h5py

def load_in_movie(path):
	#load in the movie from a given path
	stack = []
	for i in range(1, 40):
		if i < 10:
			ns = str(i)
		else:
			ns = str(i)
		img = Image.open(os.path.join(path, ns + '.jpg'))
		img = np.array(img)
		img = img.astype(np.float32)
		img = img / 255
		stack.append(img)
	return np.array(stack)

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
	return stacked

#Helper function bc sobel needs to be done channel-wise. Stacks a single channel at a time, and they're recombined in the stack function
def stack_per_channel(movie):
	stack, h, w = movie.shape
	blurred = [gaussian_filter(img, sigma = .2, mode = 'nearest') for img in movie]
	focus = np.abs(np.array([laplace(t) for t in blurred])) #laplacian of slight gaussian blurred image is proxy for in-focus-ness.
	best = np.argmax(focus, 0)
	image = movie
	image = image.reshape((stack,-1)) # image is now (stack, nr_pixels)
	image = image.transpose() # image is now (nr_pixels, stack)
	r = image[np.arange(len(image)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((h,w)) # reshape to get final result
	return r * 255

#Stack in all 3 directions and save each resulting image
def stack_all_directions(image_dir):
	movie = load_in_movie(image_dir)
	Z = stack(movie, 0)
	Y = stack(movie, 1)
	X = stack(movie, 2)
	return Z, Y, X


def create_hdf(img_path, save_dir):
	f = h5py.File(save_dir)
	Zs = []
	Ys = []
	Xs = []
	for movie_path in [x[0] for x in os.walk(img_path)][0:]:
		Z, Y, X = stack_all_directions(movie_path)
		Zs.append(Z)
		Ys.append(Y)
		Xs.append(X)
	Zs = np.array(Zs)
	Xs = np.array(Xs)
	Ys = np.array(Ys)
	f.create_dataset('Stacked-Zs', Zs.shape, data=Zs)
	f.create_dataset('Stacked-Ys', Ys.shape, data=Ys)
	f.create_dataset('Stacked-Xs', Xs.shape, data=Xs)
	f.flush()

img_path = '/home/admin/Desktop/aerogel_preprocess/sliced'
save_path = '/home/admin/aerogel_preprocess/sliced-stacked'
create_hdf(img_path, save_path)

