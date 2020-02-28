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

#Helper function bc sobel needs to be done channel-wise
def stack_per_channel(movie):
	stack, h, w = movie.shape
	focus = np.array([mh.sobel(t, just_filter=True) for t in movie]) #Using sobel to determine "infocusness of each pixel"
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






###########
#PROBABLY TEMPORARY BELOW HERE
##########


#What if we got the focus in the z direction, as if we were collapsing Z, but then we transposed both
#the movie and the focus array so that X/Y was up, and then collapsed it like normal?
def dif_method_per_channel(movie, direction):
	t_shape = (0, 1, 2)
	if direction == 1:
		t_shape = (1, 0, 2)
	elif direction == 2:
		t_shape = (2, 1, 0)
	#focus = np.array([mh.sobel(t, just_filter=True) for t in movie])
	#focus = np.array([entropy(t, disk(5)) for t in movie])
	focus = np.array([laplace(t) ** 2 for t in movie])
	image = np.transpose(movie, t_shape)
	focus = np.transpose(focus, t_shape) #Let's try it with this in first.
	best = np.argmax(focus, 0)
	stack, h, w = image.shape
	image = image.reshape((stack,-1)) # image is now (stack, nr_pixels)
	image = image.transpose() # image is now (nr_pixels, stack)
	r = image[np.arange(len(image)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((h,w)) # reshape to get final result
	return r * 255

def full_dif_method(movie, direction, save_dir):
	r = dif_method_per_channel(movie[:, :, :, 0], direction)
	g = dif_method_per_channel(movie[:, :, :, 1], direction)
	b = dif_method_per_channel(movie[:, :, :, 2], direction)
	s = r.shape
	total = np.zeros((s[0], s[1], 3))
	total[..., 0] = r
	total[..., 1] = g
	total[..., 2] = b
	total = total.astype("uint8")
	total = Image.fromarray(total)
	total.save(save_dir)

def full_test_in_3d(image_dir, save_dir):
	movie = load_in_movie(image_dir)
	for i in range(3):
		print("testing trial: " + str(i))
		save_to = os.path.join(save_dir, "TEST_THISISATEST_{}.png".format(str(i)))
		full_dif_method(movie, i, save_to)

#full_test_in_3d(path_to_movie, save_path)

###########
#PROBABLY TEMPORARY ABOVE HERE
##########










