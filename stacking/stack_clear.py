import numpy as np 
import mahotas as mh 
from PIL import Image
import os

#code = "fm_-14165_18122"
code = "fm_-60712_-54519"
path_to_movie = "/Users/loganjaeger/Desktop/aerogel/track ims/" + code
save_path = "/Users/loganjaeger/Desktop/aerogel/stacking/"

#Load in the movie from its directory
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

#Using mh.sobel as a measure of "infocusness" (mh.sobel finds edges)
def get_focus(movie, direction):
	x = []
	s = movie.shape
	if direction == 0:
		for i in range(s[0]):
			x.append(mh.sobel(movie[i, :, :], just_filter = True))
		return np.array(x)
	elif direction == 1:
		for i in range(s[1]):
			x.append(mh.sobel(movie[:, i, :], just_filter = True))
		return np.array(x)
	elif direction == 2:
		for i in range(s[2]):
			x.append(mh.sobel(movie[:, :, i], just_filter = True))
		return np.array(x)

#stack a single channel in the z direction
def stack_channel_z(channel, save = False):
	movie = load_in_movie(path_to_movie)[:, :, :, channel]
	d, h, w = movie.shape
	focus = get_focus(movie, 0)
	best = np.argmax(focus, 0)
	movie = movie.reshape((d,-1)) # movie is now (d, nr_pixels)
	movie = movie.transpose() # movie is now (nr_pixels, d)
	r = movie[np.arange(len(movie)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((h,w)) # reshape to get final result
	if save:
		r = Image.fromarray(r)
		r.save(os.path.join(save_path, "stacked_img.png"))
	else:
		return r

#Stack the whole image in the Z-direction (works great)
def stack_z():
	arr = np.zeros((384, 512, 3))
	for i in range(3):
		x = stack_channel_z(i)
		arr[..., i] = x
	arr = arr * 255
	arr = Image.fromarray(arr.astype("uint8"))
	arr.save(os.path.join(save_path, "stacked_img_z.png"))

#Stack a single channel in the X direction
def stack_channel_x(channel, save = False):
	movie = load_in_movie(path_to_movie)[:, :, :, channel]
	d, h, w = movie.shape
	focus = get_focus(movie, 0)
	best = np.argmax(focus, 1)
	movie = movie.reshape((h, -1))
	movie = movie.transpose()
	r = movie[np.arange(len(movie)), best.ravel()]
	r = r.reshape((d, w))
	return r

#Stack the whole image in X-direction (DNW)
def stack_x():
	arr = np.zeros((45, 512, 3))
	for i in range(3):
		x = stack_channel_x(i)
		arr[..., i] = x
	arr = arr * 255
	arr = Image.fromarray(arr.astype("uint8"))
	arr.save(os.path.join(save_path, "stacked_img_x.png"))

stack_z()






