import numpy as np 
import mahotas as mh 
from PIL import Image
import os

#code = "fm_-14165_18122"
code = "fm_-60712_-54519"
path_to_movie = "/Users/loganjaeger/Desktop/aerogel/track ims/" + code
save_path = "/Users/loganjaeger/Desktop/aerogel/stacking/"

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

def stack_one_channel(channel, save = False):
	movie = load_in_movie(path_to_movie)[:, :, :, channel]
	d, h, w = movie.shape
	focus = np.array([mh.sobel(t, just_filter=True) for t in movie])
	print(focus.shape)
	best = np.argmax(focus, 0)
	movie = movie.reshape((d,-1)) # movie is now (d, nr_pixels)
	movie = movie.transpose() # movie is now (nr_pixels, d)
	r = movie[np.arange(len(movie)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((h,w)) # reshape to get final result
	if save:
		r = Image.fromarray(r)
		r.save(os.path.join(save_path, "stacked_img.png"))
	return r

def stack():
	#This stacks along the z-axis (works)
	channels = 3
	arr = np.zeros((384, 512, channels))
	for i in range(channels):
		x = stack_one_channel(i)
		arr[..., i] = x
	arr = arr * 255
	arr = Image.fromarray(arr.astype("uint8"))
	arr.save(os.path.join(save_path, "stacked_img.png"))

def stack_x():
	#This stacks along the x-axis (DNW)
	channels = 3
	arr = np.zeros((512, 45, channels))
	for i in range(channels):
		x = stack_copy(i)
		arr[..., i] = x
	arr = arr * 255
	arr = Image.fromarray(arr.astype("uint8"))
	arr.save(os.path.join(save_path, "stacked_img_x.png"))

def stack_y():
	#This stacks along the y-axis (DNW)
	return 0

def stack_copy(channel, save = False):
	movie = load_in_movie(path_to_movie)[:, :, :, channel]
	d, h, w = movie.shape
	focus = get_focus(movie, 1)
	print(focus.shape)
	best = np.argmax(focus, 0)
	movie = movie.reshape((h,-1)) # movie is now (h, nr_pixels)
	movie = movie.transpose() # movie is now (nr_pixels, h)
	r = movie[np.arange(len(movie)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((w,d)) # reshape to get final result
	if save:
		r = Image.fromarray(r)
		r.save(os.path.join(save_path, "stacked_img_test.png"))
	return r

def stack_copy_2(channel, save = False):
	movie = load_in_movie(path_to_movie)[:, :, :, channel]
	d, h, w = movie.shape
	#focus = np.array([mh.sobel(t, just_filter=True) for t in movie])
	focus = get_focus(movie, 1)
	print(focus.shape)
	best = np.argmax(focus, 0)
	movie = movie.reshape((w,-1)) # movie is now (d, nr_pixels)
	movie = movie.transpose() # movie is now (nr_pixels, d)
	r = movie[np.arange(len(movie)), best.ravel()] # Select the right pixel at each location
	r = r.reshape((h,d)) # reshape to get final result
	if save:
		r = Image.fromarray(r)
		r.save(os.path.join(save_path, "stacked_img_test_2.png"))
	return r

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
#stack()

def stack_test2():
	number = 3
	arr = np.zeros((384, 45, number))
	for i in range(number):
		x = stack_copy_2(i)
		arr[..., i] = x
	arr = arr * 255
	arr = Image.fromarray(arr.astype("uint8"))
	arr.save(os.path.join(save_path, "stacked_img_33.png"))

#stack_copy_2(2, save = True)
#stack_test2()
#stack_copy(0, save = True)

#Squish along Z-axis
"""
stack()
"""

#Squish along X-axis
stack_x()



