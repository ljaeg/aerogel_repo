#This brings a track from from_id to to_id

import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
import random
from skimage.exposure import match_histograms, adjust_gamma

#from_id = "fm_21850_13198.I1016_13apr10"
from_id = "fm_-60712_-54519"
to_id = "fm_16007_-49606"
#Dir = "/Users/loganjaeger/Desktop/aerogel/"
Dir = "/home/admin/Desktop/aerogel_repo/"


#First thing is to match up the surfaces
#Note that we're 1-indexing the images, so watch for index mistakes

def load_and_getDelSq(path_base):
	i = 1
	max_DelSq = 0
	ind = 0
	arr = []
	while True:
		from_path = path_base + "/" + str(i) + ".png"
		try:
			img = plt.imread(from_path)
			arr.append(img)
			ds = np.sum(laplace(img[:, :, 0]) ** 2)
			if ds > max_DelSq:
				max_DelSq = ds
				ind = i
			i += 1
		except FileNotFoundError:
			break
	if ind > 25:
		ind = 25
	return np.array(arr), ind

def assign_ij(difference):
	if difference > 0:
		return difference, 1
	elif difference == 0:
		return 1, 1
	elif difference < 0:
		return 1, 1 - difference

def insert(from_id, to_dir):
	track, track_surface = load_and_getDelSq(Dir + "/track ims/TRACK-" + from_id)
	track_surface = id_to_surface[from_id]
	blank, blank_surface = load_and_getDelSq(to_dir)
	mask_path = Dir + "/track ims/TRACK-" + from_id + "/mask.tif"
	mask = np.array(Image.open(mask_path))
	print(mask.shape)
	print(track.shape)
	track, mask = augment(track, mask)
	dif = track_surface - blank_surface
	i, j = assign_ij(dif)
	return paste_save(track, mask, blank, track_index = i, blank_index = j)

def paste_save(track, mask, blank, track_index = 1, blank_index =1, save = False):
	x_pos = random.randint(50, 450)
	y_pos = random.randint(50, 300)
	m = Image.fromarray(np.uint8(mask))
	end = []
	i = 1
	while True:
		try:
			b_arr = blank[blank_index, :, :, :] * 255
			t_arr = track[track_index, :, :, :] * 255
			t_arr = match_histograms(t_arr, b_arr)
			b_slice = Image.fromarray(b_arr.astype(np.uint8))
			t_slice = Image.fromarray(t_arr.astype(np.uint8))
			b_slice.paste(t_slice, (x_pos, y_pos), mask = m)
			if save:
				b_slice.save("/Users/loganjaeger/Desktop/aerogel/const/seventh/" + str(i) + ".png")
			else:
				end.append(b_slice)
			blank_index += 1
			track_index += 1
			i += 1
		except IndexError:
			break
	return np.array(end)

def augment(movie, mask):
	#MOVIE is a 3d array that represents each slice of the movie stacked on top of each other
	#first, transform every vertical slice

	# datagen_zy = ImageDataGenerator(zoom_range = .15)
	# z_transform = datagen_zy.get_random_transform(movie.shape[0:2])
	# z_transform["zx"] = 0
	# for i in range(movie.shape[2]):
	# 	movie[:, :, i, :] = datagen_zy.apply_transform(movie[:, :, i, :], z_transform)

	#now transform every horizontal slice
	datagen_xy = ImageDataGenerator(zoom_range = .2, rotation_range = 45, shear_range = 10, horizontal_flip = True, vertical_flip = True)
	xy_transform = datagen_xy.get_random_transform(movie.shape[1:3])
	for i in range(movie.shape[0]):
		movie[i, :, :, :] = datagen_xy.apply_transform(movie[i, :, :, :], xy_transform)
	#transform the mask to match
	mask = datagen_xy.apply_transform(mask, xy_transform)
	return movie, mask


id_to_surface = {}
id_to_surface["fm_21850_13198.I1016_13apr10"] = 13
id_to_surface["fm_-60712_-54519"] = 20
id_to_surface["fm_-29015_12542"] = 7
id_to_surface["fm_-14165_18122"] = 18


#insert(from_id, to_id)






