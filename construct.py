#This brings a track from from_id to to_id

import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
import random
from skimage.exposure import match_histograms, adjust_gamma
import testSurfaceFinders

#from_id = "fm_21850_13198.I1016_13apr10"
from_id = "fm_-60712_-54519"
to_id = "fm_16007_-49606"
#Dir = "/Users/loganjaeger/Desktop/aerogel/"
Dir = "/home/admin/Desktop/aerogel_repo/"

#Load in the movie from disk and calculate the index of the surface.
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
		except (FileNotFoundError, OSError):
			break
	if ind > 25:
		ind = 25
	arr = np.array(arr)
	##
	if arr.shape == (0,):
		print("IT'S GOING WRONG UP HERE THIS IS IT: ")
		print(path_base)
	##
	return arr, ind

#Assign the correct starting indices for the track and the blank that match up the surfaces of the two.
def assign_ij(difference):
	if difference > 0:
		return difference, 1
	elif difference == 0:
		return 1, 1
	elif difference < 0:
		return 1, 1 - difference

#Get a sub_movie of the shape SHAPE from a random spot in the movie represented by BIG_ARR.
def get_subimages(big_arr, shape):
	bas = big_arr.shape
	if bas[1:] != (384, 512, 4):
		print(bas)
	x = np.random.randint(0, high = bas[1] - shape[0])
	y = np.random.randint(0, high = bas[2] - shape[1])
	return big_arr[:, x:x+shape[0], y:y+shape[1], :]

# Load in the track, blank, and mask and put them all together 
def insert(from_id, to_dir, shape = None):
	track, track_surface = load_and_getDelSq(Dir + "/track ims/TRACK-" + from_id)
	track_surface = id_to_surface[from_id]
	blank, blank_surface = load_and_getDelSq(to_dir)
	##
	if shape:
		blank = get_subimages(blank, shape)
	##
	mask_path = Dir + "/track ims/TRACK-" + from_id + "/mask.tif"
	mask = np.array(Image.open(mask_path))
	track, mask = augment(track, mask)
	dif = track_surface - blank_surface
	i, j = assign_ij(dif)
	return paste_save(track, mask, blank, track_index = i, blank_index = j)

# Per Andrew's suggestion, I'm putting blanks + masks into some negative images.
def insert_blank_mask(mask_id, to_dir, shape = None):
	#Note I'm using "blank" here where track would normally go, and "background" where blank might go
	mask_path = Dir + "/track ims/TRACK-" + mask_id + "/mask.tif"
	mask = np.array(Image.open(mask_path))
	key_blank = random.choice(list(testSurfaceFinders.d.keys()))
	blank_surface = testSurfaceFinders.d[key_blank]
	blank, __ = load_and_getDelSq(Dir + "forTestingSurface/" + key_blank)
	small_blank = get_subimages(blank, mask.shape)
	small_blank, mask = augment(small_blank, mask)
	background, background_sur = load_and_getDelSq(to_dir)
	##
	if shape:
		background = get_subimages(background, shape)
	##
	dif = blank_surface - background_sur
	i, j = assign_ij(dif)
	return paste_save(small_blank, mask, background, track_index = i, blank_index = j)


#Paste the track into the blank movie, and either save or return the resulting movie
def paste_save(track, mask, blank, track_index = 1, blank_index =1, save = False):
	x_shape = blank.shape[1]
	y_shape = blank.shape[2]
	x_pos = random.randint(20, x_shape - mask.shape[1])
	y_pos = random.randint(20, y_shape - mask.shape[2])
	m = Image.fromarray(np.uint8(mask))
	end = []
	i = 1
	while True:
		try:
			b_arr = blank[blank_index, :, :, :] * 255
			t_arr = track[track_index, :, :, :] * 255
			#t_arr = match_histograms(t_arr, b_arr)
			t_arr = adjust_brightness(t_arr, b_arr)
			b_slice = Image.fromarray(b_arr.astype(np.uint8))
			t_slice = Image.fromarray(t_arr.astype(np.uint8))
			b_slice.paste(t_slice, (x_pos, y_pos), mask = m)
			if save:
				b_slice.save("/Users/loganjaeger/Desktop/aerogel/const/seventh/" + str(i) + ".png")
			else:
				end.append(np.array(b_slice))
			blank_index += 1
			track_index += 1
			i += 1
		except IndexError:
			break
	return np.array(end)

#Adjust the brightness of the track so that it matches that of the background, making it look more realistic.
def adjust_brightness(im, background):
	for i in range(3):
		background_mean = int(np.mean(background[:, :, i]))
		im_mean = int(np.mean(im[:, :, i]))
		dif = background_mean - im_mean
		if dif > 0:
			im[:, :, i] = np.where(im[:, :, i] + dif > 255, 255, im[:, :, i])
			im[:, :, i] = np.where(im[:, :, i] + dif > 255, im[:, :, i], im[:, :, i] + dif)
		else:
			im[:, :, i] = np.where(im[:, :, i] + dif < 0, 0, im[:, :, i])
			im[:, :, i] = np.where(im[:, :, i] + dif < 0, im[:, :, i], im[:, :, i] + dif)
	return im

#Augment each slice of the track and the mask in the same way.
def augment(movie, mask):
	#MOVIE is a 3d array that represents each slice of the movie stacked on top of each other
	#first, transform every vertical slice

	# datagen_zy = ImageDataGenerator(zoom_range = .15)
	# z_transform = datagen_zy.get_random_transform(movie.shape[0:2])
	# z_transform["zx"] = 0
	# for i in range(movie.shape[2]):
	# 	movie[:, :, i, :] = datagen_zy.apply_transform(movie[:, :, i, :], z_transform)

	#now transform every horizontal slice
	datagen_xy = ImageDataGenerator(zoom_range = [.75, 1.1], rotation_range = 90, shear_range = 10, horizontal_flip = True, vertical_flip = True)
	xy_transform = datagen_xy.get_random_transform(movie.shape[1:3])
	for i in range(movie.shape[0]):
		movie[i, :, :, :] = datagen_xy.apply_transform(movie[i, :, :, :], xy_transform)
	#transform the mask to match
	mask = datagen_xy.apply_transform(mask, xy_transform)
	return movie, mask


id_to_surface = {} #This maps each track id to its surface index, which I have determined.
id_to_surface["fm_21850_13198.I1016_13apr10"] = 13
id_to_surface["fm_-60712_-54519"] = 20
id_to_surface["fm_-29015_12542"] = 7
id_to_surface["fm_-14165_18122"] = 18


#insert(from_id, to_id)






