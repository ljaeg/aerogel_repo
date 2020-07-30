#This is for testing the way that we dub the tracks into the blank aerogel
#Taking ideas from Andrew's MATLAB code.
#We're just going to test out on 150x150 images

import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
import random
import view_tracks
import math

path_to_blank = "../forTestingSurface/fm_-15400_-8300" #path to blank movie
path_to_track = "../track ims/TRACK-fm_-60712_-54519" #path to the track

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
	return arr, ind

#turn an img with [:, :, :, 4] dimensions into a grayscale image
def to_grayscale(rgb):
	#g = 0.299*img[:, :, :, 0] + 0.587*img[:, :, :, 1] + 0.114*img[:, :, :, 2]
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


#Assign the correct starting indices for the track and the blank that match up the surfaces of the two.
#Difference is computed by i_img - j_img.
def assign_ij(difference):
	if difference > 0:
		return difference, 1
	elif difference == 0:
		return 1, 1
	elif difference < 0:
		return 1, 1 - difference

#this assumes scale of 0 to 1
def add_constant(img, constant):
	img_copy = np.copy(img)
	img_copy += constant
	img_copy[img_copy > 1] = 1
	img_copy[img_copy < 0] = 0
	return img_copy

#I only want the aerogel surrounding the track, not the actual track, when calculating brightness
def calc_brightness_for_track(track):
	top = track[:, :3, :]
	bottom = track[:, -3:, :]
	left = track[:, :, :3]
	right = track[:, :, -3:]
	#there is some overlap, but that's ok for now
	top_mean = np.mean(top)
	bottom_mean = np.mean(bottom)
	left_mean = np.mean(left)
	right_mean = np.mean(right)
	return (top_mean + bottom_mean + left_mean + right_mean) / 4

#make the total brightness of the images the same
def adjust_brightness(track, background):
	brightness1 = calc_brightness_for_track(track)
	brightness2 = np.mean(background)
	mean_brightness = np.mean([brightness1, brightness2])
	dif1 = mean_brightness - brightness1
	dif2 = mean_brightness - brightness2
	new_track = add_constant(track, dif1)
	new_background = add_constant(background, dif2)
	return new_track, new_background

#make it so the size of the track img has even components
def evenize(img):
	arr = np.copy(img)
	shape = img.shape
	if shape[1] % 2:
		arr = img[:, :shape[1] - 1, :]
	if shape[2] % 2:
		arr = img[:, :, :shape[2] - 1]
	return arr

def get_mask(path_to_track):
	mask = plt.imread(os.path.join(path_to_track, "mask.tif"))
	mask = mask.astype(np.float32)
	mask = mask / 255.0
	return mask

#create a matrix in which each element is the distance from the y-axis (the vertical line)
def create_x_matrix(shape):
	a = np.arange(shape[1])
	a -= (shape[1] // 2)
	a[a<0]+=1
	a = np.repeat(np.expand_dims(a, axis=0), shape[0], 0)
	#I think this is it.
	return a

def create_y_matrix(shape):
	s = (shape[1], shape[0])
	x = create_x_matrix(s)
	y = np.transpose(x)
	return y

# A function for smoothing the transition into the image
def smoothing_erf(x, a = 3, b = 1.8, c = 0):
	return 1 - 0.5*(1+math.erf(abs((x-c)/a) - b))

#paste
def paste(track, blank, mask = None, location = [75, 75]):
	if mask is None:
		mask = np.ones(track.shape)
	else:
		mask = np.repeat(np.expand_dims(mask, 0), blank.shape[0], 0)
		print("mask shape: ", np.shape(mask))
	track = evenize(track)
	mask = evenize(mask)
	shape = track.shape
	s1, s2 = shape[1] // 2, shape[2] // 2
	blank[:, 75-s1:75+s1, 75-s2:75+s2] = track[:] * (1-mask) + blank[:, 75-s1:75+s1, 75-s2:75+s2] * (mask)
	return blank

blank, blank_index = load_and_getDelSq(path_to_blank)
#view_tracks.do_scroll(blank[:, :150, :150, :3])
blank = to_grayscale(blank)
#view_tracks.do_scroll(blank[:, :150, :150, 2])
track, track_index = load_and_getDelSq(path_to_track)
track = to_grayscale(track)
#mask = get_mask(path_to_track)
# print("a: ", mask.shape)
blank = blank[:, 0:150, 0:150]
"""


new_track, new_blank = adjust_brightness(track, blank)
new_track, new_blank = adjust_brightness(new_track, new_blank)

# #view_tracks.do_scroll(new_blank)

# b = paste(new_track, new_blank, mask = mask)
# view_tracks.do_scroll(b)

se_x = lambda x:smoothing_erf(x, a = 2.5, b=3, c = 0)
se_y = lambda y:smoothing_erf(y, a =1, b=25, c=0)

v_se_y = np.vectorize(se_y)
v_se_x = np.vectorize(se_x)

shape = (track.shape[1], track.shape[2])

x_matrix = create_x_matrix(shape)
y_matrix = create_y_matrix(shape)

x_matrix = v_se_x(x_matrix)
y_matrix = v_se_y(y_matrix)

mask = x_matrix*y_matrix
mask=1-mask
# mask = np.expand_dims(mask, 2)
# mask = np.repeat(mask, 4, 2)
print("b: ", mask.shape)

i, j = assign_ij(20-blank_index)
new_track = new_track[i:-7]
new_blank = new_blank[i:]

b = paste(new_track, new_blank, mask = mask)
#view_tracks.do_scroll(b)
view_tracks.do_scroll(blank)

# print('min: ', np.min(x_matrix))
# plt.imshow(mask[:, :, 0])
# plt.show()



t = np.copy(track)
b = np.copy(blank)

print("0: ", np.mean(t) - np.mean(b))
for i in range(10):
	t, b = adjust_brightness(t, b)
	print(f"difference at step {i}: {np.mean(t[0:5]) - np.mean(b)}")


"""

b25 = blank[25]

bbb = Image.fromarray((b25 * 255).astype("uint8"))
bbb.show()