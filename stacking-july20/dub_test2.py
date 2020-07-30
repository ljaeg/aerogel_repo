import numpy as np 
import matplotlib.pyplot as plt 
import os
#from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
from scipy.ndimage import rotate
import random
import view_tracks
import math
from skimage.exposure import match_histograms
from scipy.ndimage import affine_transform
from time import time

"""
Histogram matching should be done on the entire image. If it is done on the subimage that only contains the track and
a small amount of aerogel, the track makes up too much of the FOV and it will be obscured in the histogram matching process.
If histogram matching is done on the entire image, however, the background colors match up very nicely.
"""

track_id = "fm_-29015_12542"
blank_id = "fm_-15400_-8300"
random.seed(17)

blank_path = os.path.join("..", "forTestingSurface", blank_id)
track_path = os.path.join("..", "track ims", track_id)
print(f"is track a path? {os.path.isdir(track_path)} \nis blank a path? {os.path.isdir(blank_path)}" )

# when creating the mask, these are the arguments to the erf given as ((ax, bx, cx), (ay, by, cy))
erf_argument_dict = {
	"fm_-60712_-54519":((2.5, 3, 0), (1, 25, 0)),
	"fm_21850_13198.I1016_13apr10":((2.5, 3, 0), (1.45, 30, 0)),
	"fm_-29015_12542":((2.5, 3, 0), (1.05, 33, 0)), 
	"fm_-14165_18122":((2.2, 2.3, 0), (1.1, 26, 0))
}

# The arguments for rotating and slicing the image to get just_track, given as (ccw rotation angle, (y1, y2, x1, x2))
just_track_args = {
	"fm_-60712_-54519":(0, (50, 130, 110, 150)),
	"fm_21850_13198.I1016_13apr10":(35, (215, 311, 113, 157)),
	"fm_-29015_12542":(48.5, (357, 441, 47, 87)),
	"fm_-14165_18122":(0, (45, 131, 27, 67))
}


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


# Assign the correct starting indices for the track and the blank that match up the surfaces of the two.
# difference should be i_img_index - j_img_index
def assign_ij(difference):
	if difference > 0:
		return difference - 1, 0
	elif difference == 0:
		return 0, 0
	elif difference < 0:
		return 0,  -difference


# get s1 and s2 for use in pasting the image
def get_s1_s2(track_shape):
	s1, s2 = track_shape[0] // 2, track_shape[1] // 2
	return s1, s2


# A function for smoothing the transition into the image
def smoothing_erf(x, abc = (1, 1, 0)):
	a = abc[0]
	b = abc[1]
	c = abc[2]
	return 1 - 0.5*(1+math.erf(abs((x-c)/a) - b))


# create a matrix in which each element is the distance from the y-axis (the vertical line)
def create_x_matrix(shape):
	a = np.arange(shape[1])
	a -= (shape[1] // 2)
	a[a<0]+=1
	a = np.repeat(np.expand_dims(a, axis=0), shape[0], 0)
	return a



# Same thing as above, but in the horizontal direction
def create_y_matrix(shape):
	s = (shape[1], shape[0])
	x = create_x_matrix(s)
	y = np.transpose(x)
	return y


# make the particular mask associated with a certain track from the erf
def make_mask(track_code, track_shape):
	erf_args = erf_argument_dict[track_code]
	se_x = lambda z: smoothing_erf(z, abc = erf_args[0])
	se_y = lambda z: smoothing_erf(z, abc = erf_args[1])
	v_se_x = np.vectorize(se_x)
	v_se_y = np.vectorize(se_y)
	x_matrix = v_se_x(create_x_matrix(track_shape))
	y_matrix = v_se_y(create_y_matrix(track_shape))
	return x_matrix * y_matrix


# Get just the track from the full size image.
def get_just_track(track_code, track_img):
	rotation_angle, slice_points = just_track_args[track_code]
	rotated_img = rotate(track_img, rotation_angle, axes=(2, 1))
	y1, y2, x1, x2 = slice_points
	return rotated_img[:, y1:y2, x1:x2]


def insert(blank_path, track_path, mask = None):
	blank_movie, blank_index = load_and_getDelSq(blank_path)
	track_movie, track_index = load_and_getDelSq(track_path)
	dif = track_index - blank_index
	track_i, blank_j = assign_ij(dif)
	end = []
	#track_matched = match_histograms(track_movie, blank_movie, multichannel = True)
	while True:
		try:
			b = blank_movie[blank_j]
			t = match_histograms(track_movie[track_i], b, multichannel = True)
			#t = track_matched[track_i]
			just_track = get_just_track(track_path[track_path.find("fm"):], t)
			sub_blank = b[:150, :150]
			mask = make_mask(track_path[track_path.find("fm"):], just_track.shape)
			mask = np.expand_dims(mask, 2)
			mask = np.repeat(mask, 4, 2)
			s1, s2 = get_s1_s2(just_track.shape)
			sub_blank[75-s1:75+s1, 75-s2:75+s2] = just_track * mask + sub_blank[75-s1:75+s1, 75-s2:75+s2] * (1-mask)
			end.append(sub_blank)
			blank_j += 1
			track_i += 1
		except IndexError:
			break
	return np.array(end)


# Reshape mask
def reshape_mask(mask, first_dim):
	mask = np.expand_dims(mask, axis=(0, 3))
	mask = np.repeat(mask, first_dim, axis=0)
	mask = np.repeat(mask, 4, axis=-1)
	return mask


# Create a matrix from shear and scaling for use in an affine transform.
def get_transform_matrix(old_shape, y_scale=1., x_scale=1., shear=0.):
	scale_matrix = np.array([[1, 0, 0, 0],
							 [0, 1/y_scale, 0, 0],
							 [0, 0, 1/x_scale, 0],
							 [0, 0, 0, 1]])
	shear_matrix = np.array([[1, 0, 0, 0],
							 [0, 1, 0, 0],
							 [0, -shear, 1, 0],
							 [0, 0, 0, 1]])
	final = shear_matrix @ scale_matrix
	y1 = int(old_shape[1]*y_scale)
	x1 = int(old_shape[2]*x_scale+y1*abs(shear))
	return final, (old_shape[0], y1, x1, 4)


#make it so the size of the track img has even components
def evenize(img):
	arr = np.copy(img)
	shape = img.shape
	if shape[1] % 2:
		arr = arr[:, :shape[1] - 1, :]
	if shape[2] % 2:
		arr = arr[:, :, :shape[2] - 1, :]
	return arr


# Returns (reflection, transformation_matrix, new_shape, rotation_angle)
def get_random_transform(shape, scale_mean_std=(1, 0.2), rotation_bounds=(-90, 90), shear_lambda=6, reflection_chance=True):
	x_scale=random.gauss(scale_mean_std[0], scale_mean_std[1])
	y_scale=random.gauss(x_scale, 0.1)
	shear = random.expovariate(shear_lambda)
	if reflection_chance:
		reflection = random.choice([0, 1])
	else:
		reflection = 0
	rotation = random.uniform(rotation_bounds[0], rotation_bounds[1])
	matrix, new_shape = get_transform_matrix(shape, y_scale=y_scale, x_scale=x_scale, shear=shear)
	return reflection, matrix, new_shape, rotation


# Actually do the transform. Transform args are the output of get_random_transform.
def do_transform(img, transform_args):
	reflection = transform_args[0]
	matrix = transform_args[1]
	new_shape = transform_args[2]
	rotation_angle = transform_args[3]
	new_img = affine_transform(img, matrix, output_shape=new_shape)
	if reflection:
		new_img = np.flip(new_img, 2)
	new_img = rotate(new_img, rotation_angle, axes = (2, 1))
	clip(new_img)
	return new_img


# Clip an image so it's pixel values are in [0,1]
def clip(img):
	img[img>1] = 1
	img[img<0] = 0


# Do a random transform to an image and a mask, and return the image and mask
def transform_img_and_mask(img, mask):
	assert img.shape == mask.shape, "mask and image are not the same shape"
	transform_args = get_random_transform(img.shape)
	img = do_transform(img, transform_args)
	mask = do_transform(mask, transform_args)
	return evenize(img), evenize(mask)


# Insert but without a for loop
def insert_fast(blank_path, track_path):
	blank_movie, blank_index = load_and_getDelSq(blank_path)
	track_movie, track_index = load_and_getDelSq(track_path)
	dif = track_index - blank_index
	track_i, blank_j = assign_ij(dif)
	track_matched = match_histograms(track_movie, blank_movie, multichannel=True)
	just_track = get_just_track(track_path[track_path.find("fm"):], track_matched)
	blank_movie = blank_movie[:, :150, :150, :]
	mask = make_mask(track_path[track_path.find("fm"):], just_track.shape[1:3])
	first_dimension = min(blank_movie.shape[0] - blank_j, just_track.shape[0] - track_i)
	mask = reshape_mask(mask, first_dimension)
	just_track=just_track[track_i:track_i + first_dimension]
	just_track, mask = transform_img_and_mask(just_track, mask)
	s1, s2 = get_s1_s2(just_track.shape[1:3])
	blank_movie = blank_movie[blank_j:blank_j+first_dimension]
	blank_movie[:, 75-s1:75+s1, 75-s2:75+s2, :] = just_track * mask + blank_movie[:, 75-s1:75+s1, 75-s2:75+s2, :] * (1-mask)
	return blank_movie

t0 = time()
b = insert_fast(blank_path, track_path)
t1 = time()
print("fast: ", t1-t0)

view_tracks.do_scroll(b)

