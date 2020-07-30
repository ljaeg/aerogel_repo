import numpy as np
import matplotlib.pyplot as plt
import os
# from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
from scipy.ndimage import rotate
import random
import view_tracks
import math
from skimage.exposure import match_histograms
from time import time
from scipy.ndimage import affine_transform

track_id = "fm_-60712_-54519"


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

track, __ = load_and_getDelSq(os.path.join("..", "track ims", track_id))
track = track[:, 60:120, 120:140, :]
new_t = do_transform(track, get_random_transform(track.shape))
new_t = new_t[21]
plt.imshow(new_t)
plt.show()


#code new shape from old shape, introduce reflection, then implement into larger thing. Also do the same to the mask. Then apply that to the full stack pipeline.
#also see what's going on with the paper? Is Ogliore still doing it or what?