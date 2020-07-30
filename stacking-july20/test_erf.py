#For testing the erf arguments
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate
import view_tracks
import os
import math

erf_argument_dict = {
	"fm_-60712_-54519":((2.5, 3, 0), (1, 25, 0)),
	"fm_21850_13198.I1016_13apr10":((2.5, 3, 0), (1.45, 30, 0)),
	"fm_-29015_12542":((2.5, 3, 0), (1.05, 33, 0)),
	"fm_-14165_18122":((2.2, 2.3, 0), (1.1, 26, 0))
}

# The arguments for rotating and slicing the image to get just_track, given as (ccw rotation angle, (y1, y2, x1, x2))
just_track_args = {
	"fm_-60712_-54519":(0, (60, 120, 120, 140)),
	"fm_21850_13198.I1016_13apr10":(35, (215, 311, 123, 147)), 
	"fm_-29015_12542":(48.5, (357, 441, 57, 77)),
	"fm_-14165_18122":(0, (55, 121, 37, 57))
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
			ds = 1 #np.sum(laplace(img[:, :, 0]) ** 2)
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

# A function for smoothing the transition into the image
def smoothing_erf(x, abc = (1, 1, 0)):
	a = abc[0]
	b = abc[1]
	c = abc[2]
	return 1 - 0.5*(1+math.erf(abs((x-c)/a) - b))

#create a matrix in which each element is the distance from the y-axis (the vertical line)
def create_x_matrix(shape):
	a = np.arange(shape[1])
	a -= (shape[1] // 2)
	a[a<0]+=1
	a = np.repeat(np.expand_dims(a, axis=0), shape[0], 0)
	return a

#Same thing as above, but in the horizontal direction
def create_y_matrix(shape):
	s = (shape[1], shape[0])
	x = create_x_matrix(s)
	y = np.transpose(x)
	return y

#make the particular mask associated with a certain track from the erf
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
	rotated_img = rotate(track_img, rotation_angle)
	y1, y2, x1, x2 = slice_points
	return rotated_img[y1:y2, x1:x2]

track_id = "fm_-14165_18122"
track_path = os.path.join("..", "track ims", track_id)

track, __ = load_and_getDelSq(track_path)
just_track = []
for t in track:
	just_track.append(get_just_track(track_id, t))
just_track = np.array(just_track)
mask = make_mask(track_id, (just_track.shape[1], just_track.shape[2]))
print(just_track.shape)
print(mask.shape)

#plt.imshow(mask, alpha = .1)
plt.imshow(just_track[21], alpha = 1)
plt.imshow(mask, alpha = .1)
plt.show()