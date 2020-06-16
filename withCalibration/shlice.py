import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

image_size = (150, 150) #how big do you want the sliced image to be?
original_size = (512, 384)
yx = (384, 512)
original_size = yx

code_places = {
	'fm_-59163_18844' : (24, 249),
	'fm_-100689_-12884' : (158, 288),
	'fm_-7366_-82272' :  (133, 351)
} # key is the amazon code, while the value is a tuple of (x_coord, y_coord) from flair.

def determine_slice_placement(tpl):
	x_coord = tpl[0]
	y_coord = tpl[1]
	xs = determine_coord(x_coord, 1)
	ys = determine_coord(y_coord, 0)
	return (xs, ys)


def determine_coord(coord, n): #if y, make n=0. If x, make n=1.
	if coord > int(image_size[n] / 2) and coord < (original_size[n] - int(image_size[n] / 2)):
		a1 = coord - (image_size[n] / 2)
		a2 = coord + (image_size[n] / 2)
		atpl = (a1, a2)
	elif coord <= int(image_size[n] / 2):
		atpl = (0, image_size[n])
	elif coord >= (original_size[n] - int(image_size[n] / 2)):
		atpl = (original_size[n] - image_size[n], original_size[n])
	return (int(atpl[0]), int(atpl[1]))

def load_movie(code):
	#load in the movie
	X = []
	frame = 3
	while True:
		path = code + "/" + str(frame) + ".png"
		try:
			img = plt.imread(path)
			X.append(img)
			frame += 1
		except FileNotFoundError:
			break
	X = np.array(X)
	return X

def slice_movie(code):
	where_track = code_places[code]
	xs, ys = determine_slice_placement(where_track)
	print(xs)
	print(ys)
	orig_movie = load_movie(code)
	print(orig_movie.shape)
	sliced = np.zeros((orig_movie.shape[0], image_size[0], image_size[1], orig_movie.shape[3]))
	print(sliced.shape)
	rectangle_movie(code, orig_movie, xs[0], ys[0])
	for i in range(orig_movie.shape[0]):
		sliced[i, :, :, :] = orig_movie[i, ys[0]:ys[1], xs[0]:xs[1], :]
	return sliced

def save_new_movie(code, sliced_movie):
	direc = "SLICED-"+code
	if not os.path.isdir(direc): 
		os.mkdir(direc)
	i = 1
	for i in range(sliced_movie.shape[0]):
		plt.imsave(os.path.join(direc, str(i) + ".jpg"), sliced_movie[i])

def rectangle_movie(code, movie, x, y):
	for i in range(movie.shape[0]):
		fig,ax = plt.subplots(1)

		# Display the image
		ax.imshow(movie[i])

		# Create a Rectangle patch
		rect = patches.Rectangle((x, y), 150, 150, linewidth=1, edgecolor='r', facecolor='none')

		# Add the patch to the Axes
		ax.add_patch(rect)

		plt.savefig(f'rectangular/{i}.png')
		plt.close()

def do(code):
	sm = slice_movie(code)
	save_new_movie(code, sm)

do('fm_-7366_-82272')

