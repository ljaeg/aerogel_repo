import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import time

image_size = (150, 150) #how big do you want the sliced image to be?
original_size = (384, 512)

csv = pd.read_csv('containsCoords.csv')
print("length of csv: " + str(len(csv)))

def get_coords(code):
	sub = csv[csv.amazon_key == code]
	try:
		xc = sub.x_coord.item()
		yc = sub.y_coord.item()
	except ValueError:
		print(len(sub))
		print(code)
		return 1 / 0
	return xc, yc

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

def get_str_from_number(n):
	if n < 10:
		return '-00' + str(n)
	else:
		return '-0' + str(n)

def load_movie(code):
	#load in the movie
	X = []
	frame = 1
	while True:
		path = os.path.join("..", "..", "aerogel_preprocess", "blankAmazon", code, code + get_str_from_number(frame) + ".jpg")
		try:
			img = plt.imread(path)
			X.append(img)
			frame += 1
		except FileNotFoundError:
			break
	if len(X) == 0:
		print(code)
	X = np.array(X)
	return norm(X)

def norm(movie):
	m = (movie - np.min(movie)) / (np.max(movie) - np.min(movie))
	return m

def random_slice_placement():
	y = np.random.randint(0, original_size[0])
	x = np.random.randint(0, original_size[1])
	xs = determine_coord(x, 1)
	ys = determine_coord(y, 0)
	return (xs, ys)

def slice_movie(code, with_coords = True):
	if with_coords:
		where_track = get_coords(code)
		xs, ys = determine_slice_placement(where_track)
	else:
		xs, ys = random_slice_placement()
	# print(xs)
	# print(ys)
	orig_movie = load_movie(code)
	#print(orig_movie.shape)
	sliced = np.zeros((orig_movie.shape[0], image_size[0], image_size[1], orig_movie.shape[3]))
	#print(sliced.shape)
	#rectangle_movie(code, orig_movie, xs[0], ys[0])
	for i in range(orig_movie.shape[0]):
		sliced[i, :, :, :] = orig_movie[i, ys[0]:ys[1], xs[0]:xs[1], :]
	return sliced

def save_new_movie(code, sliced_movie):
	direc = os.path.join("..", "..", 'aerogel_preprocess', 'sliced-blanks', code)
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

def do_single(code, with_coords=True):
	sm = slice_movie(code, with_coords = with_coords)
	save_new_movie(code, sm)

def time_left(current, total, time_elapsed):
	pace = current / time_elapsed
	n_left = total - current
	return n_left / pace

def do_all():
	directory = os.path.join("..", "..", "aerogel_preprocess", "blankAmazon")
	allcodes = [x[0] for x in os.walk(directory)][1:]
	number_to_do = len(allcodes)
	current_number = 0
	t0 = time.time()
	for c in allcodes:
		code = c[37:]
		do_single(code, with_coords = False)
		current_number += 1
		t1 = time.time()
		t_left = time_left(current_number, number_to_do, (t1 - t0) / 60)
		print(f' done: {current_number}/{number_to_do}. Time left: {t_left}', flush=True, end = '\r')


do_all()

