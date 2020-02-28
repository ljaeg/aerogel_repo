"""
If the following is your movie, with axes labeled

   /################
 Z				 # #
/################  |
#				#  Y
#				#  |
#				#  #
#				# #
#######-X-######

(In our real movies, we have Z~40, X = 512, Y = 384)

Then what I want to create is a movie that has the following properties 
when collapsed along the following dimensions:
	Z: Medium horizontal line
	X: Diagonal line
	Y: Small horizontal line

"""
import numpy as np 
import mahotas as mh 
from PIL import Image
import os


def make():
	A = np.ones((30, 150, 200, 3))
	j = 100
	k = 100
	for i in range(30):
		A[i, j, k, 0:3] = 0
		A[i, j, k + 1, 0:3] = 0
		A[i, j + 1, k, 0:3] = 0
		A[i, j + 1, k + 1, 0:3] = 0
		j -= 1
	for w in [-1, -2, 0, 1, 2, 3]:
		for z in [-1, -2, 0, 1, 2, 3]:
			A[29, j + w, k + z, 0:3] = 0
			A[28, j + w, k + z, 0:3] = 0

	A = (A*255).astype("uint8")
	y = 1
	for z_slc in A:
		im = Image.fromarray(z_slc).convert("RGB")
		im.save("/Users/loganjaeger/Desktop/aerogel/stacking/TestMovie/" + str(y) + ".png")
		y += 1

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
	return r

#Stack the whole image in the Z-direction (works great)
def stack_z():
	arr = np.zeros((150, 200, 3))
	for i in range(1):
		x = stack_channel_z(i)
		arr[..., i] = x
	arr = arr * 255
	arr = Image.fromarray(arr.astype("uint8"))
	arr.save( "stacked_img_z_TEST.png")

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

def stack_z_new():
	movie = load_in_movie(path_to_movie)[:, :, :, 0:3]
	collapsed = np.sum(movie, axis = 0)
	print(collapsed.shape)
	normed = (collapsed - np.min(collapsed)) / (np.max(collapsed) - np.min(collapsed))
	# normed = (normed - np.mean(normed)) / np.std(normed)
	# normed = (normed - np.min(normed)) / (np.max(normed) - np.min(normed))
	normed = normed * 255
	arr = Image.fromarray(normed.astype("uint8"))
	arr.save( "stacked_img_z_TEST_justAdding.png")

def sobel(i):
	#what does a sobel look like?
	x = np.zeros((45, 384, 3))
	for j in range(3):
		slc = load_in_movie(path_to_movie)[i, :, :, j]
		s = mh.sobel(slc, just_filter = True)
		s = ((s - np.min(s))/ (np.max(s) - np.min(s))) * 255
		x[..., j] = 255 - s
	arr = Image.fromarray(x.astype("uint8"))
	arr.save("/Users/loganjaeger/Desktop/aerogel/sobel/{}.png".format(str(i + 1)))

def full_sobel():
	i = 0
	while True:
		try:
			sobel(i)
			i+=1
		except IndexError:
			break

path_to_movie = "/Users/loganjaeger/Desktop/aerogel/stacking/TestMovie/"
code = "fm_-60712_-54519"
path_to_movie = "/Users/loganjaeger/Desktop/aerogel/track ims/" + code
#stack_z()
#stack_z_new()
#make()
#sobel()
full_sobel()






