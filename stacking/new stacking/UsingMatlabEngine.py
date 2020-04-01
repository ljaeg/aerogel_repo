import matlab.engine as mle
import matlab
import numpy as np 
from PIL import Image
import os
from time import time

path = "/Users/loganjaeger/Desktop/aerogel/track ims/fm_-60712_-54519/"

def load_in_movie(path):
	big_arr = np.zeros((384, 512, 3, 45))
	i = 1
	while True:
		from_path = os.path.join(path, str(i) + ".png")
		try:
			img = np.array(Image.open(from_path).convert("RGB")) # Just going to go with greyscale images bc there's not much info in the color
			big_arr[:, :, :, i - 1] = img
			#big_arr.append(img)
			i += 1
		except FileNotFoundError:
			break
	return np.array(big_arr)

def renorm(img):
	rn = (img - np.min(img)) / (np.max(img) - np.min(img))
	rn = rn * 255
	rn = rn.astype(np.uint8)
	return rn
# t1 = time()
# eng = mle.start_matlab()
# t2 = time()
# print("started the engine: {} seconds".format(t2 - t1))

# movie = load_in_movie(path)
# movie = ((movie - np.min(movie)) / (np.max(movie) - np.min(movie))) * 255
# movie = movie.astype(np.uint8)
# t3 = time()
# print("loaded in movie: {} seconds".format(t3 - t2))

# matlab_movie = matlab.uint8(movie.tolist())
# #matlab_movie = movie.tolist()
# #matlab_movie = eng.mat2cell(matlab_movie, 384, 512, 3, 45)
# t4 = time()
# print("converted to matlab: {} seconds".format(t4 - t3))

# stacked = eng.fstack(matlab_movie)
# t5 = time()
# print("stacked!!! {} seconds".format(t5 - t4))

# eng.exit()
# print("exited the engine!")
# t6 = time()
# print("total time: {} seconds, {} minutes".format(t6 - t1, (t6 - t1) / 60))

# stacked_arr = np.array(stacked).astype(np.uint8)
# print(stacked_arr.shape)
# im = Image.fromarray(stacked_arr)
# im.show()

t1 = time()
eng = mle.start_matlab()
im = eng.stack_all(path)
im = np.array(im)
eng.quit()
print("done")
t2 = time()
print("total time: ", t2 - t1, " seconds")

im = renorm(im)
img = Image.fromarray(im.astype(np.uint8))
img.show()
