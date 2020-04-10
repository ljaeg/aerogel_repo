import matlab.engine as mle
#import matlab
import numpy as np 
from PIL import Image
import os
from time import time

path = "/../Desktop/aerogel/track ims/fm_-60712_-54519/"

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

directory = "~/Desktop/aerogel_preprocess"
dataset_file = os.path.join(directory, "FOV100.hdf5")
save_file = os.path.join(directory, "stacked_w_matlab.hdf5")

eng = mle.start_matlab()
eng.stack_hdf(dataset_file, save_file, nargout = 0)
eng.quit()
print('done')