#For testing different methods of finding the surface of an aerogel movie
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import laplace
from skimage.filters.rank import entropy
from skimage.morphology import disk

d = {}
d["fm_-105136_-69834"] = 19
d["fm_12154_-10485"] = 20
d["fm_34454_39576"] = 16
d["fm_12703_6956"] = 20
d["fm_16805_10676"] = 19
d["fm_-10903_21842"] = 15
d["fm_-26664_19672"] = 16
d["fm_-15400_-8300"] = 7
d["fm_-19338_-7990"] = 15
d["fm_-2238_-5510"] = 7
d["fm_27707_-59216"] = 25

def find_min(arr):
	mn = 100
	ind = 1
	i = 1
	for v in arr:
		if v < mn:
			mn = v
			ind = i 
		i = i + 1
		if i > 20:
			break
	return ind

def find_max(arr):
	mx = -100
	ind = 1
	i = 1
	for v in arr:
		if v > mx:
			mx = v
			ind = i
		i += 1
		if i > 20:
			break
	return ind 

def get_error(code, guess):
	miss = d[code] - guess
	return miss ** 2

def predict_surface(code):
	i = 1
	xs = []
	e0s = []
	hist_stds = []
	stds = []
	delSq_sums = []
	delSq_sq_sums = []
	while True:
		path = "/Users/loganjaeger/Desktop/aerogel/forTestingSurface/" + code + "/" + str(i) + ".png"
		try:
			img = plt.imread(path)
			e0 = entropy(img[:, :, 0], disk(3))
			e0s.append(np.mean(e0))

			hist = np.histogram(img[:, :, 0], bins = np.arange(0, 1.1, .05))
			hs = np.std(hist[0])
			hist_stds.append(hs)

			std = np.std(img[:, :, 0])
			stds.append(std)

			delSq = laplace(img[:, :, 0])
			delSq_sums.append(np.sum(delSq))
			delSq_sq_sums.append(np.sum(delSq ** 2))

			xs.append(i)
			i += 1
		except FileNotFoundError:
			break
	m = {}
	m["entropy max"] = get_error(code, find_max(e0s))
	m["entropy min"] = get_error(code, find_min(e0s))
	m["hist std max"] = get_error(code, find_max(hist_stds))
	m["hist std min"] = get_error(code, find_min(hist_stds))
	m["std max"] = get_error(code, find_max(stds))
	m["std min"] = get_error(code, find_min(stds))
	m["delSq sum max"] = get_error(code, find_max(delSq_sums))
	m["delSq sum min"] = get_error(code, find_min(delSq_sums))
	m["delSq_sq_sums max"] = get_error(code, find_max(delSq_sq_sums))
	m["delSq_sq_sums min"] = get_error(code, find_min(delSq_sq_sums))
	print(" ")
	print("ds_ss_max guess for code {} is {}".format(code, find_max(delSq_sq_sums)))
	print("actual surface is at {}".format(d[code]))
	print(" ")
	return m

def do():
	e_max = 0
	e_min = 0
	hs_max = 0
	hs_min = 0
	s_max = 0
	s_min = 0
	ds_s_max = 0
	ds_s_min = 0
	ds_ss_max = 0
	ds_ss_min = 0

	codes_done = 1
	for code in d:
		ms = predict_surface(code)
		e_max += ms["entropy max"]
		e_min += ms["entropy min"]
		hs_max += ms["hist std max"]
		hs_min += ms["hist std min"]
		s_max += ms["std max"]
		s_min += ms["std min"]
		ds_s_max += ms["delSq sum max"]
		ds_s_min += ms["delSq sum min"]
		ds_ss_max += ms["delSq_sq_sums max"]
		ds_ss_min += ms["delSq_sq_sums min"]
		print("done with code: {}".format(codes_done))
		codes_done += 1

	print(" ")
	print("mse for e_max: {}".format(e_max))
	print("mse for e_min: {}".format(e_min))
	print("mse for hs_max: {}".format(hs_max))
	print("mse for hs_min: {}".format(hs_min))
	print("mse for s_max: {}".format(s_max))
	print("mse for s_min: {}".format(s_min))
	print("mse for ds_s_max: {}".format(ds_s_max))
	print("mse for ds_s_min: {}".format(ds_s_min))
	print("mse for ds_ss_max: {}".format(ds_ss_max))
	print("mse for ds_ss_min: {}".format(ds_ss_min))

do()




