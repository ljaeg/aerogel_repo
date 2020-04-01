#Loose translation of fstack.m to python.
#Aight this is quite shit, I'm moving to just using the matlab engine in python, and directly using the matlab code.

#imports
import numpy as np 
from PIL import Image
import os
from scipy.ndimage.filters import uniform_filter
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import matplotlib.pyplot as plt

#global variables
path = "/Users/loganjaeger/Desktop/aerogel/track ims/fm_-60712_-54519"
nhsize = 5
alpha = .2
thresh = 13



### load in movie.

def load_in_movie(path):
	big_arr = []
	i = 1
	while True:
		from_path = os.path.join(path, str(i) + ".png")
		try:
			img = np.array(Image.open(from_path).convert(mode = "L")) # Just going to go with greyscale images bc there's not much info in the color
			big_arr.append(img)
			i += 1
		except FileNotFoundError:
			break
	return np.array(big_arr)

### ++++++++++++++++ compute fmeasure (Focus Measure). +++++++++++

# They use gray-level variance in the paper. 
def gmeasure(im):
	im = (im - np.min(im)) / (np.max(im) - np.min(im))
	filt = uniform_filter(im, size = nhsize, mode = "nearest")
	var = (im - filt) ** 2
	glv = uniform_filter(var, size = nhsize, mode = "nearest")
	return glv

#This is a measure of focus at a certain point
def fmeasure(big_arr):
	fms = np.zeros(big_arr.shape)
	for i, im in enumerate(big_arr):
		fms[i, :, :] = gmeasure(im)
	fms = np.array(fms)
	return fms

### ++++++++++++ compute S measure (Selective Measure). +++++++++++++++

#In the paper, they use a fast 3 point gaussian interpolation, but I'm going to first try using a simple scipy function, though it might be a little slower.
def gaussian(x, A, mu, s):
	return A * np.exp(-(x-mu)**2/(2*s**2))

def fit(x):
	A0 = np.max(x)
	m0 = np.argmax(x)
	xs =np.arange(x.shape[0])
	if m0 - 3 >= 0 and m0 + 3 <= x.shape[0]:
		x = x[m0 - 3 : m0 + 3]
		xs = xs[m0 - 3 : m0 + 3]
	try:
		popt, _ = curve_fit(gaussian, xs, x, p0 = (A0, m0, 3), bounds = ([A0 - 4, m0 - 2, 1], [A0 + 2, m0 + 2, 7]))
	except RuntimeError:
		popt = A0, m0, 7
	return popt

#Hmmmm, I think I need to make the two above applicable to an MxN array? Confused.
#
# def sMeasure(focus_measure, shape):
# 	errs = np.zeros(shape[0], shape[1])
# 	A, m, s = fit()
# 	for i in range(shape[2]):
# 		errs = errs + abs(fm[..., i] - gaussian(i, ))

#super ineffecient but whatever I've gotta get it down and I'll fix it later.
#Hoooooly shit it's so slow... going to have to speed this up some how
def my_sMeasure(focus_measure):
	shape = focus_measure.shape
	#errs = np.zeros(shape[0], shape[1])
	selectivity = np.zeros((shape[1], shape[2]))
	for i in range(shape[1]):
		for j in range(shape[2]):
			if j == 0:
				print(i)
			A, m, s = fit(focus_measure[:, i, j])
			rms_errs = 0 #approximating rms as sum(abs(dif)), instead of the strict sqrt(sum(dif^2))
			for z in range(shape[0]):
				rms_errs += abs(gaussian(z, A, m, s) - focus_measure[z, i, j])
			selectivity[i, j] = 20 * np.log10(np.max(focus_measure[:, i, j]) / rms_errs)
	return selectivity

def phi(selectivity_score):
	return (1 + np.tanh(alpha * (selectivity_score - thresh))) / (2 * alpha)

def omega(phi, fm):
	return .5 + .5 * np.tanh(phi * (fm - 1))

def do(path):
	big_arr = load_in_movie(path)
	focus_measure = fmeasure(big_arr) #shape of big array
	selectivity_measure = my_sMeasure(focus_measure) #shape of image
	phi_ = phi(selectivity_measure)
	phi_ = medfilt(phi_, kernel_size = 3)
	omegas = np.zeros(focus_measure.shape)
	print("   ")
	print(focus_measure.shape)
	for i in range(focus_measure.shape[0]):
		print(i)
		omegas[i, :, :] = omega(phi_, focus_measure[i, :, :])
	norm_factors = np.sum(omegas, axis = 0)
	stacked_im = np.sum(big_arr * omegas, axis = 0) / norm_factors
	stacked_im = (stacked_im - np.min(stacked_im)) / (np.max(stacked_im) - np.min(stacked_im))
	stacked_im = (stacked_im * 255).astype(np.uint8)
	stacked_im = Image.fromarray(stacked_im)
	stacked_im.show()

do(path)





