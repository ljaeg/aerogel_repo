

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import laplace
from skimage.filters.rank import entropy
from skimage.morphology import disk

amazon_code = "fm_22751_-26061"

i = 1
xs = []
e0s = []
hist_stds = []
while True:
	#path = "/Users/loganjaeger/Desktop/aerogel/track ims/" + amazon_code + "/" + str(i) + ".png"
	path = "/Users/loganjaeger/Desktop/aerogel/blanks/" + amazon_code + "/" + str(i) + ".png"
	try:
		img = plt.imread(path)
		e0 = entropy(img[:, :, 0], disk(3))
		e0s.append(np.mean(e0))
		hist = np.histogram(img[:, :, 0], bins = np.arange(0, 1.1, .05))
		s = np.std(hist[0])
		hist_stds.append(s)
		xs.append(i)
		i += 1
	except FileNotFoundError:
		break


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

print(find_max(e0s))
plt.subplot(121)
plt.plot(xs, e0s)
plt.title("e0")

print(find_min(hist_stds))
plt.subplot(122)
plt.plot(xs, hist_stds)
plt.title("hist stds")
plt.show()

# print("min of maxes: ", find_min(maxes))
# print("max of mins: ", find_max(mins))
# plt.subplot(131)
# plt.plot(xs, maxes)
# plt.title("maxes")
# plt.subplot(132)
# plt.plot(xs, mins)
# plt.title("mins")
# plt.subplot(133)
# plt.plot(xs, stds)
# plt.title("stds")
# plt.show()