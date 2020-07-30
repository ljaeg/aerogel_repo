import numpy as np 
import matplotlib.pyplot as plt 
import os
#from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
import random
import view_tracks
import math
from skimage.exposure import match_histograms
from scipy.ndimage import rotate

track_path = os.path.join("..", "track ims", "fm_-29015_12542")
blank_path = os.path.join("..", "forTestingSurface", "fm_-15400_-8300")

# track = plt.imread(track_path)
# blank = plt.imread(blank_path)

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

track, __ = load_and_getDelSq(track_path)
blank, __ = load_and_getDelSq(blank_path)
rotated = rotate(track, 48.5, axes = (2, 1))

matched = match_histograms(track, blank, multichannel=True)
rotated_matched = match_histograms(rotated, blank, multichannel=True)
matched_rotated = rotate(matched, 48.5, axes = (2, 1))
print(np.max(rotated_matched - matched_rotated))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)
for aa in (ax1, ax2):
	aa.set_axis_off()
ax1.imshow(rotated_matched[19, 357:441, 57:77])
ax1.set_title("rotated then matched")
ax2.imshow(matched_rotated[19, 357:441, 57:77])
ax2.set_title("matched then rotated")
plt.tight_layout()
plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()

# ax1.imshow(track)
# ax1.set_title('Source')
# ax2.imshow(blank)
# ax2.set_title('Reference')
# ax3.imshow(matched)
# ax3.set_title('Matched')

# plt.tight_layout()
# plt.show()


