
import numpy as np 
import matplotlib.pyplot as plt 

path = "/Users/loganjaeger/Desktop/aerogel/track ims/TRACK-"
amazon_code = "fm_-14165_18122"
mask_path = path + amazon_code + "/" + "mask.tif"

mask = 1 - plt.imread(mask_path) / 255

def load_in_ims():
	frame = 1
	while True:
		p = path + amazon_code + "/" + str(frame) + ".png"
		try:
			img = plt.imread(p)
			masked_img = img * mask + np.ones(img.shape) * (1 - mask)
			masked_img[:, :, 3] = masked_img[:, :, 3] * (1 - mask[:, :, 3])
			plt.imsave(path + amazon_code + "/" + str(frame) + "copy.png", masked_img, format = "png")
			frame += 1
		except FileNotFoundError:
			break

# img = plt.imread(path + amazon_code + "/" + "32copy.png")
# # img = 1 - img 
# # img[:, :, 3] = 1 - img[:, :, 3]
# plt.imshow(img)
# plt.show()

load_in_ims()