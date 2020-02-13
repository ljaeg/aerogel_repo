from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

name ="fm_-14165_18122"
info = [(50, 150), (0, 100)]
x0 = info[1][0]
x1 = info[1][1]
y0 = info[0][0]
y1 = info[0][1]

Dir = "/Users/loganjaeger/Desktop/aerogel/track ims/"

def load_in_ims(code, save_dir):
	frame = 1
	while True:
		path = Dir + code + "/" + str(frame) + ".png"
		try:
			img = plt.imread(path)
			#print(img.shape)
		except FileNotFoundError:
			break
		img = img[y0:y1, x0:x1, :]
		plt.imsave(save_dir + "/" + str(frame) + ".png", img)
		frame += 1

save_dir = Dir + "TRACK-" + name

load_in_ims(name, save_dir)