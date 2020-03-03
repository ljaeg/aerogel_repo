from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

name ="fm_-15400_-8300" #the amazon code of the movie
info = [(59, 126), (116, 146)] #The coordinates of the rectangle containing the track to cut out of the larger movie.
x0 = info[1][0]
x1 = info[1][1]
y0 = info[0][0]
y1 = info[0][1]

Dir = "/Users/loganjaeger/Desktop/aerogel/forTestingSurface/"

def load_in_ims(code, save_dir):
	frame = 1
	while True:
		path = Dir + code + "/" + str(frame) + ".png"
		try:
			img = plt.imread(path)
			#print(img.shape)
		except FileNotFoundError:
			break
		img = img[y0:y1, x0:x1, :] #slicing movie so that we only get the part with the track
		plt.imsave(save_dir + "/" + str(frame) + ".png", img)
		frame += 1

save_dir = Dir + "TRACK-" + name

load_in_ims(name, save_dir)