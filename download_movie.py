#download all the images associated with the movie in a directory that has its name as the amazon code

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

amazon_code = "fm_27707_-59216"
Dir = "/Users/loganjaeger/Desktop/aerogel/forTestingSurface/" + amazon_code

frame = 1
os.mkdir(Dir)
while frame < 100:
	print(frame)
	if frame < 10:
		fnumber = "0" + str(frame)
	else:
		fnumber = str(frame)
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-0{y}.jpg".format(x = amazon_code, y = fnumber)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img)
	except OSError:
		print("got error from URL")
		break
	plt.imsave(Dir + "/" + str(frame) + ".png", img)
	frame += 1