#download all the images associated with the movie into a directory that has its name as the amazon code.

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

#amazon_code = "fm_27707_-59216"
#Dir = "/Users/loganjaeger/Desktop/aerogel/forTestingSurface/" + amazon_code
Dir = "/home/admin/Desktop/aerogel_preprocess/blanks/"
txt_path_file = "/home/admin/Desktop/aerogel_repo/aerogel_codes.txt"

#Get a single movie from the amazon server and store it locally. Helper function for make_a_bunch
def make_one(amazon_code):
	frame = 1
	direc = Dir + amazon_code
	if os.path.isdir(direc): #this is here so that you can run make_a_bunch without worrying about downloading duplicate movies locally
		return
	else:
		os.mkdir(direc)
	while frame < 100:
		#print(frame)
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
			#print("got error from URL")
			break
		plt.imsave(direc + "/" + str(frame) + ".png", img)
		frame += 1

#Get a bunch of movies from the amazon server and store them locally.
def make_a_bunch(code_txt_file_path):
	f = open(code_txt_file_path, "r")
	number = 0
	for code in f.read().splitlines():
		make_one(code)
		number += 1
		if not number % 100:
			print(str(number) + "/20,000")


make_a_bunch(txt_path_file)