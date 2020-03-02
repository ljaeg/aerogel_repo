#this program is used to view tracks easily, given a certain amazon code
#You can also forego this amazon code and just hardcode a path into the function LOAD_IN_IMS

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

amazon_code = "fm_-15400_-8300"

d = {"fm_-60712_-54519" : [(59, 126), (116, 146)], 
"fm_21850_13198.I1016_13apr10" : [(11, 94), (90, 166)],
"fm_-29015_12542" : [(20,100), (0,70)],
"fm_-14165_18122" : [(50, 150), (0, 100)]
}

class ScrollMovie():
	def __init__(self, ax, X):
		self.ax = ax
		self.X = X
		self.frame = 0
		self.im = ax.imshow(self.X[self.frame, :, :, :])
		ax.set_title(str(self.frame + 1))
		self.x_coords = []
		self.y_coords = []
		self.update()

	def on_key(self, event):
		#print(event.key)
		if event.key == "up":
			if self.frame == 0:
				self.update()
				return
			self.frame = self.frame - 1
		elif event.key == "down":
			if self.frame + 1 == self.X.shape[0]:
				self.update()
				return
			self.frame = self.frame + 1
		self.update()

	def onclick(self, event):
		self.x_coords.append(event.xdata)
		self.y_coords.append(event.ydata)
		print(self.x_coords)
		print(self.y_coords)
		self.update()

		
	def update(self):
		self.im.set_data(self.X[self.frame, :, :, :])
		ax.set_title(str(self.frame + 1))
		self.im.axes.figure.canvas.draw()

def load_in_ims(code):
	X = []
	frame = 3
	while True:
		#path = "/Users/loganjaeger/Desktop/aerogel/track ims/" + code + "/" + str(frame) + ".png"
		#path = "/Users/loganjaeger/Desktop/aerogel/blanks/" + code + "/" + str(frame) + ".png"
		#path = "/Users/loganjaeger/Desktop/aerogel/const/seventh/" + str(frame) + ".png"
		#path = "/Users/loganjaeger/Desktop/aerogel/forTestingSurface/" + code + "/" + str(frame) + ".png"
		path = "/Users/loganjaeger/Desktop/aerogel/fromHDF/yes4/" + str(frame) + ".png"
		#path = "/Users/loganjaeger/Desktop/aerogel/sobel/" + str(frame) + ".png"
		try:
			img = plt.imread(path)
			#print(img.shape)
			X.append(img)
			frame += 1
		except FileNotFoundError:
			break
	X = np.array(X)
	return X

X = load_in_ims(amazon_code)
print(X.shape)

fig, ax = plt.subplots(1, 1)

movie = ScrollMovie(ax, X)

fig.canvas.mpl_connect('key_press_event', movie.on_key)
fig.canvas.mpl_connect('button_press_event', movie.onclick)
plt.show()











