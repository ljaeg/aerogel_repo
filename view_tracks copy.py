from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

path_to_img = "/Users/loganjaeger/Desktop/aerogel/fromHDF/no3"

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
		
	def update(self):
		self.im.set_data(self.X[self.frame, :, :, :])
		ax.set_title(str(self.frame + 1))
		self.im.axes.figure.canvas.draw()

def load_in_ims():
	X = []
	frame = 3
	while True:
		path = os.path.join(path_to_img, str(frame) + ".png")
		try:
			img = plt.imread(path)
			X.append(img)
			frame += 1
		except FileNotFoundError:
			break
	X = np.array(X)
	return X

X = load_in_ims()
print(X.shape)

fig, ax = plt.subplots(1, 1)

movie = ScrollMovie(ax, X)

fig.canvas.mpl_connect('key_press_event', movie.on_key)
plt.show()











