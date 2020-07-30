import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import laplace
from scipy.ndimage import rotate
import random
import view_tracks
import math
from skimage.exposure import match_histograms
from time import time

