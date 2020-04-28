import numpy as np
import os
import pandas as pd

directory = "..."

ims = [t[0] for t in os.walk(directory)][1:]
ims = set(ims)

cal1 = pd.read_csv('calibration.csv')
cal2 = pd.read_csv('calibration2.csv')
cal3 = pd.read_csv('calibration3.csv')

codes1 = set(cal1["amazon_key"])
codes2 = set(cal2["amazon_key"])
codes3 = set(cal3["amazon_key"])

def intersection(set1, set2):
	print(len(set1 & set2))
	if len(set1 & set2) < 20:
		print(set1 & set2) 

intersection(codes1, ims)
intersection(codes2, ims)
intersection(codes3, ims)

