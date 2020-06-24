import os

path = os.path.join("..", "..", "aerogel_preprocess", "blankAmazon")
subdirs = os.listdir(path)

hasStuff = 0
noStuff = 0
for subdir in subdirs:
	if os.listdir(os.path.join(path, subdir)):
		hasStuff += 1
	else:
		noStuff += 1

print(f'{hasStuff} have things. {noStuff} do not have anything.')