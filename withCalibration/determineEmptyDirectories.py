import os

path = os.path.join("..", "..", "aerogel_preprocess", "blankAmazon")
subdirs = os.listdir(path)
print(all([c[:2] == 'fm' for c in subdirs]))