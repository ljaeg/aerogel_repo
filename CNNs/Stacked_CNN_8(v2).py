"""
This is the 8th iteration of a CNN to classify stacked images of aerogel with or without tracks. 
This comes from gen 6.
This is used with the images from /aerogel_preprocess/sliced-stacked

Hmmmmmm why is this one running much slower than CNN_5???? An epoch there is ~2s, while here it's ~6s.
^Caveat: I updated to tf-2.1, and now it runs just as fast. However, every epoch throws an error:
'Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled'
This error may be spurious, but it may reflect memory leak, and it's quite annoying.

To run tensorboard cd into: ~/anaconda3/envs/aerogel_tf2/lib/python3.7/site-packages/tensorboard
and run: python main.py --logdir=/home/admin/Desktop/aerogel_preprocess/TB/Mar16/
have to figure out a full fix to this^

One thing I should think about is making sure that the hdf5 files read and write in the same order, so that the orthogonal projections are actually matched up.
If not then I'll have to rewrite the data to the hdf5 files. But that shouldn't take much time.

I also didn't do anything with finding the surface of these files. Maybe the model could figure it out? Although, I suppose it only matters for the X and Y projections.
"""

import numpy as np 
import os
import h5py
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
	print(f'setting memory growth in {gpu}')
	tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import Input 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Concatenate, Multiply, Average, Maximum, Add
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

Dir = "/home/admin/Desktop/aerogel_preprocess"
TB_dir = os.path.join(Dir, "TB")
h5_file_yes = "sliced-stacked/Yes.hdf5"
h5_file_no = 'sliced-stacked/No.hdf5'
dfp_yes = os.path.join(Dir, h5_file_yes)
dfp_no = os.path.join(Dir, h5_file_no)

batch_size = 32
class_weights = {0:1, 1:1} #Just in case you want to make the NN biased towards positives or negatives
trainTestValSplit = [.33, .33, .33] # [Train, Test, Val]
conv_scale = 32
dense_scale = 128
dropout_rate = .3
spatial_d_rate = .25

def norm(ims):
	return ims / 255

#### FIRST, LOAD IN THE IMAGES ####
DFy = h5py.File(dfp_yes, "r")
DFn = h5py.File(dfp_no, "r")

Zyes = norm(np.array(DFy['Stacked-Zs']))
Xyes = norm(np.array(DFy['Stacked-Xs']))[:, :, -13:, :]
Yyes = norm(np.array(DFy['Stacked-Ys']))[:, -13:, :, :]

Zno = norm(np.array(DFn['Stacked-Zs']))
Xno = norm(np.array(DFn['Stacked-Xs']))[:, :, -13:, :]
Yno = norm(np.array(DFn['Stacked-Ys']))[:, -13:, :, :]

print(f'max: {np.max(Zno)}, min: {np.min(Zno)}')

def ttv_split(yes, no, split = [.33, .33, .33]):
	length = min(len(yes), len(no))
	yes = yes[:length]
	no = no[:length]
	a = int(length * split[0])
	b = int(length * (split[0] + split[1]))
	Ytr, Yte, Yv = np.split(yes, [a, b])
	Ntr, Nte, Nv = np.split(no, [a, b])
	train = np.concatenate((Ytr, Ntr), axis= 0)
	test = np.concatenate((Yte, Nte), axis=0)
	val = np.concatenate((Yv, Nv), axis=0)
	trainAnswers = np.ones(len(Ytr) + len(Ntr))
	trainAnswers[len(Ytr):] = 0
	testAnswers = np.ones(len(Yte) + len(Nte))
	testAnswers[len(Yte):] = 0
	valAnswers = np.ones(len(Yv) + len(Nv))
	valAnswers[len(Yv):] = 0
	return train, test, val, trainAnswers, testAnswers, valAnswers

Ztrain, Ztest, Zval, trainAnswers, testAnswers, valAnswers = ttv_split(Zyes, Zno, split = trainTestValSplit)
Xtrain, Xtest, Xval, trainAnswers1, testAnswers1, valAnswers1 = ttv_split(Xyes, Xno, split = trainTestValSplit)
Ytrain, Ytest, Yval, trainAnswers2, testAnswers2, valAnswers2 = ttv_split(Yyes, Yno, split = trainTestValSplit)
assert np.all(testAnswers1 == testAnswers2) and np.all(trainAnswers1 == trainAnswers2) and np.all(valAnswers1 == valAnswers2)
assert np.all(testAnswers1 == testAnswers) and np.all(trainAnswers1 == trainAnswers) and np.all(valAnswers1 == valAnswers)



#Concatenate them and make a generator

#This generator makes the images match up correspondingly.
generator = ImageDataGenerator()
def multi_img_generator(Z, X, Y, answers, seed = 7, shuffle = True):
	genZ = generator.flow(Z, answers, seed = seed, shuffle = shuffle)
	genX = generator.flow(X, seed = seed, shuffle = shuffle)
	genY = generator.flow(Y, seed = seed, shuffle = shuffle)
	while True:
		Zi, ans_i = genZ.next()
		Xi = genX.next()
		Yi = genY.next()
		yield [Zi, Xi, Yi], ans_i

#Do training set
# trainZ = np.concatenate((TrainYes_Z, TrainNo_Z), axis = 0)
# trainX = np.concatenate((TrainYes_X, TrainNo_X), axis = 0)
# trainY = np.concatenate((TrainYes_Y, TrainNo_Y), axis = 0)
# trainAnswers = np.ones(len(TrainYes_Z) + len(TrainNo_Z))
# trainAnswers[len(TrainYes_Z):] = 0
TrainGenerator = multi_img_generator(Ztrain, Xtrain, Ytrain, trainAnswers, seed = 13)

#Do validation set
# valZ = np.concatenate((ValYes_Z, ValNo_Z), axis = 0)
# valX = np.concatenate((ValYes_X, ValNo_X), axis = 0)
# valY = np.concatenate((ValYes_Y, ValNo_Y), axis = 0)
# valAnswers = np.ones(len(ValYes_Z) + len(ValNo_Z))
# valAnswers[len(ValYes_Z):] = 0
ValGenerator = multi_img_generator(Zval, Xval, Yval, valAnswers, seed = 192)

#do testing set
# testZ = np.concatenate((TestYes_Z, TestNo_Z), axis = 0)
# testX = np.concatenate((TestYes_X, TestNo_X), axis = 0)
# testY = np.concatenate((TestYes_Y, TestNo_Y), axis = 0)
# testAnswers = np.ones(len(TestYes_Z) + len(TestNo_Z))
# testAnswers[len(TestYes_Z):] = 0
TestGenerator = multi_img_generator(Ztest[:100], Xtest[:100], Ytest[:100], testAnswers[:100], seed = 21, shuffle = False)

#For verbosity, I like to be able to see how it performs on positive samples and negative samples
# Pos_TestGen = multi_img_generator(TestYes_Z[:200], TestYes_X[:200], TestYes_Y[:200], np.ones(200), seed = 3)
# Neg_TestGen = multi_img_generator(TestNo_Z[:200], TestNo_X[:200], TestNo_Y[:200], np.zeros(200), seed = 2)


#### NOW CREATE THE ACTUAL NETWORK ####

#the input and conv layers for images stacked in the Z-direction.
visible_Z = Input(shape = (150, 150, 3))
convZ_1 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(visible_Z)
spatial_d1 = SpatialDropout2D(spatial_d_rate)(convZ_1)
convZ_2 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(spatial_d1)
spatial_d2 = SpatialDropout2D(spatial_d_rate)(convZ_2)
poolZ_1 = MaxPooling2D(pool_size = (2, 2))(spatial_d2)
convZ_3 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(poolZ_1)
spatial_d3 = SpatialDropout2D(spatial_d_rate)(convZ_3)
poolZ_2 = MaxPooling2D(pool_size = (2, 2))(spatial_d3)
convZ_4 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(poolZ_2)
spatial_d4 = SpatialDropout2D(spatial_d_rate)(convZ_4)
poolZ_3 = MaxPooling2D(pool_size = (2, 2))(spatial_d4)
convZ_5 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(poolZ_3)
convZ_6 =Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(convZ_5)
poolZ_4 = MaxPooling2D(pool_size = (2, 2))(convZ_6)

#The input and conv layers for images stacked in the X-direction.
visible_X = Input(shape = (150, 13, 3))
convX_1 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(visible_X)
poolX_1 = MaxPooling2D(pool_size = (2, 2))(convX_1)
convX_2 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(poolX_1)
spatialX_1 = SpatialDropout2D(spatial_d_rate)(convX_2)
#poolX_2 = MaxPooling2D(pool_size = (2, 2))(convX_2)
convX_3 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(spatialX_1)
#convX_4 = Conv2D(conv_scale, kernel_size = (3, 3))(convX_3)
spatialX_2 = SpatialDropout2D(spatial_d_rate)(convX_3)

#The input and conv layers for images stacked in the Y-direction.
visible_Y = Input(shape = (13, 150, 3))
convY_1 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(visible_Y)
poolY_1 = MaxPooling2D(pool_size = (2, 2))(convY_1)
convY_2 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(poolY_1)
spatialY_1 = SpatialDropout2D(spatial_d_rate)(convY_2)
#poolY_2 = MaxPooling2D(pool_size = (2, 2))(convY_2)
convY_3 = Conv2D(conv_scale, kernel_size = (3, 3), activation = "relu")(spatialY_1)
#convY_4 = Conv2D(conv_scale, kernel_size = (3, 3))(convY_3)
spatialY_2 = SpatialDropout2D(spatial_d_rate)(convY_3)

#Flatten and concatenate
flat_Z = GlobalMaxPooling2D()(poolZ_4) #Flatten()(poolZ_4) 
flat_X = GlobalMaxPooling2D()(spatialX_2) #Flatten()(spatialX_2) 
flat_Y = GlobalMaxPooling2D()(spatialY_2) #Flatten()(spatialY_2) 
merge = Concatenate()([flat_Z, flat_X, flat_Y])

#Interpretation Phase
dense_1 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = .00001, l2 = .00001))(merge)
dropout_1 = Dropout(dropout_rate)(dense_1)
dense_2 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = .00001, l2 = .00001))(dropout_1)
dropout_2 = Dropout(dropout_rate)(dense_2)
dense_3 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = .00001, l2 = .00001))(dropout_2)
dropout_3 = Dropout(dropout_rate)(dense_3)
# dense_4 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l2(.0001))(dropout_3)
# dropout_4 = Dropout(dropout_rate)(dense_4)
output = Dense(1, activation = "sigmoid")(dropout_3)

#Create the model
model = Model(inputs = [visible_Z, visible_X, visible_Y], outputs = output)

#Summarize the model
model.summary()

#compile the model
model.compile(optimizer=Nadam(lr=0.00015), loss='binary_crossentropy', metrics=['acc'])

#train the model
Checkpoint_Loss = ModelCheckpoint('/home/admin/Desktop/aerogel_CNNs/loss_FOV150x150x30.h5', verbose=1, save_best_only=True, monitor='val_loss')
Checkpoint_Acc = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/acc_FOV150x150x30.h5', verbose=1, save_best_only=True, monitor='val_acc')
from time import time
TB = TensorBoard(log_dir = os.path.join(TB_dir, "Jun25", str(time())))

#In tf-2, fit_generator is deprecated and fit now supports generators
model.fit(
	x = TrainGenerator,
	steps_per_epoch = len(trainAnswers) // batch_size,
	epochs = 75,
	verbose = 2,
	validation_data = ValGenerator,
	validation_steps = len(valAnswers) // batch_size,
	callbacks = [Checkpoint_Acc, Checkpoint_Loss, TB],
	class_weight = class_weights
	)

#See performance on testing set
high_acc = load_model('/home/admin/Desktop/Saved_CNNs/acc_FOV150x150x30.h5')
low_loss = load_model('/home/admin/Desktop/aerogel_CNNs/loss_FOV150x150x30.h5')

def pred(model_name, model):
	preds = model.predict([Ztest, Xtest, Ytest], verbose = 1) #ok, gotta figure out what's going on with predict...
	neg_acc, pos_acc = pos_neg_accs(preds, testAnswers)
	print("Performance of the model {} on POSITIVE testing samples is: {}".format(model_name, pos_acc))
	print("Performance of the model {} on NEGATIVE testing samples is: {}".format(model_name, neg_acc))
	print("total acc: {}".format(.5 * (pos_acc + neg_acc)))
	print(" ")

def newPred(model, gen):
	return 0


def pos_neg_accs(preds, actuals):
	#return specificity, sensitivity
	print("pre round")
	preds = np.round(preds)
	print('post round')
	tn = 0
	tp = 0
	fp = 0
	fn = 0
	for i, p in enumerate(preds):
		print(f" predicting: {(i / len(actuals)) * 100} %", end = '\r', flush = True)
		if p == 0 and actuals[i] == 0:
			tn += 1
		elif p == 0 and actuals[i] == 1:
			fn += 1
		elif p == 1 and actuals[i] == 1:
			tp += 1
		elif p == 1 and actuals[i] == 0:
			fp += 1
	return tn / (fp + tn), tp / (fn + tp)


pred("HIGH ACC", high_acc)
pred("LOW LOSS", low_loss)


