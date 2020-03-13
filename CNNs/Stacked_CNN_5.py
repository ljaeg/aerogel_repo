"""
This is the 5th iteration of a CNN to classify stacked images of aerogel with or without tracks. 
We will be using the Keras Functional API, instead of the Sequential API, so that we can feed in multiple inputs.
Let's just play around at this one!
"""

import numpy as np 
import os
import h5py
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Model, load_model 
from keras.layers import Input 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Dropout, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

Dir = "/home/admin/Desktop/aerogel_preprocess"
h5_file = "stacked_1.hdf5"
datafile_path = os.path.join(Dir, h5_file)

batch_size = 32
class_weights = {0:1, 1:1} #Just in case you want to make the NN biased towards positives or negatives
conv_scale = 32
dense_scale = 128
dropout_rate = .2
spatial_d_rate = .25

#### FIRST, LOAD IN THE IMAGES ####
DF = h5py.File(datafile_path, "r")

#Load them in
TrainYes_Z = np.array(DF["TrainYes_Z"])
TrainYes_X = np.array(DF["TrainYes_X"])
TrainYes_Y = np.array(DF["TrainYes_Y"])

TrainNo_Z = np.array(DF["TrainNo_Z"])
TrainNo_X = np.array(DF["TrainNo_X"])
TrainNo_Y = np.array(DF["TrainNo_Y"])

TestYes_Z = np.array(DF["TestYes_Z"])
TestYes_X = np.array(DF["TestYes_X"])
TestYes_Y = np.array(DF["TestYes_Y"])

TestNo_Z = np.array(DF["TestNo_Z"])
TestNo_X = np.array(DF["TestNo_X"])
TestNo_Y = np.array(DF["TestNo_Y"])

ValYes_Z = np.array(DF["ValYes_Z"])
ValYes_X = np.array(DF["ValYes_X"])
ValYes_Y = np.array(DF["ValYes_Y"])

ValNo_Z = np.array(DF["ValNo_Z"])
ValNo_X = np.array(DF["ValNo_X"])
ValNo_Y = np.array(DF["ValNo_Y"])

#Concatenate them and make a generator

#This generator makes the images match up correspondingly.
generator = ImageDataGenerator()
def multi_img_generator(Z, X, Y, answers, seed = 7):
	genZ = generator.flow(Z, answers, seed = seed)
	genX = generator.flow(X, seed = seed)
	genY = generator.flow(Y, seed = seed)
	while True:
		Zi, ans_i = genZ.next()
		Xi = genX.next()
		Yi = genY.next()
		yield [Zi, Xi, Yi], ans_i

#Do training set
trainZ = np.concatenate((TrainYes_Z, TrainNo_Z), axis = 0)
trainX = np.concatenate((TrainYes_X, TrainNo_X), axis = 0)
trainY = np.concatenate((TrainYes_Y, TrainNo_Y), axis = 0)
trainAnswers = np.ones(len(TrainYes_Z) + len(TrainNo_Z))
trainAnswers[len(TrainYes_Z):] = 0
TrainGenerator = multi_img_generator(trainZ, trainX, trainY, trainAnswers, seed = 13)

#Do validation set
valZ = np.concatenate((ValYes_Z, ValNo_Z), axis = 0)
valX = np.concatenate((ValYes_X, ValNo_X), axis = 0)
valY = np.concatenate((ValYes_Y, ValNo_Y), axis = 0)
valAnswers = np.ones(len(ValYes_Z) + len(ValNo_Z))
valAnswers[len(ValYes_Z):] = 0
ValGenerator = multi_img_generator(valZ, valX, valY, valAnswers, seed = 192)

#do testing set
testZ = np.concatenate((TestYes_Z, TestNo_Z), axis = 0)
testX = np.concatenate((TestYes_X, TestNo_X), axis = 0)
testY = np.concatenate((TestYes_Y, TestNo_Y), axis = 0)
testAnswers = np.ones(len(TestYes_Z) + len(TestNo_Z))
testAnswers[len(TestYes_Z):] = 0
TestGenerator = multi_img_generator(testZ, testX, testY, testAnswers, seed = 21)

#For verbosity, I like to be able to see how it performs on positive samples and negative samples
Pos_TestGen = multi_img_generator(TestYes_Z[:200], TestYes_X[:200], TestYes_Y[:200], np.ones(200), seed = 3)
Neg_TestGen = multi_img_generator(TestNo_Z[:200], TestNo_X[:200], TestNo_Y[:200], np.zeros(200), seed = 2)


#### NOW CREATE THE ACTUAL NETWORK ####

#the input and conv layers for images stacked in the Z-direction.
visible_Z = Input(shape = (100, 100, 3))
convZ_1 = Conv2D(conv_scale, kernel_size = (3, 3))(visible_Z)
spatial_d1 = SpatialDropout2D(spatial_d_rate)(convZ_1)
convZ_2 = Conv2D(conv_scale, kernel_size = (3, 3))(spatial_d1)
spatial_d2 = SpatialDropout2D(spatial_d_rate)(convZ_2)
poolZ_1 = MaxPooling2D(pool_size = (2, 2))(spatial_d2)
convZ_3 = Conv2D(conv_scale, kernel_size = (3, 3))(poolZ_1)
spatial_d3 = SpatialDropout2D(spatial_d_rate)(convZ_3)
poolZ_2 = MaxPooling2D(pool_size = (2, 2))(spatial_d3)
convZ_4 = Conv2D(conv_scale, kernel_size = (3, 3))(poolZ_2)
spatial_d4 = SpatialDropout2D(spatial_d_rate)(convZ_4)
poolZ_3 = MaxPooling2D(pool_size = (2, 2))(spatial_d4)
convZ_5 = Conv2D(conv_scale, kernel_size = (3, 3))(poolZ_3)
poolZ_4 = MaxPooling2D(pool_size = (2, 2))(convZ_5)

#The input and conv layers for images stacked in the X-direction.
visible_X = Input(shape = (100, 13, 3))
convX_1 = Conv2D(conv_scale // 2, kernel_size = (3, 3))(visible_X)
poolX_1 = MaxPooling2D(pool_size = (2, 2))(convX_1)
convX_2 = Conv2D(conv_scale // 2, kernel_size = (3, 3))(poolX_1)
spatialX_1 = SpatialDropout2D(spatial_d_rate)(convX_2)
#poolX_2 = MaxPooling2D(pool_size = (2, 2))(convX_2)
convX_3 = Conv2D(conv_scale, kernel_size = (3, 3))(spatialX_1)
spatialX_2 = SpatialDropout2D(spatial_d_rate)(convX_3)

#The input and conv layers for images stacked in the Y-direction.
visible_Y = Input(shape = (13, 100, 3))
convY_1 = Conv2D(conv_scale // 2, kernel_size = (3, 3))(visible_Y)
poolY_1 = MaxPooling2D(pool_size = (2, 2))(convY_1)
convY_2 = Conv2D(conv_scale // 2, kernel_size = (3, 3))(poolY_1)
spatialY_1 = SpatialDropout2D(spatial_d_rate)(convY_2)
#poolY_2 = MaxPooling2D(pool_size = (2, 2))(convY_2)
convY_3 = Conv2D(conv_scale, kernel_size = (3, 3))(spatialY_1)
spatialY_2 = SpatialDropout2D(spatial_d_rate)(convY_3)

#Flatten and concatenate
flat_Z = GlobalMaxPooling2D()(poolZ_4) #Flatten()(poolZ_4) 
flat_X = GlobalMaxPooling2D()(spatialX_2) #Flatten()(spatialX_2) 
flat_Y = GlobalMaxPooling2D()(spatialY_2) #Flatten()(spatialY_2) 
merge = concatenate([flat_Z, flat_X, flat_Y])

#Interpretation Phase
dense_1 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = .005, l2 = .01))(merge)
dropout_1 = Dropout(dropout_rate)(dense_1)
dense_2 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = .005, l2 = .01))(dropout_1)
dropout_2 = Dropout(dropout_rate)(dense_2)
dense_3 = Dense(dense_scale, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = .005, l2 = .01))(dropout_2)
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
Checkpoint_Loss = ModelCheckpoint('/home/admin/Desktop/aerogel_CNNs/loss_FOV100.h5', verbose=1, save_best_only=True, monitor='val_loss')
Checkpoint_Acc = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/acc_FOV100.h5', verbose=1, save_best_only=True, monitor='val_acc')
model.fit_generator(
	generator = TrainGenerator,
	steps_per_epoch = len(trainAnswers) // batch_size,
	epochs = 500,
	verbose = 2,
	validation_data = ValGenerator,
	validation_steps = len(valAnswers) // batch_size,
	callbacks = [Checkpoint_Acc, Checkpoint_Loss],
	class_weight = class_weights
	)

#See performance on testing set
high_acc = load_model('/home/admin/Desktop/Saved_CNNs/acc_FOV100.h5')
low_loss = load_model('/home/admin/Desktop/aerogel_CNNs/loss_FOV100.h5')

def pred(model_name, model):
	pos_preds = model.predict_generator(Pos_TestGen, steps = 200, verbose = 0)
	pos_preds = np.round(pos_preds)
	pos_acc = np.count_nonzero(pos_preds == 1) / len(pos_preds)
	neg_preds = model.predict_generator(Neg_TestGen, steps = 200, verbose = 0)
	neg_preds = np.round(neg_preds)
	neg_acc = np.count_nonzero(neg_preds == 0) / len(neg_preds)
	print("Performance of the model {} on POSITIVE testing samples is: {}".format(model_name, pos_acc))
	print("Performance of the model {} on NEGATIVE testing samples is: {}".format(model_name, neg_acc))
	print("total acc: {}".format(.5 * (pos_acc + neg_acc)))
	print(" ")

pred("HIGH ACC", high_acc)
pred("LOW LOSS", low_loss)




