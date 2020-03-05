"""
This is the third iteration of a CNN to classify stacked images of aerogel with or without tracks. 
We will now be using the Sequential API, instead of the functional API. I think this makes it easier to 
Add and subtract layers.
"""

import numpy as np 
import os
import h5py
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Model, load_model, Sequential
from keras.layers import Input 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Dropout, SpatialDropout2D, concatenate, BatchNormalization, ReLU, GlobalAveragePooling2D
#from keras.layers.merge import concatenate
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
#from keras.activations import ReLU

Dir = "/home/admin/Desktop/aerogel_preprocess"
h5_file = "stacked_1.hdf5"
datafile_path = os.path.join(Dir, h5_file)

batch_size = 32
class_weights = {0:1, 1:1} #Just in case you want to make the NN biased towards positives or negatives
dropout_rate = .15
spatial_d_rate = .2
conv_scale = 64 // 2
dense_scale = 256 // 2

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
Zmodel = Sequential()
Zmodel.add(Conv2D(conv_scale // 2, kernel_size = (3, 3), input_shape = (100, 100, 3)))
#Zmodel.add(BatchNormalization())
Zmodel.add(ReLU())
Zmodel.add(SpatialDropout2D(spatial_d_rate))
Zmodel.add(Conv2D(conv_scale, kernel_size = (3, 3)))
#Zmodel.add(BatchNormalization())
Zmodel.add(ReLU())
Zmodel.add(SpatialDropout2D(spatial_d_rate))
Zmodel.add(MaxPooling2D(pool_size = (2, 2)))
Zmodel.add(Conv2D(conv_scale, kernel_size = (3, 3)))
#Zmodel.add(BatchNormalization())
Zmodel.add(ReLU())
Zmodel.add(SpatialDropout2D(spatial_d_rate))
Zmodel.add(MaxPooling2D(pool_size = (2, 2)))
Zmodel.add(Conv2D(conv_scale * 2, kernel_size = (3, 3)))
#Zmodel.add(BatchNormalization())
Zmodel.add(ReLU())
Zmodel.add(SpatialDropout2D(spatial_d_rate))
Zmodel.add(MaxPooling2D(pool_size = (2, 2)))
Zmodel.add(Conv2D(conv_scale * 2, kernel_size = (3, 3)))
#Zmodel.add(BatchNormalization())
Zmodel.add(ReLU())
Zmodel.add(MaxPooling2D(pool_size = (2, 2)))
Zmodel.add(GlobalAveragePooling2D())

#For the X-direction
Xmodel = Sequential()
Xmodel.add(Conv2D(8, kernel_size = (3, 3), activation = "relu", input_shape = (100, 13, 3)))
Xmodel.add(MaxPooling2D(pool_size = (2, 2)))
Xmodel.add(Conv2D(8, kernel_size = (3, 3), activation = "relu"))
Xmodel.add(SpatialDropout2D(spatial_d_rate))
Xmodel.add(Conv2D(8, kernel_size = (3, 3), activation = "relu"))
Xmodel.add(SpatialDropout2D(spatial_d_rate))
Xmodel.add(GlobalAveragePooling2D())

#For the Y-direction
Ymodel = Sequential()
Ymodel.add(Conv2D(8, kernel_size = (3, 3), activation = "relu", input_shape = (13, 100, 3)))
Ymodel.add(MaxPooling2D(pool_size = (2, 2)))
Ymodel.add(Conv2D(8, kernel_size = (3, 3), activation = "relu"))
Ymodel.add(SpatialDropout2D(spatial_d_rate))
Ymodel.add(Conv2D(8as, kernel_size = (3, 3), activation = "relu"))
Ymodel.add(SpatialDropout2D(spatial_d_rate))
Ymodel.add(GlobalAveragePooling2D())

#Concatenate and make synthesized model with interpretation phase
# model = Sequential()
# model.add(concatenate([Zmodel, Xmodel, Ymodel]))
# model.add(Dense(256, activation = "relu"))
# model.add(Dropout(dropout_rate))
# model.add(Dense(128, activation = "relu"))
# model.add(Dropout(dropout_rate))
# model.add(Dense(128, activation = "relu"))
# model.add(Dropout(dropout_rate))
# model.add(Dense(1, activation = "sigmoid"))

#Use the defined sequential models to encode, then use functional API to combine.
Z_input = Input(shape = (100, 100, 3))
Z_encoded = Zmodel(Z_input)

X_input = Input(shape = (100, 13, 3))
X_encoded = Xmodel(X_input)

Y_input = Input(shape = (13, 100, 3))
Y_encoded = Ymodel(Y_input)

merged = concatenate([Z_encoded, X_encoded, Y_encoded])
dense1 = Dense(dense_scale)(merged)
#bn1 = BatchNormalization(momentum = .95)(dense1)
ReLU1 = ReLU()(dense1)
dropout1 = Dropout(dropout_rate)(ReLU1)
dense2 = Dense(dense_scale)(dropout1)
#bn2 = BatchNormalization(momentum = .95)(dense2)
ReLU2 = ReLU()(dense2)
dropout2 = Dropout(dropout_rate)(ReLU2)
dense3 = Dense(dense_scale // 2)(dropout2)
#bn3 = BatchNormalization(momentum = .95)(dense3)
ReLU3 = ReLU()(dense3)
dropout3 = Dropout(dropout_rate)(ReLU3)
output = Dense(1, activation = "sigmoid")(dropout3)

#Create the model
model = Model(inputs = [Z_input, X_input, Y_input], outputs = output)

#Summarize the model
model.summary()

#compile the model
model.compile(optimizer=Nadam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

#train the model
Checkpoint_Loss = ModelCheckpoint('/home/admin/Desktop/aerogel_CNNs/loss_FOV100_3.h5', verbose=1, save_best_only=True, monitor='val_loss')
Checkpoint_Acc = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/acc_FOV100_3.h5', verbose=1, save_best_only=True, monitor='val_acc')
model.fit_generator(
	generator = TrainGenerator,
	steps_per_epoch = len(trainAnswers) // batch_size,
	epochs = 150,
	verbose = 2,
	validation_data = ValGenerator,
	validation_steps = len(valAnswers) // batch_size,
	callbacks = [Checkpoint_Acc, Checkpoint_Loss],
	class_weight = class_weights
	)

#See performance on testing set
high_acc = load_model('/home/admin/Desktop/Saved_CNNs/acc_FOV100_3.h5')
low_loss = load_model('/home/admin/Desktop/aerogel_CNNs/loss_FOV100_3.h5')

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








