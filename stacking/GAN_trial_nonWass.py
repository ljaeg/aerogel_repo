import numpy as np 
#from PIL import Image
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, GlobalMaxPooling2D, Reshape, UpSampling2D, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend
from keras.constraints import MinMaxNorm
from keras.optimizers import RMSprop, Adam
from keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import os

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

conv_scale = 32
kernel_size = (3, 3)
dense_scale = 8
Save_dir = "/home/admin/Desktop/aerogel_repo/mnist.h5"
img_save_dir = "/home/admin/Desktop/aerogel_repo/mnist_ims"

def load_real_samples(number, amount = 800):
	x = []
	i = 0
	added = 0
	while added < amount:
		if y_train[i] == number:
			added += 1
			x.append(x_train[i])
		i += 1
	X = np.array(x).astype("float32")
	X = np.expand_dims(X, axis = 3)
	y = np.ones(amount)
	return (X - 127.5) / 127.5, y

def load_all_real_samples(amount = 1000):
	(x_train, _), (_, _) = mnist.load_data()
	X = np.expand_dims(x_train, axis = -1)
	X = X.astype("float32")
	y = np.ones(X.shape[0])
	return X, y

def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

def make_discriminator():
	mmn = MinMaxNorm(min_value = -.01, max_value = .01)
	model = Sequential()
	model.add(Conv2D(conv_scale, kernel_size, padding = "same"))
	model.add(LeakyReLU(alpha = .2))
	model.add(Conv2D(2*conv_scale, kernel_size, padding = "same"))
	model.add(LeakyReLU(alpha = .2))
	model.add(Conv2D(2*conv_scale, kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .8))
	model.add(LeakyReLU(alpha = .2))
	model.add(Conv2D(4*conv_scale, kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .8))
	model.add(LeakyReLU(alpha = .2))
	model.add(Flatten())
	model.add(Dense(1, activation = "sigmoid"))
	model.compile(optimizer = Adam(.0002, .5), loss = binary_crossentropy, metrics = ["accuracy"])
	return model

def make_generator(latent_dim = 100):
	model = Sequential()
	model.add(Dense(128 * 7 * 7, activation = "relu", input_shape = (latent_dim,)))
	model.add(Reshape((7, 7, 128)))
	model.add(UpSampling2D())
	model.add(Conv2D(4*conv_scale, kernel_size = kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .8))
	model.add(Activation("relu"))
	model.add(UpSampling2D())
	model.add(Conv2D(2*conv_scale, kernel_size = kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .8))
	model.add(Activation("relu"))
	model.add(Conv2D(1, kernel_size = kernel_size, padding = "same", activation = "tanh"))
	return model

def make_combined(generator, discriminator):
	discriminator.trainable = False 
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(optimizer = Adam(.0002, .5), loss = binary_crossentropy)
	return model

def generate_fake_samples(generator, latent_dim, n_samples, noise):
	X = generator.predict(noise)
	y = np.zeros(n_samples)
	return X, y

def save_ims(epoch, generator, latent_dim):
	epoch_number = epoch + 1
	noise = np.random.randn(9 * latent_dim).reshape(9, latent_dim)
	gen_ims = generator.predict(noise)
	for i, im in enumerate(gen_ims, 1):
		plt.subplot(3, 3, i)
		plt.imshow(im.reshape(28, 28), cmap = "gray")
	plt.savefig(os.path.join(img_save_dir, "epoch_{}.png".format(epoch_number)))
	plt.close()



def train(generator, discriminator, combined, latent_dim = 100, epochs = 100, batch_size = 128, number_to_do = 8, save_interval = 30):
	#load real samples
	#real, _ = load_real_samples(number_to_do)
	real, _ = load_all_real_samples()

	#perform training for epochs = EPOCHS
	for epoch in range(epochs):
		#Get batch size amount of real images
		idx = np.random.randint(0, real.shape[0], batch_size)
		real_imgs = real[idx]
		real_y = np.ones(batch_size)

		#get batch size amount of generated images
		noise = np.random.randn(latent_dim * batch_size).reshape(batch_size, latent_dim)
		gen_ims, gen_y = generate_fake_samples(generator, latent_dim, batch_size, noise)

		#train discriminator
		d_loss_real, acc_real = discriminator.train_on_batch(real_imgs, real_y)
		d_loss_fake, acc_fake = discriminator.train_on_batch(gen_ims, gen_y)
		d_total_loss = .5 * np.add(d_loss_fake, d_loss_real)
		d_total_acc = .5 * (acc_real + acc_fake)

		#train generator
		g_loss = combined.train_on_batch(noise, real_y)

		#show progress
		print("epoch {}/{}".format(epoch + 1, epochs))
		print("d_loss_real: {}".format(d_loss_real))
		print("d_loss_fake: {}".format(d_loss_fake))
		print("d_total_loss: {}".format(d_total_loss))
		print("d acc: {}".format(d_total_acc))
		print("g_loss: {}".format(g_loss))
		print(" ")

		#save ims
		if not epoch % save_interval:
			save_ims(epoch, generator, latent_dim)

	#save the generator
	generator.save(Save_dir)


def do():
	gen = make_generator()
	disc = make_discriminator()
	comb = make_combined(gen, disc)
	train(gen, disc, comb)

do()



