from keras.layers.pooling import GlobalMaxPooling3D, GlobalAveragePooling3D
import keras.backend as K
from loading import *
from discriminator import *
from generator import *
from lambdas import *

''' 
	A GANs model for synthetic microstructure generation fitted to 
	data. The input to the generator is a set of microstructural 
	statistics (ODF, MDF, RDF, and shape distribution) concatenated
	with a random vector. The input to the discriminator is a 
	80x80x80x4 microstructure, where the 4-dimensional vector at each
	voxel represents the quaternion orientation of the pixel. 

	The concatenation of a random vector provides a stochastic output, 
	given a fixed set of microstructural statistics, which is useful
	for the creation of a multitude of representative microstructures.
'''
def G_Max(y_true,y_predict):
	#Maximizes crossentropy by Minimizing negative crossentropy
	return -1 * K.categorical_crossentropy(y_pred,y_true)

def set_trainable(model,trainable):
	for layer in model.layers:
		layer.trainable = trainable
	
def gan(D,G,optimizer,stats_shape):
	inputs = Input(shape = stats_shape)
	set_trainable(G,True)
	X = G(inputs[0])
	X = Concatenate(X,inputs[1])
	set_trainable(D,False)
	outputs = D(X)
	model = Model(inputs=inputs,outputs=outputs)
	model.compile(optimizer=optimizer,loss='categorical_crossentropy')

def train_switch(D,G):
	if both_trainable(D,G):
		set_trainable(G,False)
	elif G.trainable == True:
		set_trainable(G,False)
		set_trainable(D,True)
	else:
		set_trainable(D,False)
		set_trainable(G,True)

def train_gan(D,G,folder,optimizer='adagrad'):
	GAN = gan(D,G)
	epoch_tracking = []
	D_loss_tracking = []
	G_loss_tracking = []

	dataset = pre_process(dataset)

	minmax = False
	minmax_switch = 0
	epochs = 10000

	print('Training GANs model...')
	for epoch in range(epochs):
		x = load_microstructure_batch(G,dataset)
		train_switch(D,G)
		#TODO: When I get the data, write the generate_batch code
		D_x,D_y = generate_D_batch(G,x)
		D_loss = D.train_on_batch(D_x,D_y,shuffle=True)
			
		train_switch(D,G)
		G_x = x
		G_y = np.vstack((np.ones(len(x)),np.zeros(len(x)))).T

		if epoch == minmax_switch:
			GAN.compile(optimizer=GAN.optimizer,loss=G_Max)
			minmax = True

		if minmax:
			extend = np.vstack((np.zeros(len(x)),np.ones(len(x)))).T
		else: #minmin
			extend = np.vstack((np.ones(len(x)),np.zeros(len(x)))).T	
		
		G_y = np.vstack((G_y,extend))
		G_loss = GAN.train_on_batch(G_x,G_y,shuffle=True)
		
		print('{}/{}'.format(epoch+1,epochs),end='\r')

if __name__ == '__main__':
	optimizer = 'adagrad'
	D,G = discriminator(optimizer),generator(optimizer)
	train_gan(D,G,folder)	
