from keras.layers.pooling import GlobalMaxPooling3D, GlobalAveragePooling3D
import keras.backend as K
from loading import *
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
		layer.trainable = False

def discriminator(optimizer):
	'''Given a 3D Microstructure, output if Experimental or Synthetic'''
	model = Sequential()
	model.add(Conv3D(16,3))
	model.add(Conv3D(32,3,strides=(2,2,2)))
	model.add(Conv3D(64,3,strides=(2,2,2)))
	model.add(Conv3D(128,3,strides=(2,2,2)))
	model.add(Conv3D(128,3,strides=(2,2,2)))
	model.add(Conv3D(2,3))
	model.add(GlobalMaxPooling3D(data_format='channels_last'))
	model.add(Activation('softmax'))
	model.compile(optimizer=optimizer,loss='categorical_crossentropy')

def generator(optimizer):
	'''Given a statistics vector, output microstructure'''
	model = Sequential()
	model.add(Conv1D(5*5*5,3,padding='same',activation='relu'))
	model.add(Lambda(transpose,output_shape=transposed_shape))
	model.add(Reshape(5,5,5,16+odf_length))
	model.add(DeConv3D(2,128,activation='relu'))
	model.add(DeConv3D(2,64,activation='relu'))
	model.add(DeConv3D(2,32,activation='relu'))
	model.add(DeConv3D(2,16,activation='relu'))
	model.add(Conv3D(4,(3,3,3),trainable=False))
	model.compile(optimizer=optimizer,loss='mse')
	
def gan(D,G,optimizer):
	inputs = Input(shape=)
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
	else:
		if D.trainable == True:
			set_trainable(D,False)
			set_trainable(G,True)
		else:
			set_trainable(G,False)
			set_trainable(D,True)

def train_gan(D,G,folder,optimizer='adagrad'):
	GAN = gan(D,G)
	epoch_tracking = []
	D_loss_tracking = []
	G_loss_tracking = []

	minmax = False
	minmax_switch = 0
	epochs = 10000

	print('Training GANs model...')
	for epoch in range(epochs):
		x = load_microstructure_batch(G,folder)
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
