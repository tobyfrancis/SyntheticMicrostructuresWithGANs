from keras.layers.pooling import GlobalMaxPooling3D, GlobalAveragePooling3D
import keras.backend as K
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

def squeeze_3D(x):
	return K.reshape(x.shape[1],x.shape[2])

def squeeze_3Dshape(shape):
	return (shape[1],shape[2])

def squeeze_6D(x):
	return K.reshape(x.shape[1],x.shape[2],x.shape[3],x.shape[4],x.shape[5])

def squeeze_6Dshape(shape):
	return (shape[1],shape[2],shape[3],shape[4],shape[5])

def expand(x):
	return
	
def gan(D,G,optimizer,a):
	inputs = [Input(batch_shape=(1,8,16)),Input(batch_shape=(1,8,a,a,a,4))]
	set_trainable(G,True)
	X = G(inputs[0])
	X = merge(X,inputs[1],concat_axis=0)
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
