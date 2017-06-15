''' This is the generator for the point-cloud microstructure.'''

def generator_model():
	'''Given a statistics vector, outputs quaternion point-cloud microstructure.'''
	model = Sequential()
	model.add(Conv1D(5*5*5,3,padding='same',activation='relu'))
	model.add(Lambda(transpose,output_shape=transposed_shape))
	model.add(Reshape(5,5,5,16+odf_length))
	model.add(DeConv3D(2,128,activation='relu'))
	model.add(DeConv3D(2,64,activation='relu'))
	model.add(DeConv3D(2,32,activation='relu'))
	model.add(DeConv3D(2,16,activation='relu'))
	model.add(Conv3D(4,(3,3,3),trainable=False))
	

def generator(optimizer,mesh=False):
    if not mesh:
        model = generator_model(optimizer)
        model.compile(optimizer=optimizer,loss='mse')
        return model 

    if mesh:
        model = Sequential()
        model.add(generator_model())
        model.add(meshing_model())
        model.add(voxelizing_model())
        model.compile(optimizer=optimizer,loss='mse')
        return model 