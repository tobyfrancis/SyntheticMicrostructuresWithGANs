def discriminator(optimizer):
	''' Given a 3D Microstructure of any shape, 
		output if Experimental or Synthetic'''
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