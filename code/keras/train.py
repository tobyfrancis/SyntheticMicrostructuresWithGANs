from gan import *
from loading import *

import crystallography as xtal

def train_gan(D,G,GAN,dataset,cube_stats,optimizer='adagrad'):
	epoch_tracking = []
	D_loss_tracking = []
	G_loss_tracking = []

    batch_size = 8
	minmax = False
	minmax_switch = 2500
	epochs = 50000

	print('Training GANs model...')
	for epoch in range(epochs):
		x = load_batch(dataset,batch_size,cube_stats)
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
	D,G = discriminator(optimizer), generator(optimizer)
    GAN = gan(D,G)
    symmetry = xtal.Symmetry('Cubic')
	dataset = load_quats('rene88/2_19_13_ped_reconstruction.dream3d',symmetry)
    rot_cube_stats = cube_stats(dataset)
    

