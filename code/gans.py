from keras
def gMax(y_true,y_predict):
	
def train_gan(D,G,folder,optimizer='adagrad'):
	epoch = []
	D_loss = []
	G_loss = []
	for epoch in epochs:
		x,stats,y = generate_batch(G,folder)
		train_switch(D,G)
		D.train_on_batch(x,y)
		train_switch(D,G)
		if epoch > minmax:
			G.train_on_batch(x,y)
		else:
			G.train_on_batch(x,np.ones(len(x)))
			if epoch == minmax:
				switch_loss(G,gMax)
				
		G.train_on_batch(x,y)
		if epoch == minmax:
			change_loss(G,
		
