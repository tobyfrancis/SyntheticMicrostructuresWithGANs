''' Training paradigms from wiseodd: '''
''' https://github.com/wiseodd/generative-models/blob/master/GAN/ '''
import torch
import torch.optim as optim
from generator import *
from discriminator import *

def log(x):
    return torch.log(x + 1e-8)

def reset_grad(D,G):
    G.zero_grad()
    D.zero_grad()

def anneal(X, epoch, epochs):
    if epoch < (epochs*.75):
        
        noise = np.random.normal(loc=1.0,scale=scale,size=size)
        noise = noise.reshape(X.shape)
        X = X*noise
    return X

def wasserstein_train(dataset,load_batch,batch_size,epochs,anneal):
    lr = 1e-4
    G = Generator()
    D = Discriminator()
    G_solver = optim.RMSprop(G.parameters(), lr=lr)
    D_solver = optim.RMSprop(D.parameters(), lr=lr)
    
    
    for epoch in range(epochs):
        for _ in range(5):
            # Sample data
            z = np.random.rand(batch_size,6,6,6,4)
            z = Variable(torch.from_numpy(z)).cuda()
            X = load_batch(dataset,batch_size)
            X = Variable(torch.from_numpy(X)).cuda()

            # Dicriminator forward-loss-backward-update
            G_sample = G(z)
            D_real = D(X)
            D_fake = D(G_sample)

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Housekeeping - reset gradient
            reset_grad()

        # Generator forward-loss-backward-update
        z = np.random.rand(batch_size,6,6,6,4)
        z = Variable(torch.from_numpy(z))
        X = load_batch(dataset,batch_size)
        X = Variable(torch.from_numpy(X))

        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        G_solver.step()

        # Housekeeping - reset gradient
        reset_grad()
    return D,G

def boundary_train(dataset,load_batch,batch_size,epochs,anneal):
    lr = 1e-3
    G = Generator()
    D = Discriminator()
    G_solver = optim.Adam(G.parameters(), lr=lr)
    D_solver = optim.Adam(D.parameters(), lr=lr)
    
    epochs = 1000000
    for epoch in range(epochs):
        z = np.random.rand(batch_size,6,6,6,4)
        z = Variable(torch.from_numpy(z))
        X = load_batch(dataset)
        X = Variable(torch.from_numpy(X))

        # Discriminator
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = -torch.mean(log(D_real) + log(1 - D_fake))

        D_loss.backward()
        D_solver.step()
        reset_grad()

        # Generator
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = 0.5 * torch.mean((log(D_fake) - log(1 - D_fake))**2)

        G_loss.backward()
        G_solver.step()
        reset_grad()
    return D,G

def pytorch_train(dataset,load_batch,batch_size,train,epochs,anneal):
    with torch.cuda.device(1):
        if train.lower() == 'wasserstein':
            D,G = wasserstein_train(dataset,load_batch,batch_size,epochs,anneal)
        elif train.lower() == 'boundary':
            D,G = boundary_train(dataset,load_batch,batch_size,epochs,anneal)
        else:
            raise NotImplementedError('Please specify Wassterstein or Boundary, other training paradigms are not implemented.')

    if os.isdir('models'):
        breaker = True
        i = 0
        while breaker:
            discriminator_path = 'models/pytorch_discriminatory_{}.py'.format(i)
            generator_path = 'models/pytorch_generator_{}.py'.format(i)
            if os.path.exists(discriminator_path) or os.path.exists(generator_path):
                i += 1
            else:
                torch.save(D.state_dict(), discriminator_path)
                torch.save(G.state_dict(), generator_path)
                breaker = False
