from loading import *
from pytorch.train import *
import click
import crystallography as xtal

@click.command()
@click.option('--framework', default='pytorch')
@click.option('--filepath', default='../rene88/6_16_2017_rene88_clustered.dream3d')
@click.option('--symmetry', default='Cubic')
@click.option('--batch_size', default=2)
@click.option('--training_method',default='wasserstein')
@click.option('--epochs',default=1000000)
@click.option('--anneal',default=False)
def train(framework,filepath,symmetry,batch_size,training_method,epochs,anneal):
    dataset = load_quats(filepath,xtal.Symmetry(symmetry))
    stats = get_cube_stats(dataset)
    load_batch = batch_loader(cube_stats)
    if framework.lower() == 'pytorch':
        pytorch_train(dataset,load_batch,batch_size=batch_size,train=training_method)
    elif framework.lower() == 'keras':
        keras_train(dataset,load_batch,batch_size=batch_size,)
    else:
        raise NameError('Framework unavailable: Please choose Keras or Pytorch.')

if __name__ == '__main__':
    train()