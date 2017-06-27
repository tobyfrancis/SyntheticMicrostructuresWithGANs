import numpy as np
import h5py
import crystallography as xtal
import pandas as pd
import gc

def rotate(dataset,big_center,small_center,wlh,rotation_matrix):
    def rot(index):
        ijk = np.matrix(np.unravel_index(index,wlh)).reshape(3,1) - small_center
        hkl = tuple(np.array(np.round(big_center + np.linalg.inv(rotation_matrix)*ijk),dtype=int).flatten())
        return dataset[hkl]

    return rot

def fuZqu(sym):
    def fzQu(quat):
        quat = xtal.do2qu(quat)
        fZquat = sym.fzQu(quat)
        return xtal.qu2do(fZquat)
    return fzQu

def save_fzQu(filename,symmetry):
	''' symmetry == xtal.Symmetry('Cubic') '''
	fzQu = fuZqu(symmetry) #stage fzQu function with symmetry
	f = h5py.File(filename,'r+')
	cluster_quats = f['DataContainers']['ImageDataContainer']['CellFeatureData']['AvgQuats']
	cluster_quats = np.array(cluster_quats,dtype='float32')
	shape = cluster_quats.shape
	cluster_quats = np.roll(cluster_quats,-1,1)
	cluster_quats = np.array(list(map(fzQu,cluster_quats)))
	cluster_quats = np.roll(cluster_quats.reshape(-1,4),1,-1)
	cluster_quats = cluster_quats.reshape(shape)
	del f['DataContainers']['ImageDataContainer']['CellFeatureData']['AvgQuats']
	f['DataContainers']['ImageDataContainer']['CellFeatureData'].create_dataset('AvgQuats',data=cluster_quats)
	print('Done converting all quaternions into Fundamental Zone.')

def load_quats(filename):
    print('Loading File...')
    f = h5py.File(filename)

    dataset = f['DataContainers']['ImageDataContainer']['CellData']['ClusterQuats']
    dataset = np.array(dataset,dtype='float32')
    return dataset

def get_cube_stats(dataset,custom_size=96):
    d = min(dataset.shape[0],dataset.shape[1],dataset.shape[2])
    a = int(2*np.sqrt(d**2/12))
    if custom_size and custom_size <= a:
        a = custom_size
    elif custom_size and custom_size > a:
        raise IndexError('Custom Size too big to ensure rotations will fit in dataset.')
    else:
        pass

    shape = np.array([a,a,a])
    index_array = np.arange(a**3)
    small_center = np.matrix(wlh/2).reshape(3,1)
    big_center = np.matrix([dataset.shape[0]/2,dataset.shape[1]/2,dataset.shape[2]/2]).reshape(3,1)
    return [shape,small_center,big_center,index_array]

def random_rotated_cube(dataset,cube_stats):
    shape,small_center,big_center,index_array = cube_stats
    rotation_matrix = xtal.cu2om(xtal.randomOrientations(1))[0]
    dataset_shape = np.array([dataset.shape[0],dataset.shape[1],dataset.shape[2]])
    padding = 25
    centering = dataset_shape - min(dataset_shape)
    for i in range(3):
        shift = centering[i]-padding
        if shift > 5: #just in case the padding makes things negative
            big_center[i] += numpy.random.uniform(low=-shift, high=shift)

    rot_func = rotate(dataset,big_center,small_center,wlh,rotation_matrix)

    return np.array(list(map(rot_func,index_array))).reshape(shape[0],shape[1],shape[2],4)

def batch_loader(cube_stats):
    def load_batch(dataset,batch_size,cube_stats):
        shape = cube_stats[0]
        batch = np.zeros(batch_size,shape[0],shape[1],shape[2],4)
        for i in range(batch_size):
            batch[i] = random_rotated_cube(dataset,cube_stats)
        return batch

''' 
    Footnote 1:
    DREAM3D stores quaternions as [x,y,z,w], we change this to [w,x,y,z]
    for the xtal library.

    Footnote 2:
    Given n (xi+yj+zk) and w (a rotation), q = cos(w/2) + sin(w/2)*n 

    Footnote 3:
    Symmetry operators are pure quaternions, they do not rotate around the axis.

    Footnote 4:
    The hard-coded crop here was determined using Paraview
    by IPF Magnitude:
        X: 27 - 796
        Y: 33 - 460
        Z: 0 - 199
''' 
