import numpy as np
import h5py
import crystallography as xtal
import pandas as pd
import gc

from scipy.spatial import KDTree


def fuZqu(sym):
    def fzQu(quat):
        quat = xtal.do2qu(quat)
        fZquat = sym.fzQu(quat)
        return xtal.qu2do(fZquat)
    return fzQu

def rotate(q):
    def rot(v):
        v = xtal.do2qu(v)
        return np.array(xtal.qu2do(q*v*q.conjugate()))
    return rot
        
''' sym = xtal.Symmetry('Cubic') '''
def load_quats(filename,sym):
    f = h5py.File(filename)
    voxel_quats = f['DataContainers']['ImageDataContainer']['CellData']['Quats']
    voxel_ids = f['DataContainers']['ImageDataContainer']['CellData']['ClusterIds']
    cluster_quats = f['DataContainers']['ImageDataContainer']['CellFeatureData']['AvgQuats']
    cluster_ids = f['DataContainers']['ImageDataContainer']['CellFeatureData']['ClusterIds']

    voxel_quats,voxel_ids = np.array(voxel_quats,dtype='float32'),np.array(voxel_ids,dtype=int)
    cluster_quats,cluster_ids = np.array(cluster_quats,dtype='float32'),np.array(cluster_ids,dtype=int)
    shape = voxel_quats.shape

    voxel_quats,voxel_ids = voxel_quats.reshape(-1,4),voxel_ids.reshape(-1,1)
    cluster_quats = cluster_quats.reshape(-1,4)
    voxel_quats,cluster_quats = np.roll(voxel_quats,-1,1),np.roll(cluster_quats,-1,1) #Footnote 1
    
    fzQu = fuZqu(sym) #stage fundamental zone function by symmetry
    cluster_quats = np.array(list(map(fzQu,cluster_quats)))
    cluster_quats = cluster_quats.reshape(-1,4)

    columns = ['clusterId','w','x','y','z']

    voxel_dict = np.hstack((voxel_ids,voxel_quats))
    cluster_dict = np.hstack((cluster_ids,cluster_quats))

    voxels = pd.DataFrame(voxel_dict,index=voxel_ids,columns=columns)
    clusters = pd.DataFrame(cluster_dict,index=cluster_ids,columns=columns)
    quats = voxels.update(clusters).value
    
    del voxel_quats
    del voxel_ids
    del cluster_quats
    del cluster_ids
    del voxel_dict
    del cluster_dict
    del voxels
    del clusters
    gc.collect()
    return quats.reshape(shape)
    
def random_rotation(dataset):
    ''' Perform random rotation on dataset '''
    rot = xtal.cu2qu(xtal.randomOrientations(1))
    center = dataset.shape

    i,j,k = np.expand_dims(np.indices(dataset.shape[:-1]),1)
    l,m,n = np.expand_dims(np.indices(dataset.shape[:-1]),1)
    ijk,lmn = np.vstack((i,j,k)),np.vstack((l,m,n))
    ijk,lmn = np.rollaxis(ijk,0,4),np.rollaxis(lmn,0,4)
    ijk,lmn = ijk.reshape(-1,3),lmn.reshape(-1,3)

    lmn = np.hstack((np.zeros((len(lmn),1)),lmn))
    np.apply_along_axis(rotate(q),v)

# TODO:
'''
    Crop rotated dataset
def crop(dataset,quality_map):

    
    Load training batch
def load_batch(dataset):

'''

''' 
    Footnote 1:
    DREAM3D stores quaternions as [x,y,z,w], we change this to [w,x,y,z]
    for the xtal library.

    Footnote 2:
    Given n (xi+yj+zk) and w (a rotation), q = cos(w/2) + sin(w/2)*n 

    Footnote 3:
    Symmetry operators are pure quaternions, they do not rotate around the axis.

''' 