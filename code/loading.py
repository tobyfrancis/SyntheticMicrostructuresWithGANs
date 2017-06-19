import numpy as np
import h5py
import crystallography as xtal
import pandas as pd
import gc

def rotate(dataset,big_center,small_center,wlh,rotation_matrix):
    def rot(index):
        ijk = small_center - np.matrix(np.unravel_index(index,wlh)).reshape(3,1)   
        hkl = np.array(np.round(big_center + rotation_matrix*ijk),dtype=int)
        return dataset[hkl]

    return rot

def fuZqu(sym):
    def fzQu(quat):
        quat = xtal.do2qu(quat)
        fZquat = sym.fzQu(quat)
        return xtal.qu2do(fZquat)
    return fzQu

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

    voxels = pd.DataFrame(voxel_dict,columns=columns,index=list(voxel_ids.flatten()))
    clusters = pd.DataFrame(cluster_dict,columns=columns,index=list(cluster_ids.flatten()))
    voxels.update(clusters)
    dataset = np.array(voxels.values[:,1:],dtype='float32')
    
    del voxel_quats
    del voxel_ids
    del cluster_quats
    del cluster_ids
    del voxel_dict
    del cluster_dict
    del voxels
    del clusters
    gc.collect()
    return dataset.reshape(shape)

def random_rotated_cube(dataset):
    rotation_matrix = xtal.cu2om(xtal.randomOrientations(1))[0]

    d = min(dataset.shape[0],dataset.shape[1],dataset.shape[2]) #diameter of sphere in prism
    a = int(2*np.sqrt(d**2/12)) #inner cube dimension
    wlh = np.array([a,a,a])

    index_array = np.arange(a**3)

    small_center = np.matrix(wlh/2).reshape(3,1) #center of inner cube
    big_center = np.matrix([dataset.shape[0]/2,dataset.shape[1]/2,dataset.shape[2]/2]).reshape(1,3)
    rot_func = rotate(dataset,big_center,small_center,wlh,rotation_matrix)
    return np.array(list(map(rot_func,index_array))).reshape(wlh[0],wlh[1],wlh[2],4)

# TODO:
'''


    
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

    Footnote 4:
    The hard-coded crop here was determined using Paraview
    by IPF Magnitude:
        X: 27 - 796
        Y: 33 - 460
        Z: 0 - 199
''' 