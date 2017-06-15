import numpy as np
import h5py
import crystallography as xtal

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

def load_quats(f,sym):
    dataset = f['DataContainers']['ImageDataContainer']['CellData']['Quats']
    dataset = np.array(dataset,dtype='float32')
    dataset = dataset[:,33:461,27:797] #Reference Footnote 1
    return np.apply_along_axis(fuZqu(sym),-1,dataset)    
    
def random_rotation(dataset):
    ''' Perform random rotation on dataset '''
    rot = xtal.cu2qu(xtal.randomOrientations(1))
    
    i,j,k = np.expand_dims(np.indices(dataset.shape[:-1]),1)
    l,m,n = np.expand_dims(np.indices(dataset.shape[:-1]),1)
    ijk,lmn = np.vstack((i,j,k)),np.vstack((l,m,n))
    ijk,lmn = np.rollaxis(ijk,0,4),np.rollaxis(lmn,0,4)
    ijk,lmn = ijk.reshape(-1,3),lmn.reshape(-1,3)

    lmn = np.hstack((np.zeros((len(lmn),1)),lmn))
    np.apply_along_axis


def crop(dataset,quality_map):
    ''' Crop rotated dataset '''
    bad_quality = np.where(quality == 0.0)

def load_batch(dataset):

''' 
    Footnote 1:
    The hard-coded crop here was determined using Paraview
    by IPF Magnitude:
        X: 27 - 796
        Y: 33 - 460
        Z: 0 - 199

    Footnote 2:
    DREAM3D stores quaternions as [x,y,z,w], we change this to [w,x,y,z]

    Footnote 3:
    Given n (xi+yj+zk) and w (a rotation), q = cos(w/2) + sin(w/2)*n 

    Footnote 4:
    Symmetry operators are pure quaternions, they do not rotate around the axis.

''' 