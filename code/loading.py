import numpy as np
import h5py

quaternion_symmetry =   [
                            []
                        ]
    
def same_zone(first_quat,second_quat):
    ''' 
        We first compute the misorientation between the two quaternions, which is a quaternion.
        
    '''
    compute_misorientation(first_quat,second_quat)

def load_quats(f):
    dataset = np.array(f['DataContainers']['ImageDataContainer']['CellData']['Quats'],dtype='float32')
    dataset = dataset[:,33:461,27:797] #Reference Footnote 1
    return pre_process(dataset)

def pre_process(dataset):
    ''' Transforms all quaternions so as to lie in one fundamental zone '''
    first_quat = dataset[0,0,0]
    not_same = list(zip(*np.where(not same_zone(dataset,fundamental_zone))))
    
def random_rotation(dataset):
    ''' Perform random rotation on dataset '''

def crop(dataset,quality_map):
    ''' Crop rotated dataset '''
    bad_quality = np.where(quality == 0.0)

def load_batch(dataset):


''' 
    Footnote 1:
    The hard-coded crop here was determined using Paraview for  the IPF Color Magnitude
    IPF Magnitude:
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