'''
Part of the code is adapted from Christoph Gohlke's code (https://www.lfd.uci.edu/~gohlke/code/transformations.py.html).
'''
from torch.utils import data
import scipy.io as sio
import numpy as np
from scipy.interpolate import interpn
import math
import itertools
import random
from random import uniform
from imgaug import augmenters as iaa
import cv2
from skimage.morphology import convex_hull_image
import sys
import scipy
from scipy import ndimage
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
import torchio as tio
import SimpleITK as sitk
import csv
import os
import nibabel as nib
import torch
import torch.nn.functional as F

from geometry import combine_scale_rot_trans_inverse, combine_scale_rot_trans


def find_norm_vector(azimuth, elevation):
#    azimuth = azimuth*2*math.pi/360
#    elevation = elevation*2*math.pi/360

    a = np.cos(elevation)*np.cos(azimuth)
    b = np.cos(elevation)*np.sin(azimuth)
    c = np.sin(elevation)
    #d = r*((a**2+b**2+c**2)**0.5)-(a*80+b*80+c*80)

    return np.array([a,b,c])

def R_2vect(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = np.arccos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    R = np.zeros((3,3))
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)

    return R

def EulerToRotation_Z(angle):
    angle = angle*2*math.pi/360
    ca = np.cos(angle)
    sa = np.sin(angle)

    R = np.zeros((3,3))
    R[0,0] = ca
    R[0,1] = -sa
    R[1,0] = sa
    R[1,1] = ca
    R[2,2] = 1.0

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])#*360/2/math.pi

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    direction = np.squeeze(direction)
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def image_normalize(image):
    image_normalized = image/127.5-1

#    image_non_zero = image[np.nonzero(image)]
#    image_mean = np.mean(image_non_zero)
#    image_std = np.std(image_non_zero)
#
#    image_normalized = (image-image_mean)/image_std
#    image_normalized[image==0] = 0

    return image_normalized

def procrustes(X, Y, scaling=True, reflection=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def grid_translation(grid, x, y, z):
    grid_copy = grid.copy()
    direction_row = grid[:,0,0]-grid[:,0,159]
    direction_col = grid[:,0,0]-grid[:,159,0]
    normal = np.cross(direction_row, direction_col)

    direction_row = unit_vector(direction_row)*x
    direction_col = unit_vector(direction_col)*y
    normal = unit_vector(normal)*z

    movement = direction_row+direction_col+normal
    grid_copy[0,:,:] = grid_copy[0,:,:]+movement[0]
    grid_copy[1,:,:] = grid_copy[1,:,:]+movement[1]
    grid_copy[2,:,:] = grid_copy[2,:,:]+movement[2]

    return grid_copy, movement

def Sampling_grid(azimuth, elevation, r_z, t_x, t_y, t_z, rotation_random=None):
    # Normal vector of the plane
    norm_vector = find_norm_vector(azimuth, elevation)
    norm_vector = norm_vector/np.linalg.norm(norm_vector)
    if rotation_random is not None:
        norm_vector = np.matmul(rotation_random, norm_vector)
    # norm_vector[2]=abs(norm_vector[2])

    # Rotation matrix
    rotation_z = EulerToRotation_Z(r_z)
    rotation_p = R_2vect(np.array([0,0,1]), norm_vector)
    rotation = np.matmul(rotation_p,rotation_z)
#    rotation = np.matmul(rotation_random,rotation)

    # euler angle
    #euler = rotationMatrixToEulerAngles(rotation)
    angle_xy = np.array([azimuth, elevation])
    angle_z = np.array([r_z*2*math.pi/360])
    
    # Grid for sampling
    sampling_xrange = np.arange(-80,80)
    sampling_yrange = np.arange(-80,80)
    X, Y = np.meshgrid(sampling_xrange, sampling_yrange)
    grid = np.dstack([X, Y])
    grid = np.concatenate((grid,np.zeros([160,160,1])),axis=-1)
    grid_rot = np.einsum('ji, mni -> jmn', rotation, grid)
    grid_final = grid_translation(grid_rot, t_x, t_y, t_z)
    
    diff = grid_final-grid_rot
    
    return grid_final, rotation, np.mean(diff, (-1,-2))


def flood_fill_hull(image):
    image = image.copy().astype(int)    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def tform_from_grid(source_grid, target_grid):
    target_pt1 = np.array([target_grid[0,0,0],
                           target_grid[1,0,0],
                           target_grid[2,0,0]])
    target_pt2 = np.array([target_grid[0,159,0],
                           target_grid[1,159,0],
                           target_grid[2,159,0]])
    target_pt3 = np.array([target_grid[0,159,159],
                           target_grid[1,159,159],
                           target_grid[2,159,159]])
    target = np.stack((target_pt1,target_pt2,target_pt3), axis=0)

    sample_pt1 = np.array([source_grid[0,0,0],
                           source_grid[1,0,0],
                           source_grid[2,0,0]])
    sample_pt2 = np.array([source_grid[0,159,0],
                           source_grid[1,159,0],
                           source_grid[2,159,0]])
    sample_pt3 = np.array([source_grid[0,159,159],
                           source_grid[1,159,159],
                           source_grid[2,159,159]])
    sample = np.stack((sample_pt1,sample_pt2,sample_pt3), axis=0)
    _, tform_pts, tform_parms = procrustes(target, sample, scaling=False, reflection=False)
    
    return tform_parms

class Dataset_end_to_end(data.Dataset):
    def __init__(self, image_list, **params):
        'Initialization'
        self.mode = params['mode']
        self.sample_num = params['sample_num'] # number of normal per volume, 50
        self.size = 160
        self.image_list = image_list
        self.create_grid_ref()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_list)

    def shuffle_list(self):
        random.shuffle(self.image_list)
        
    def __getitem__(self, index):
        'Load data'
        img_vol, img_mask = self.import_volume(self.image_list[index])
        
        'Augment Volume'
        # scale = uniform(0.8,1.2)
        # transform = tio.RandomAffine(
        #     scales=(scale,scale),
        #     # degrees=(-10,10),
        #     # translation=(-10,10)
        #     image_interpolation='bspline'
        #     )
        # img_vol = np.squeeze(transform(img_vol[np.newaxis,...]))
        # img_vol = img_vol**uniform(0.5,2)
        
        # transform = tio.RandomAffine(
        #     scales=(scale,scale),
        #     # degrees=(-10,10),
        #     # translation=(-10,10)
        #     image_interpolation='nearest'
        #     )
        # img_mask = np.squeeze(transform(img_mask[np.newaxis,...]))
        
        transform = tio.Resize(self.size, image_interpolation='linear')
        img_vol = np.squeeze(transform(img_vol[np.newaxis,...]))
        img_vol = img_vol**uniform(0.5,2)
        
        transform = tio.Resize(self.size, image_interpolation='nearest')
        img_mask = np.squeeze(transform(img_mask[np.newaxis,...]))
        
        'To torch'
        img_vol_torch = torch.from_numpy(img_vol).unsqueeze(0).unsqueeze(0) #1,1,H,W,D
        img_mask_torch = torch.from_numpy(img_mask).unsqueeze(0).unsqueeze(0) #1,1,H,W,D
        
        
        'Sample'
        images = []#np.empty((self.sample_num,1,self.size,self.size))
        y = []#np.zeros((self.sample_num,3))
        scales = []
        rotations = []
        translations = []
        
        for count, num in enumerate(range(self.sample_num)):
            while True:
                'Get the parameters for sampling'
                r_r = random.uniform(-75, -25)
                r_c = random.uniform(-60, 60)
                r_z = random.uniform(-90, 90)#random.uniform(-45, 45)
                t_z = random.uniform(-40, 60)#random.uniform(-10, 10)
                
                # r_r = -0
                # r_c = -0
                # r_z = 0
                # t_z = 0
                
                sampling_grid_zero_center, _, _= self.Sampling_grid(r_r, r_c, r_z, 0, 0, t_z)
                sampling_grid = sampling_grid_zero_center + self.size//2
                
                'sample image slice'
                xx = np.arange(self.size)
                yy = np.arange(self.size)
                zz = np.arange(self.size)
                interp_arr = interpn((xx, yy, zz), img_vol, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0)
                img_slice = interp_arr.reshape((self.size,self.size))
                interp_arr_mask = interpn((xx, yy, zz), img_mask, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0, method='nearest')
                mask = interp_arr_mask.reshape((self.size,self.size))
                mask_original = mask.copy()
                
                'Plot'
                if False:
                    plt.figure()
                    for i,z in enumerate(range(-60,60,5)):
                        plt.subplot(6,6,i+1)
                        sampling_grid_zero_center, _, _= self.Sampling_grid(r_r, r_c, r_z, 0, 0, z)
                        sampling_grid = sampling_grid_zero_center + self.size//2
                        
                        'sample image slice'
                        xx = np.arange(self.size)
                        yy = np.arange(self.size)
                        zz = np.arange(self.size)
                        interp_arr = interpn((xx, yy, zz), img_vol, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0)
                        img_slice = interp_arr.reshape((self.size,self.size))
                        interp_arr_mask = interpn((xx, yy, zz), img_mask, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0, method='nearest')
                        mask = interp_arr_mask.reshape((self.size,self.size))
                        mask_original = mask.copy()
                        plt.imshow(img_slice*mask)
                        plt.gca().set_title(str(z)+' '+str(np.count_nonzero(mask)))
                        
                
                if np.count_nonzero(mask)>3000:
                    break
            
            
            'get maximum possible scale'
            col_index = [n for n,i in enumerate(list(mask.sum(0))) if i>0 ]
            row_index = [n for n,i in enumerate(list(mask.sum(1))) if i>0 ]
            r1 = row_index[0]
            r2 = row_index[-1]
            c1 = col_index[0]
            c2 = col_index[-1]
            
            
            rescale_factor = random.uniform(0.75, min(self.size/max(r2-r1, c2-c1), 1.8))
            # rescale_factor = self.size/max(r2-r1, c2-c1)
            
            
            sampling_grid_original, _, _ = self.Sampling_grid(r_r, r_c, r_z, 0, 0, t_z, scale=rescale_factor)
            sampling_grid = sampling_grid_original+ self.size//2
            xx = np.arange(self.size)
            yy = np.arange(self.size)
            zz = np.arange(self.size)
            interp_arr = interpn((xx, yy, zz), img_vol, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0)
            img_slice = interp_arr.reshape((self.size,self.size))
            interp_arr_mask = interpn((xx, yy, zz), img_mask, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0, method='nearest')
            mask = interp_arr_mask.reshape((self.size,self.size))
            
            
            
            'get maximum possible translation'
            col_index = [n for n,i in enumerate(list(mask.sum(0))) if i>0 ]
            row_index = [n for n,i in enumerate(list(mask.sum(1))) if i>0 ]
            r1 = row_index[0]
            r2 = row_index[-1]
            c1 = col_index[0]
            c2 = col_index[-1]
            row_shift = random.uniform(-r1,self.size-r2)
            col_shift = random.uniform(-c1,self.size-c2)
            # row_shift = -r1
            # col_shift = -c1
            
            
            row_shift = row_shift*(1/rescale_factor)
            col_shift = col_shift*(1/rescale_factor)
            
            
            'sample image slice'
            sampling_grid_original, rotation, translation = self.Sampling_grid(r_r, r_c, r_z, col_shift, row_shift, t_z, scale=rescale_factor)
            sampling_grid = sampling_grid_original+ self.size//2
            xx = np.arange(self.size)
            yy = np.arange(self.size)
            zz = np.arange(self.size)
            interp_arr = interpn((xx, yy, zz), img_vol, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0)
            img_slice = interp_arr.reshape((self.size,self.size))
            interp_arr_mask = interpn((xx, yy, zz), img_mask, np.transpose(sampling_grid.reshape((3,self.size*self.size))), bounds_error=False, fill_value=0, method='nearest')
            mask = interp_arr_mask.reshape((self.size,self.size))
            mask_original = mask.copy()
            
            'Get rotation and translation for pytorch transform'
            reverse = np.array(((0,0,1), (0,1,0),(1,0,0)))
            translation = np.array((translation[2], translation[1], translation[0]))
            rotation = np.einsum('ij,jk->ik', reverse, rotation)
            
            
            '(Test) sampling using torch'
            # torch_slice, torch_mask = self.sample_slice_torch(img_vol_torch, img_mask_torch, rescale_factor, rotation, translation)
            # plt.imshow(torch_slice)
            # plt.figure()
            # plt.imshow(img_slice)
            
            'random version of skull mask'
            contours = cv2.findContours(cv2.convertScaleAbs(mask).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(contours)>1:
                mask = convex_hull_image(mask).astype(int)
                contours = cv2.findContours(cv2.convertScaleAbs(mask).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(contours)>1:
                    print('more than 1 contour')
            try:
                contours = cv2.approxPolyDP(contours[0], 3, True)
            except:
                pass
            centers, radius = cv2.minEnclosingCircle(contours)
            circle = create_circular_mask(self.size, self.size, center=centers, radius=random.uniform(radius,self.size//2))
            
            mode_num = random.uniform(0,3)
            if mode_num<1:
                img_slice =  img_slice*mask
            elif mode_num>=1 and mode_num<2:
                img_slice =  img_slice*circle
            
            
            
            'Intensity augmentation'
            img_slice = img_slice*np.random.uniform(low=0.99, high=1.01, size=img_slice.shape)

            'Final preprocessing'
            # img_slice = image_normalize(img_slice)
            images.append(img_slice[np.newaxis,::])
            
            
            'Get groundtruth'
            sampling_grid = sampling_grid-self.size//2
            y_temp = self.reference_points(grid=sampling_grid_original)
            
            y.append(y_temp)
            scales.append(np.ones((1,))*rescale_factor)
            rotations.append(rotation)
            translations.append(translation/(self.size//2))
            
            'check if finite'
            if not np.isfinite(img_slice).all():
                print('input not finite')
                
        # self.reconstruct_vol_torch(np.stack(scales), np.stack(rotations), np.stack(translations), np.stack(images), img_vol)
            
        
        return np.stack(images), np.stack(y), np.stack(scales), np.stack(rotations), np.stack(translations), img_vol
        
        
    def import_volume(self, fname):
        'Read file'
        if '.mat' in fname:
            mat = sio.loadmat(fname) # mat contains the img volume and the binar mask of the skull (region of interest)
            img_vol = np.squeeze(mat['img_brain'])
            img_mask = np.squeeze(mat['img_brain_mask'])
            img_mask[img_mask>0]=1
        elif '.nii.gz' in fname:
            nii_data = nib.load(fname)
            img_vol = nii_data.get_fdata()
            
            f = fname
            age = int(f[f.find('_',f.rfind('/'))+1:f.find('days')])
            vol_path = f[0:f.rfind('/')]
            mask_data = nib.load(os.path.join(vol_path, str(round(age/7))+'wks.nii.gz').replace('scans_aligned_scaled','masks_aligned_scaled'))
            img_mask = mask_data.get_fdata()
        else:
            pass
        
        'normalize'
        img_vol = img_vol/img_vol.max()
        
        return img_vol, img_mask
    
    def sample_slice_torch(self, img_vol_torch, img_mask_torch, rescale_factor, rotation, translation):
        scale = np.array(((1/rescale_factor,0,0), (0,1/rescale_factor,0),(0,0,1/rescale_factor)))
        rotation = np.einsum('ij,jk->ik', scale, rotation)
        rotation = torch.from_numpy(rotation).unsqueeze(0)
        translation = (torch.from_numpy(translation).unsqueeze(0).unsqueeze(-1))/(self.size//2)
        rotation_translation = torch.cat((rotation, translation), dim=-1)
        
        'Official grid'
        grid = F.affine_grid(rotation_translation, (1,1,1,self.size,self.size))
        # grid = grid.squeeze().permute(2,0,1)
        # grid = torch.stack((grid[2],grid[1],grid[0])).unsqueeze(1).permute(1,2,3,0).unsqueeze(0)
        
        'Trial grid'
        # grid = torch.from_numpy(self.sampling_grid_ref/(self.size//2))
        # grid = torch.einsum('ij,jmn->mni', rotation.squeeze(0), grid)
        # translation_grid = translation.squeeze().repeat(self.size, self.size, 1)
        # grid = grid+translation_grid
        # grid = grid.squeeze().permute(2,0,1)
        # grid = torch.stack((grid[2],grid[1],grid[0])).unsqueeze(1).permute(1,2,3,0).unsqueeze(0)
        
        # img_vol_torch = img_vol_torch.permute(0,1,4,2,3)
        
        img_slice = F.grid_sample(img_vol_torch, grid)
        img_slice = img_slice.squeeze().detach().numpy()
        
        mask_slice = F.grid_sample(img_mask_torch, grid, mode='nearest')
        mask_slice = mask_slice.squeeze().detach().numpy()
        
        return img_slice, mask_slice
    
    def reconstruct_vol_torch(self, scales, rotations, translations, images, vol):
        '''
        check reconstructed match with original
        '''
        
        'to torch'
        scales = torch.from_numpy(1/scales).unsqueeze(1)    #grid_scale
        rotations = torch.from_numpy(rotations).unsqueeze(1)
        translations = torch.from_numpy(translations).unsqueeze(1)
        images_torch = torch.from_numpy(images)
        
        
        
        
        'slice to volume'
        B, C, H, W = images_torch.size()
        
        images_torch = images_torch.unsqueeze(-3)
        
        zeros = torch.zeros((B, C, (H-1)//2, H, W))
        ones = torch.ones((B, 1, 1, H, W))
        
        images_torch = torch.cat((zeros, images_torch/2, images_torch/2, zeros), dim=-3)
        
        masks = torch.cat((zeros+1e-7, ones/2, ones/2, zeros+1e-7), dim=2)
        
        images_torch = torch.cat((images_torch, masks), dim=1)
        
        
 
        'get grid'
        matrix = combine_scale_rot_trans_inverse(scales, rotations, translations)     #N,1,3,4
        matrix = matrix.squeeze(1)
        grid = F.affine_grid(matrix, images_torch.size())
        # grid = torch.cat((grid[...,2:],grid[...,1:2],grid[...,0:1]), dim=-1)
        
        
        
        'transform'
        images_torch = F.grid_sample(images_torch, grid)
        
        
        'sum'
        images_torch = images_torch.sum(0)
        images_torch = images_torch[0]/images_torch[1]
        # images_torch = images_torch.squeeze().detach().numpy()
        
        'resample'
        num = 3
        scale = torch.eye(3)*scales[num,0]
        rotation = torch.einsum('ij,jk->ik', scale, rotations[num,0]).unsqueeze(0)
        translation = translations[num].unsqueeze(-1)
        rotation_translation = torch.cat((rotation, translation), dim=-1)
        
        grid = F.affine_grid(rotation_translation, (1,1,1,self.size,self.size))
        # grid = grid.squeeze().permute(2,0,1)
        # grid = torch.stack((grid[2],grid[1],grid[0])).unsqueeze(1).permute(1,2,3,0).unsqueeze(0)
        
             
        img_slice = F.grid_sample(images_torch.unsqueeze(0).unsqueeze(0), grid)
        # img_slice = F.grid_sample(torch.from_numpy(vol).unsqueeze(0).unsqueeze(0), grid)
        img_slice = img_slice.squeeze().detach().numpy()
        
        plt.imshow(img_slice)
        plt.figure()
        plt.imshow(images[num,0])
        
        
            
    def create_grid_ref(self):
        sampling_xrange = np.arange(-self.size//2, self.size//2)
        sampling_yrange = np.arange(-self.size//2, self.size//2)
        X, Y = np.meshgrid(sampling_xrange, sampling_yrange)
        grid = np.dstack([X, Y])
        grid = np.concatenate((grid,np.zeros([self.size,self.size,1])),axis=-1)
        rotation = np.array(((1,0,0),(0,1,0),(0,0,1)))
        # rotation = eulerAnglesToRotationMatrix([0,math.pi/2,0])
        sampling_grid_ref = np.einsum('ji, mni -> jmn', rotation, grid)
        self.sampling_grid_ref = sampling_grid_ref
    
    def Sampling_grid(self, r_r, r_c, r_z, t_x, t_y, t_z, scale=None):
        'random out-of-plane rotation'
        direction_row = self.sampling_grid_ref[:,0,0]-self.sampling_grid_ref[:,0,-1]
        direction_col = self.sampling_grid_ref[:,0,0]-self.sampling_grid_ref[:,-1,0]
        # direction_col = np.cross(direction_row, direction_col)
        
        rot_row = rotation_matrix(r_r/360*2*math.pi, direction_row, point=None)[0:3,0:3] #top-bottom head
        rot_col = rotation_matrix(r_c/360*2*math.pi, direction_col, point=None)[0:3,0:3]
        matrix_ground = rot_col@rot_row
        sampling_grid = np.einsum('ji, imn -> jmn', matrix_ground, self.sampling_grid_ref)
        
        'Normal of the grid'
        direction_row = sampling_grid[:,0,0]-sampling_grid[:,0,-1]
        direction_col = sampling_grid[:,0,0]-sampling_grid[:,-1,0]
        norm_vector = np.cross(direction_row, direction_col)
        norm_vector = norm_vector/np.linalg.norm(norm_vector)
        
        'Rotation matrix'
        rotation_z = EulerToRotation_Z(r_z)
        rotation_p = R_2vect(np.array([0,0,1]), norm_vector)
        rotation = np.matmul(rotation_p,rotation_z)
    #    rotation = np.matmul(rotation_random,rotation)
    
        'Generate grid'
        grid_rot = np.einsum('ji, imn -> jmn', rotation, self.sampling_grid_ref)
       
        if scale is not None:
            grid_rot = grid_rot*(1/scale)
        grid_final, translation = grid_translation(grid_rot, t_x, t_y, t_z)

        
        return grid_final, rotation, translation
    
    def reference_points(self, grid=None):
        if grid is None:
            grid = self.sampling_grid_ref
            
        y1 = np.array([grid[0,0,0],
                        grid[1,0,0],
                        grid[2,0,0]])
        y2 = np.array([grid[0,-1,0],
                        grid[1,-1,0],
                        grid[2,-1,0]])
        y3 = np.array([grid[0,-1,-1],
                        grid[1,-1,-1],
                        grid[2,-1,-1]])
        y4 = np.array([grid[0,0,-1],
                        grid[1,0,-1],
                        grid[2,0,-1]])

        return np.stack((y1, y2, y3, y4))
    
