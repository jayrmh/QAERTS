#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:21:19 2022

@author: hugoyeung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch

def axisAngletoRotation_torch_nerf(r):
    B, _ = r.shape
    zero = torch.zeros((B, 1), dtype=torch.float32, device=r.device)
    skew_v0 = torch.cat([zero, -r[:, 2:3], r[:, 1:2]], dim=1)  # (B, 3)
    skew_v1 = torch.cat([r[:, 2:3], zero, -r[:, 0:1]], dim=1)
    skew_v2 = torch.cat([-r[:, 1:2], r[:, 0:1], zero], dim=1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (B, 3, 3)
    # print('Skew_v', skew_v.shape)
    norm_r = torch.norm(r, dim=1) + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device).unsqueeze(0).repeat(B, 1, 1)
    sin_component = (torch.sin(norm_r) / norm_r).unsqueeze(1).unsqueeze(2)
    cos_component = ((1 - torch.cos(norm_r)) / norm_r**2).unsqueeze(1).unsqueeze(2)
    # print('Sin component', sin_component.shape)
    # print('Cos component', cos_component.shape)
    # print('Eye:', eye.shape)
    R = eye + sin_component * skew_v + (cos_component) * (skew_v @ skew_v)
    return R

def axisAngletoRotation_torch_jay(axs):
    """
    :param axs:  (B, 4) torch tensor
    :return:     (B, 3, 3)
    """
    B, _ = axs.shape

    exponential_points = axs[:, :3]
    angle = axs[:, 3].unsqueeze(1)  # Add unsqueeze operation here
    axis = exponential_points / torch.norm(exponential_points, dim=1, keepdim=True)

    identity = torch.eye(3, device=axs.device, dtype=axs.dtype).unsqueeze(0).expand(B, -1, -1)
    skew_symmetric = torch.zeros(B, 3, 3, device=axs.device, dtype=axs.dtype)
    skew_symmetric[:, 0, 1] = -axis[:, 2]
    skew_symmetric[:, 0, 2] = axis[:, 1]
    skew_symmetric[:, 1, 0] = axis[:, 2]
    skew_symmetric[:, 1, 2] = -axis[:, 0]
    skew_symmetric[:, 2, 0] = -axis[:, 1]
    skew_symmetric[:, 2, 1] = axis[:, 0]

    rotation_matrix = identity + torch.sin(angle) * skew_symmetric + (1 - torch.cos(angle)) * torch.matmul(skew_symmetric, skew_symmetric)

    return rotation_matrix


def eulerAnglesToRotationMatrix_torch_jay(theta):
    """
    :param v:  (B, S, S, 3) torch tensor
    :return:   (B, S, S, 3, 3)
    """
    B,_ = theta.shape
    
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    sinx = sin[:,0:1,:]
    siny = sin[:,1:2,:]
    sinz = sin[:,2:3,:]
    cosx = cos[:,0:1,:]
    cosy = cos[:,1:2,:]
    cosz = cos[:,2:3,:]
    zero = torch.zeros((B,1,1), dtype=torch.float32, device=theta.device, requires_grad=False)
    one = torch.ones((B,1,1), dtype=torch.float32, device=theta.device, requires_grad=False)
    
    x0 = torch.cat([ one,    zero,   zero], dim=-1)  # (3, 1)
    x1 = torch.cat([ zero,    cosx,   -sinx], dim=-1)  # (3, 1)
    x2 = torch.cat([ zero,    sinx,   cosx], dim=-1)  # (3, 1)
    x = torch.cat([x0, x1, x2], dim=-2)  # (3, 3)
    
    y0 = torch.cat([ cosy,    zero,   siny], dim=-1)  # (3, 1)
    y1 = torch.cat([ zero,    one,   zero], dim=-1)  # (3, 1)
    y2 = torch.cat([ -siny,    zero,   cosy], dim=-1)  # (3, 1)
    y = torch.cat([y0, y1, y2], dim=-2)  # (3, 3)
    
    z0 = torch.cat([ cosz,   -sinz,   zero], dim=-1)  # (3, 1)
    z1 = torch.cat([ sinz,   cosz,   zero], dim=-1)  # (3, 1)
    z2 = torch.cat([ zero,    zero,   one], dim=-1)  # (3, 1)
    z = torch.cat([z0, z1, z2], dim=-2)  # (3, 3)
    
    
    R = torch.matmul(z, torch.matmul( y, x ))
    
    return R  # (3, 3)



def eulerAnglesToRotationMatrix_torch(theta):
    """
    :param v:  (B, S, 3) torch tensor
    :return:   (B, S, 3, 3)
    """
    B,S,_ = theta.shape
    
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    sinx = sin[:,:,0:1,:]
    siny = sin[:,:,1:2,:]
    sinz = sin[:,:,2:3,:]
    cosx = cos[:,:,0:1,:]
    cosy = cos[:,:,1:2,:]
    cosz = cos[:,:,2:3,:]
    zero = torch.zeros((B,S,1,1), dtype=torch.float32, device=theta.device, requires_grad=False)
    one = torch.ones((B,S,1,1), dtype=torch.float32, device=theta.device, requires_grad=False)
    
    x0 = torch.cat([ one,    zero,   zero], dim=-1)  # (3, 1)
    x1 = torch.cat([ zero,    cosx,   -sinx], dim=-1)  # (3, 1)
    x2 = torch.cat([ zero,    sinx,   cosx], dim=-1)  # (3, 1)
    x = torch.cat([x0, x1, x2], dim=-2)  # (3, 3)
    
    y0 = torch.cat([ cosy,    zero,   siny], dim=-1)  # (3, 1)
    y1 = torch.cat([ zero,    one,   zero], dim=-1)  # (3, 1)
    y2 = torch.cat([ -siny,    zero,   cosy], dim=-1)  # (3, 1)
    y = torch.cat([y0, y1, y2], dim=-2)  # (3, 3)
    
    z0 = torch.cat([ cosz,   -sinz,   zero], dim=-1)  # (3, 1)
    z1 = torch.cat([ sinz,   cosz,   zero], dim=-1)  # (3, 1)
    z2 = torch.cat([ zero,    zero,   one], dim=-1)  # (3, 1)
    z = torch.cat([z0, z1, z2], dim=-2)  # (3, 3)
    
    
    R = torch.matmul(z, torch.matmul( y, x ))
    
    return R  # (3, 3)


def transform_relative(rot, trans, ref):
    """
    rot: B, S, 3, 3
    trans: B, S, 3
    ref: 3,3
    """
    B,S,_,_ = rot.shape
    
    rot_in = rot.repeat(1,S,1,1)
    rot = torch.repeat_interleave(rot, S, dim=1)
    
    trans_in = trans.repeat(1,S,1)
    trans = torch.repeat_interleave(trans, S, dim=1)
    
    matrix = combine_rot_trans(rot, trans)
    matrix_in = combine_rot_trans_inverse(rot_in, trans_in)
    
    one = torch.ones((1,ref.shape[-1]), dtype=torch.float32, device=ref.device, requires_grad=False)
    ref_expand = torch.cat((ref, one), dim=-2) 
    ref_expand = ref_expand.repeat(B,S**2,1,1) #B,SS,3,3
    
    pts = torch.einsum('bsij, bsjk -> bsik', matrix, ref_expand)#torch.matmul(rot, ref)
    pts = torch.einsum('bsij, bsjk -> bsik', matrix_in, pts)
    
    matrix_final = torch.einsum('bsij, bsjk -> bsik', matrix_in, matrix)
    
    quat = matrix_to_quaternion(matrix_final[:,:,0:3,0:3])
    
    
    
    # rot_in = torch.transpose(rot, -1,-2)
    # trans_in = -trans
    
    # rot = torch.repeat_interleave(rot, S, dim=1)
    # trans = torch.repeat_interleave(trans, S, dim=1)
    # rot_in = rot_in.repeat(1,S,1,1)
    # trans_in = trans_in.repeat(1,S,1)
    
    # # trans = torch.transpose(trans.unsqueeze(-1).repeat(1,1,1,3),-1,-2)
    # # trans_in = torch.transpose(trans_in.unsqueeze(-1).repeat(1,1,1,3),-1,-2)
    
    # trans = trans.unsqueeze(-1).repeat(1,1,1,3)
    # trans_in = trans_in.unsqueeze(-1).repeat(1,1,1,3)
    
    # ref = ref.repeat(B,S**2,1,1) #B,SS,3,3
    
    # 'Transform'
    # pts = torch.einsum('bsij, bsjk -> bsik', rot, ref)#torch.matmul(rot, ref)
    # pts = pts+trans
    # pts = pts+trans_in
    # pts = torch.einsum('bsij, bsjk -> bsik', rot_in, pts)#torch.matmul(rot_in, pts)
    
    return pts[:,:,0:3,:], matrix_final

def transform_absolute(rot, trans, ref):
    """
    rot: B, S, 3, 3
    trans: B, S, 3
    ref: 3,3
    """
    # quat = matrix_to_quaternion(rot)
    B,S,_,_ = rot.shape
    one = torch.ones((1,ref.shape[-1]), dtype=torch.float32, device=ref.device, requires_grad=False)
    
    ref_expand = torch.cat((ref, one), dim=-2) 
    ref_expand = ref_expand.repeat(B,S,1,1) #B,SS,3,3
    
    matrix = combine_rot_trans(rot, trans)
    
    pts = torch.einsum('bsij, bsjk -> bsik', matrix, ref_expand)#torch.matmul(rot, ref)
    
    
    return pts[:,:,0:3,:], matrix

def transform_intra(rot, trans, pts):
    """
    rot: B, S, 3, 3
    trans: B, S, 3
    pts: B, S, 3, N
    """
    B,S,_,_ = rot.shape
    one = torch.ones((pts.shape[0], pts.shape[1], 1,pts.shape[-1]), dtype=torch.float32, device=pts.device, requires_grad=False)
    
    rot_in = rot.repeat(1,S,1,1) # 123123
    rot = torch.repeat_interleave(rot, S, dim=1) #112233
    
    trans_in = trans.repeat(1,S,1)
    trans = torch.repeat_interleave(trans, S, dim=1)
    
    matrix = combine_rot_trans(rot, trans)
    matrix_in = combine_rot_trans_inverse(rot_in, trans_in)
    
    pts_expand = torch.cat((pts, one), dim=-2)
    pts_source = torch.repeat_interleave(pts_expand, S, dim=1)
    pts_target = pts_expand.repeat(1,S,1,1)
    
    pts_source = torch.einsum('bsij, bsjk -> bsik', matrix, pts_source)#torch.matmul(rot, ref)
    pts_source = torch.einsum('bsij, bsjk -> bsik', matrix_in, pts_source)
    
    
    return pts_source[:,:,0:3,:], pts_target[:,:,0:3,:]

def transform_quat_absolute(scale, rot, trans, ref):
    """
    scale: B, S, 1
    rot: B, S, 3, 3 or B, S, 4
    trans: B, S, 3
    ref: N,3
    """
    if rot.shape[-1]==4:
        factor = torch.sqrt(torch.sum(rot**2, dim=-1, keepdim=True))
        quat = rot/factor
    elif rot.shape[-1]==3:
        quat = matrix_to_quaternion(rot)
    B,S,_ = quat.shape
    N,_ = ref.shape
    
    ref_expand = ref.repeat(B,S,1,1)#.permute(0,1,3,2) #B,S,N,3
    scale_expand = scale.unsqueeze(1).repeat(1,1,N,3)
    
    quat = quat.unsqueeze(-2).repeat(1,1,ref_expand.shape[-2],1)
    
    pts = scale_expand*quaternion_apply(quat, ref_expand)
    
    trans = trans.unsqueeze(-2).repeat(1,1,N,1)
    
    pts = pts+trans #B,S,N,3
    
    
    
    return pts



def transform_quat_relative(rot, trans, ref):
    """
    rot: B, S, 3, 3 or B, S, 4
    trans: B, S, 3
    ref: 3,3
    """
    if rot.shape[-1]==4:
        factor = torch.sqrt(torch.sum(rot**2, dim=-1, keepdim=True))
        quat = rot/factor
    elif rot.shape[-1]==3:
        quat = matrix_to_quaternion(rot)
        
     
    B,S,_ = quat.shape
    
    ref_expand = ref.repeat(B,S**2,1,1).permute(0,1,3,2) #B,SS,N,3
    
    
    quat_in = quaternion_invert(quat).unsqueeze(-2).repeat(1,S,ref_expand.shape[-2],1)
    quat = torch.repeat_interleave(quat.unsqueeze(-2).repeat(1,1,ref_expand.shape[-2],1), S, dim=1)
    
    trans_in = -trans.unsqueeze(-1).repeat(1,S,1,ref_expand.shape[-2])
    trans = torch.repeat_interleave(trans.unsqueeze(-1).repeat(1,1,1,ref_expand.shape[-2]), S, dim=1)
    
    # 'transform'
    # pts = quaternion_apply(quat, ref_expand)
    # pts = pts.permute(0,1,3,2)+trans
    # pts = (pts+trans_in).permute(0,1,3,2)
    # pts = quaternion_apply(quat_in, pts)
    
    'transform v2'
    Qq = quaternion_raw_multiply(quat_in, quat)
    
    T = trans+trans_in
    T = (quaternion_apply(quat_in, T.permute(0,1,3,2))).permute(0,1,3,2)
    
    pts = quaternion_apply(Qq, ref_expand)
    pts = pts.permute(0,1,3,2)+T
    
    
    
    return pts, Qq, T

def transform_quat_relative_cat(rot, trans, ref):
    """
    rot: B, SS, 4
    trans: B, SS, 3
    ref: 3,3
    """
    if rot.shape[-1]==4:
        factor = torch.sqrt(torch.sum(rot**2, dim=-1, keepdim=True))
        quat = rot/factor
    elif rot.shape[-1]==3:
        quat = matrix_to_quaternion(rot)
        
     
    B,SS,_ = quat.shape
    
    ref_expand = ref.repeat(B,SS,1,1).permute(0,1,3,2) #B,SS,N,3
    
    
    quat = quat.unsqueeze(-2).repeat(1,1,ref_expand.shape[-2],1)
    
    trans = trans.unsqueeze(-1).repeat(1,1,1,ref_expand.shape[-2])
    
    'transform'
    pts = quaternion_apply(quat, ref_expand)
    pts = pts.permute(0,1,3,2)+trans
    
    
    return pts, quat, trans


    
 

def combine_rot_trans(rot, trans):
    """
    rot: B, S, 3, 3
    trans: B, S, 3
    """
    B,S,_,_ = rot.shape
    
    zero = torch.zeros((B,S,1,3), dtype=torch.float32, device=rot.device, requires_grad=False)
    one = torch.ones((B,S,1,1), dtype=torch.float32, device=rot.device, requires_grad=False)
    
    matrix = torch.cat((rot,zero), dim=-2)
    matrix_trans = torch.cat((trans.unsqueeze(-1), one), dim=-2)
    
    matrix = torch.cat((matrix, matrix_trans), dim=-1)
    
    return matrix

def combine_scale_rot_trans(scale, rot, trans, requires_grad=False):
    """
    scale: B, S, 1
    rot: B, S, 3, 3
    trans: B, S, 3
    
    return B,S,3,4
    """
    B,S,_,_ = rot.shape
    
    matrix_scale = scale.unsqueeze(-1)*torch.eye(3, dtype=torch.float32, device=rot.device).repeat(B, S, 1, 1)  #B,S,3,3
    matrix = torch.einsum('bsij,bsjk->bsik', matrix_scale, rot)
    
    
    # zero = torch.zeros((B,S,1,3), dtype=torch.float32, device=rot.device, requires_grad=requires_grad)
    # one = torch.ones((B,S,1,1), dtype=torch.float32, device=rot.device, requires_grad=requires_grad)
    
    # matrix = torch.cat((matrix,zero), dim=-2)
    # matrix_trans = torch.cat((trans.unsqueeze(-1), one), dim=-2)
    
    matrix_trans = trans.unsqueeze(-1)  #B,S,3,1
    
    matrix = torch.cat((matrix, matrix_trans), dim=-1)
    
    return matrix

def combine_scale_rot_trans_inverse(scale, rot, trans, requires_grad=False):
    """
    scale: B, S, 1
    rot: B, S, 3, 3
    trans: B, S, 3
    
    return B,S,3,4
    """
    B,S,_,_ = rot.shape
    
    matrix_scale = (1/scale).unsqueeze(-1)*torch.eye(3, dtype=torch.float32, device=rot.device).repeat(B, S, 1, 1)  #B,S,3,3
    
    # zero = torch.zeros((B,S,1,3), dtype=torch.float32, device=rot.device, requires_grad=False)
    # one = torch.ones((B,S,1,1), dtype=torch.float32, device=rot.device, requires_grad=False)
    
    # matrix = torch.transpose(rot, -2,-1)
    
    # matrix_trans = torch.einsum('bsij, bsjk -> bsik', matrix, trans.squeeze(-1)) #B,S,3,1
    
    
    dot_0 = torch.einsum('bsij,bsi->bsj', rot[:,:,:,0:1], trans)    #B,S,1
    dot_1 = torch.einsum('bsij,bsi->bsj', rot[:,:,:,1:2], trans)
    dot_2 = torch.einsum('bsij,bsi->bsj', rot[:,:,:,2:3], trans)
    
    matrix = torch.transpose(rot, -2,-1)
    matrix = torch.einsum('bsij,bsjk->bsik', matrix_scale, matrix)
    # matrix = torch.cat((matrix,zero), dim=-2)
    
    matrix_trans = torch.cat((-dot_0, -dot_1, -dot_2), dim=-1) #B,S,3
    matrix_trans = matrix_trans.unsqueeze(-1)  #B,S,3,1
    # matrix_trans = torch.cat((matrix_trans.unsqueeze(-1), one), dim=-2)
    
    matrix = torch.cat((matrix, matrix_trans), dim=-1)
    
    return matrix


def combine_rot_trans_inverse(rot, trans):
    """
    rot: B, S, 3, 3
    trans: B, S, 3
    """
    B,S,_,_ = rot.shape
    
    zero = torch.zeros((B,S,1,3), dtype=torch.float32, device=rot.device, requires_grad=False)
    one = torch.ones((B,S,1,1), dtype=torch.float32, device=rot.device, requires_grad=False)
    
    # matrix = torch.transpose(rot, -2,-1)
    
    # matrix_trans = torch.einsum('bsij, bsjk -> bsik', matrix, trans.squeeze(-1)) #B,S,3,1
    
    
    dot_0 = torch.einsum('bsij,bsi->bsj', rot[:,:,:,0:1], trans)    #B,S,1
    dot_1 = torch.einsum('bsij,bsi->bsj', rot[:,:,:,1:2], trans)
    dot_2 = torch.einsum('bsij,bsi->bsj', rot[:,:,:,2:3], trans)
    
    matrix = torch.transpose(rot, -2,-1)
    matrix = torch.cat((matrix,zero), dim=-2)
    
    matrix_trans = torch.cat((-dot_0, -dot_1, -dot_2), dim=-1) #B,S,3
    matrix_trans = torch.cat((matrix_trans.unsqueeze(-1), one), dim=-2)
    
    matrix = torch.cat((matrix, matrix_trans), dim=-1)
    
    return matrix
    
def pts_to_rot(pts, ref):
    B,S,_,_ = pts.shape
    batch_rot_list = []
    batch_trans_list = []
    for b in range(B):
        set_rot_list = []
        set_trans_list = []
        for s in range(S):
            temp_rot, temp_trans = compute_similarity_transform_torch(pts[b,s], ref)
            temp_rot = temp_rot.unsqueeze(0).unsqueeze(0)
            temp_trans = temp_trans.unsqueeze(0).unsqueeze(0)
            set_rot_list.append(temp_rot)
            set_trans_list.append(temp_trans)
        batch_rot_list.append(torch.cat(set_rot_list, dim=1))
        batch_trans_list.append(torch.cat(set_trans_list, dim=1))
    batch_rot_list = torch.cat(batch_rot_list, dim=0)
    batch_trans_list = torch.cat(batch_trans_list, dim=0)
    
    return batch_rot_list, batch_trans_list
    
    
    
def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return R, t.squeeze()






def angle_loss(pred, gt):
    loss = torch.sin((pred-gt)/2)**2
    
    loss=torch.mean(loss)
    
    return loss

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # clipping is not important here; if q_abs is small, the candidate won't be picked
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    
def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def quaternion_raw_multiply(a, b):
    # """
    # Multiply quaternion(s) q with quaternion(s) r.
    # Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    # Returns q*r as a tensor of shape (*, 4).
    # """
    # assert a.shape[-1] == 4
    # assert b.shape[-1] == 4
    
    # original_shape = a.shape
    
    # # Compute outer product
    # terms = torch.bmm(b.view(-1, 4, 1), a.view(-1, 1, 4))

    # w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    # x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    # y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    # z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    # return torch.stack((w, x, y, z), dim=1).view(original_shape)

    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

    # """
    # Multiply two quaternions.
    # Usual torch rules for broadcasting apply.

    # Args:
    #     a: Quaternions as tensor of shape (..., 4), real part first.
    #     b: Quaternions as tensor of shape (..., 4), real part first.

    # Returns:
    #     The product of a and b, a tensor of quaternions shape (..., 4).
    # """
    # # aw, ax, ay, az = torch.unbind(a, -1)
    # # bw, bx, by, bz = torch.unbind(b, -1)
    # ow = a[:,:,:,0:1] * b[:,:,:,0:1] - a[:,:,:,1:2] * b[:,:,:,1:2] - a[:,:,:,2:3] * b[:,:,:,2:3] - a[:,:,:,3:4] * b[:,:,:,3:4]
    # ox = a[:,:,:,0:1] * b[:,:,:,1:2] + a[:,:,:,1:2] * b[:,:,:,0:1] + a[:,:,:,2:3] * b[:,:,:,3:4] - a[:,:,:,3:4] * b[:,:,:,2:3]
    # oy = a[:,:,:,0:1] * b[:,:,:,2:3] - a[:,:,:,1:2] * b[:,:,:,3:4] + a[:,:,:,2:3] * b[:,:,:,0:1] + a[:,:,:,3:4] * b[:,:,:,1:2]
    # oz = a[:,:,:,0:1] * b[:,:,:,3:4] + a[:,:,:,1:2] * b[:,:,:,2:3] - a[:,:,:,2:3] * b[:,:,:,1:2] + a[:,:,:,3:4] * b[:,:,:,0:1]
    # return torch.cat((ow, ox, oy, oz), dim=-1)

def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    # q = torch.cat([quaternion[...,0:1], -quaternion[...,1:]+quaternion[...,1:]/1e5], dim=-1)
    q = torch.cat([quaternion[...,0:1], -quaternion[...,1:]], dim=-1)
         
    # sign = torch.tensor([1, -1, -1, -1], device=quaternion.get_device(), requires_grad=False, dtype=torch.float)

    return q #* sign#quaternion.new_tensor([1, -1, -1, -1])



def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def quaternion_loss(pred, gt):
    plus = torch.sqrt(torch.sum((pred+gt)**2, dim=-1, keepdim=False))
    minus = torch.sqrt(torch.sum((pred-gt)**2, dim=-1, keepdim=False))
    
    return torch.mean(torch.min(plus,minus))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe