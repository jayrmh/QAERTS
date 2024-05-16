import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import glob
import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import sys 
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import platform
import random
import pandas as pd

from model_end_to_end_vunc8 import *
from dataseteuler import *
# from convnext import ConvNeXt
from dataset_end_to_end_vunc1 import Dataset_end_to_end as Dataset_end_to_end
from dataset_end_to_end_vunc1 import procrustes
from geometry import *

try:
    import matplotlib
    matplotlib.use("TKAgg")
except:
    a=1

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.unsqueeze(1).mm(x.unsqueeze(0))
    return res


def euler_to_rotation_matrix(euler_angles):
    batch_size = euler_angles.shape[0]
    rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=euler_angles.dtype, device=euler_angles.device)
    cos = torch.cos(euler_angles)
    sin = torch.sin(euler_angles)
    rotation_matrix[:, 0, 0] = cos[:, 1] * cos[:, 2]
    rotation_matrix[:, 0, 1] = -cos[:, 1] * sin[:, 2]
    rotation_matrix[:, 0, 2] = sin[:, 1]
    rotation_matrix[:, 1, 0] = cos[:, 0] * sin[:, 2] + sin[:, 0] * sin[:, 1] * cos[:, 2]
    rotation_matrix[:, 1, 1] = cos[:, 0] * cos[:, 2] - sin[:, 0] * sin[:, 1] * sin[:, 2]
    rotation_matrix[:, 1, 2] = -sin[:, 0] * cos[:, 1]
    rotation_matrix[:, 2, 0] = sin[:, 0] * sin[:, 2] - cos[:, 0] * sin[:, 1] * cos[:, 2]
    rotation_matrix[:, 2, 1] = sin[:, 0] * cos[:, 2] + cos[:, 0] * sin[:, 1] * sin[:, 2]
    rotation_matrix[:, 2, 2] = cos[:, 0] * cos[:, 1]
    return rotation_matrix

# Function to apply rotation to grid points using rotation matrices
def apply_rotation(grid, rotation_matrix):
    batch_size, _, H, W = grid.shape
    grid_flat = grid.view(batch_size, 3, -1)
    rotated_grid_flat = torch.matmul(rotation_matrix, grid_flat)
    rotated_grid = rotated_grid_flat.view(batch_size, 3, H, W)
    return rotated_grid

def create_model(model, model_name, save_path, paras, dropoutrate=0, weeksage=18, relative=True):
    
    'CUDA for PyTorch'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
#    torch.backends.cudnn.benchmark = False
    
    'model'
    model.apply(weight_init)
    # for p in model.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    # model.load_state_dict(torch.load(current_model))
    model = model.to(device)

    
    'Path and files for saving the models'
    save_path_model = os.path.join(save_path, model_name)
    saved_model = os.path.join(save_path_model,'best_model_mtl_twomeanslw_{}_{}.pth'.format(weeksage, dropoutrate))
    current_model = os.path.join(save_path_model, 'current_model_twomeanslw_{}_{}.pth'.format(weeksage, dropoutrate))
    
    if not os.path.exists(save_path_model):
        os.makedirs(save_path_model)

    'load current weight'
    # model.load_state_dict(torch.load(saved_model))

    'Loss and optimizer'
    criterion = {'points':nn.MSELoss(),
                 'reconstruction':nn.MSELoss()
                 }
    optimizer = optim.Adam(model.parameters(), lr=paras['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=paras['patience'], verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    'Early stop when loss plateu'
    stop_count = 0
    
    'Evaluation metrics'
    loss = {'points':[],
            'overall':[]}
    best = np.Infinity
    loss_on_test = {}
    
    
    'Group all'
    model_group = {'model':model,
                   'name':model_name,
                   'saved_model':saved_model,
                   'current_model':current_model,
                   'optimizer': optimizer,
                   'learning_rate':paras['learning_rate'],
                   'scheduler': scheduler,
                   'criterion':criterion,
                   'loss': loss,
                   'best': best,
                   'stop_count':stop_count,
                   'stop':False,
                   'loss_on_test':loss_on_test,
                   'relative': relative}
    
    return model_group

def run_euler(model_group, local_batch, local_y, local_scale, local_rotation, local_translation,  train=None):
    if train:
        model_group['model'] = model_group['model'].train()
        model_group['optimizer'].zero_grad()
        
    'predict'
    B,C,H,W = local_batch.size()
    _, pred_pts, euler_angles, translation, scale = model_group['model'](local_batch)
    
    'pred points'
    pred_pts = pred_pts.view(B,3,3).permute(1,0,2)
    
    'Transform groundtruth points'
    local_scale = 1/local_scale
    # print('Local Scale', local_scale.shape)
    local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(B, 1, 1)  #B,3,3
    # print('Local Scale', local_scale.shape)
    local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
    # print('Local Rotation', local_rotation.shape)
    local_translation = (local_translation).unsqueeze(-1)
    # print('Local Translation', local_translation.shape)
    local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
    # print('Local Rotation Updated', local_rotation.shape)
    
    local_grid = F.affine_grid(local_rotation, (B,1,1,H,W)).squeeze(1)
    # print('Local Grid', local_grid.shape)
    local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0]))
    
    'Predictions'
    scale = scale.unsqueeze(-1)*torch.eye(3, dtype=scale.dtype, device=scale.device).repeat(B, 1, 1)  #B,3,3
    # print('Pred Scale Angles', scale.shape)
    # print('Euler Angles', euler_angles.shape)
    rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
    # print('Euler Matrix', rot.shape)
    pred_rotation = torch.einsum('bij,bjk->bik', scale, rot)
    # print('Pred Rotation', pred_rotation.shape)
    translation = (translation).unsqueeze(-1)
    # print('PredTranslation', translation.shape)
    pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
    # print('Pred Rotation Updated', pred_rotation.shape)
  
    my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
    param_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))
    # print('Param Pts Shape', param_pts.shape)


    'Loss'
    loss_points = model_group['criterion']['points'](pred_pts*(H/2), local_pts*(H/2)) + (0.5 * model_group['criterion']['points'](param_pts*(H/2), local_pts*(H/2)))
    print('Loss for pose', model_group['criterion']['points'](pred_pts*(H/2), local_pts*(H/2)))
    print('Loss for params', model_group['criterion']['points'](param_pts*(H/2), local_pts*(H/2)))
    
    # loss_pose = model_group['criterion']['points'](quat, local_quat)+model_group['criterion']['points'](scale, local_scale)+model_group['criterion']['points'](translation, local_translation)
    
    loss = loss_points
    if train:
        loss.backward()
        model_group['optimizer'].step()
        model_group['optimizer'].zero_grad()
    
    
    model_group['loss']['overall'].append(loss.item())
    model_group['loss']['points'].append(loss_points.item())   

def run_points(model_group, local_batch, local_y, local_scale, local_rotation, local_translation,  train=None):
    if train:
        model_group['model'] = model_group['model'].train()
        model_group['optimizer'].zero_grad()
        
    'predict'
    B,C,H,W = local_batch.size()

    fets, pred_pts, quat, translation_og, axis_angles, euler_angles, rot_matrix_pred, scale_og = model_group['model'](local_batch)

    # print('Model features originally',fets.shape)

    'Computing mean and covariance for each frame'
    # with torch.no_grad():
    #     framewise_mean_features = torch.stack([torch.mean(rep_features[i], dim=0) for i in range(len(local_batch))])
    #     print('Framewise mean features', framewise_mean_features.shape)
    #     framewise_covaiance_features = torch.stack(
    #         [centered_cov_torch(rep_features[i] - framewise_mean_features[i]) for i in range(len(local_batch))]
    #     )
    #     print('Framewise covariance features', framewise_covaiance_features.shape)
    
    'pred points'
    pred_pts = pred_pts.view(B,3,3).permute(1,0,2)
    
    'Transform groundtruth points'
    local_scale = 1/local_scale
    local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(B, 1, 1)  #B,3,3
    local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
    local_translation = (local_translation).unsqueeze(-1)
    local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
    
    local_grid = F.affine_grid(local_rotation, (B,1,1,H,W)).squeeze(1)
    local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0]))

    'Euler Predictions'
    scale = scale_og.unsqueeze(-1)*torch.eye(3, dtype=scale_og.dtype, device=scale_og.device).repeat(B, 1, 1)  #B,3,3
    # print('Pred Scale Angles', scale.shape)
    # print('Euler Angles', euler_angles.shape)
    rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
    # print('Euler Matrix', rot.shape)
    pred_rotation = torch.einsum('bij,bjk->bik', scale, rot)
    # print('Pred Rotation', pred_rotation.shape)
    translation = (translation_og).unsqueeze(-1)
    # print('PredTranslation', translation.shape)
    pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
    # print('Pred Rotation Updated', pred_rotation.shape)
  
    my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
    euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

    'Axis Predictions'
    scale = scale_og.unsqueeze(-1)*torch.eye(3, dtype=scale_og.dtype, device=scale_og.device).repeat(B, 1, 1)  #B,3,3
    # print('Pred Scale Angles', scale.shape)
    # print('Axis Angles', axis_angles.shape)
    rot = axisAngletoRotation_torch_nerf(axis_angles)
    # print('Axis Matrix', rot.shape)
    pred_rotation = torch.einsum('bij,bjk->bik', scale, rot)
    # print('Pred Rotation', pred_rotation.shape)
    translation = (translation_og).unsqueeze(-1)
    # print('PredTranslation', translation.shape)
    pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
    # print('Pred Rotation Updated', pred_rotation.shape)
  
    my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
    axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

    'Rot Predictions'
    rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
    scale = scale_og.unsqueeze(-1)*torch.eye(3, dtype=scale_og.dtype, device=scale_og.device).repeat(B, 1, 1)  #B,3,3
    rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
    # print('Pred Rotation', rot_matrix_pred.shape)
    pred_rotation = torch.einsum('bij,bjk->bik', scale, rot_matrix_pred)
    # print('Pred Rotation w/ scale', pred_rotation.shape)
    translation = (translation_og).unsqueeze(-1)
    # print('PredTranslation', translation.shape)
    pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
    # print('Pred Rotation Updated', pred_rotation.shape)

    my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
    rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

    'Quat Predictions'
    quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    print('Quat', quat.shape)
    print(scale_og.shape, 'Scale')
    scale = scale_og.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    print('Scale', scale.shape)
    translation = translation_og.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    print('Translation', translation.shape)
    matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=scale_og.dtype, device=scale_og.device).unsqueeze(0).repeat(batch_size, 1, 1)
    print('Matrix Temp', matrix_temp.shape)
    grid_temp = F.affine_grid(matrix_temp, (batch_size,1,1,H,W)).squeeze(1)
    print('Grid Temp', grid_temp.shape)
    my_grid = quaternion_apply(quat, grid_temp)
    print('My Grid', my_grid.shape)
    my_grid = scale*my_grid+translation
    quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))
    'Loss'
    all_losses = []
    all_losses.append(model_group['criterion']['points'](pred_pts*(H/2), local_pts*(H/2)))
    all_losses.append(model_group['criterion']['points'](quat_pts*(H/2),local_pts*(H/2)))
    all_losses.append(model_group['criterion']['points'](rot_pts*(H/2),local_pts*(H/2)))
    all_losses.append(model_group['criterion']['points'](axis_pts*(H/2),local_pts*(H/2)))
    all_losses.append(model_group['criterion']['points'](euler_pts*(H/2),local_pts*(H/2)))

    multi_loss_layer = WeightedMultiTaskLoss(all_losses,scale_og.device)
    loss_points = multi_loss_layer.get_loss()
   
    print('Loss for pose with learned sigmas', loss_points)

    # loss_pose = model_group['criterion']['points'](quat, local_quat)+model_group['criterion']['points'](scale, local_scale)+model_group['criterion']['points'](translation, local_translation)
    
    loss = loss_points
    if train:
        loss.backward()
        model_group['optimizer'].step()
        model_group['optimizer'].zero_grad()
    
    
    model_group['loss']['overall'].append(loss.item())
    model_group['loss']['points'].append(loss_points.item()) 


def run_points(model_group, local_batch, local_y, local_scale, local_rotation, local_translation,  train=None):
    if train:
        model_group['model'] = model_group['model'].train()
        model_group['optimizer'].zero_grad()

        
    'predict'
    B,C,H,W = local_batch.size()

    fets, pred_pts, sigma, mu, vars = model_group['model'](local_batch)
  
    'pred points'

    landmark_pts = pred_pts[0]
    landmark_sigma = sigma[0]

    'Transform groundtruth points'
    local_scale = 1/local_scale
    local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(B, 1, 1)  #B,3,3
    local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
    local_translation = (local_translation).unsqueeze(-1)
    local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
    
    local_grid = F.affine_grid(local_rotation, (B,1,1,H,W)).squeeze(1)
    local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0]))

  
    loss_points = torch.mean(model_group['criterion']['points'](landmark_pts*(H/2), local_pts*(H/2), landmark_sigma*(H/2)))
    


    loss = loss_points
    if train:
        loss.backward()
        model_group['optimizer'].step()
        model_group['optimizer'].zero_grad()
    
    
    model_group['loss']['overall'].append(loss.item())
    model_group['loss']['points'].append(loss_points.item()) 
    
def run_quat(model_group, local_batch, local_y, local_scale, local_rotation, local_translation,  train=None):
    if train:
        model_group['model'] = model_group['model'].train()
        model_group['optimizer'].zero_grad()
        
    'predict'
    B,C,H,W = local_batch.size()
    _, scale, quat, translation,salpha = model_group['model'](local_batch)
    'Transform pred points - approach 1'
    quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    scale = scale.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=scale.dtype, device=scale.device).unsqueeze(0).repeat(B, 1, 1)
    grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)

    grid = quaternion_apply(quat, grid_temp)
    grid = scale*grid+translation

    'Transform pred points - approach 2'
    #jay
    # scale = scale.view(scale.shape[0],3,3).permute(1,0,2)
    
    # grid = torch.zeros((B, H, W, 3))
    # y1_pred = scale[0].detach().squeeze().cpu().numpy()
    # y2_pred = scale[1].detach().squeeze().cpu().numpy()
    # y3_pred = scale[2].detach().squeeze().cpu().numpy()
            
    # for i in range(B):

    #     matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=torch.float).unsqueeze(0).repeat(1, 1, 1)
    #     grid_temp = F.affine_grid(matrix_temp, (1,1,1,160,160)).squeeze().detach().cpu().numpy()
        
    #     sample_pt1 = grid_temp[0,0]
    #     sample_pt2 = grid_temp[0,-1]
    #     sample_pt3 = grid_temp[-1,0]
        
    #     sample = np.stack((sample_pt1,sample_pt2,sample_pt3), axis=0)
    #     target = np.stack((y1_pred[i],y2_pred[i],y3_pred[i]), axis=0)
        
    #     _, tform_pts, tform_parms = procrustes(target, sample, scaling=True, reflection=False)
        
    #     'Transform'

    #     rot = np.transpose(tform_parms['rotation'])
    #     quat = matrix_to_quaternion(torch.from_numpy(rot))
    #     grid_test = quaternion_apply(quat.unsqueeze(0).unsqueeze(0).repeat(160,160,1), torch.from_numpy(grid_temp))
    #     grid_test = grid_test.permute(2,0,1).cpu().numpy()
    #     grid_pred=grid_test
        
    #     grid_pred = grid_pred*tform_parms['scale']
    #     grid_pred[0,:,:] = grid_pred[0,:,:]+tform_parms['translation'][0]#*tform_parms['scale']
    #     grid_pred[1,:,:] = grid_pred[1,:,:]+tform_parms['translation'][1]#*tform_parms['scale']
    #     grid_pred[2,:,:] = grid_pred[2,:,:]+tform_parms['translation'][2]#*tform_parms['scale']

    #     grid_pred_torch = torch.from_numpy(grid_pred).to(dtype=scale.dtype, device=scale.device).permute(1,2,0)
    #     grid[i,:, :,:] = grid_pred_torch
    # grid = grid.to(scale.device)
    # grid.requires_grad_()
    # #jay 
    
    # print(grid.shape)


    pred_pts = torch.stack((grid[:,0,0],grid[:,0,-1],grid[:,-1,0],grid[:,-1,-1]))
    'Transform groundtruth points'
    local_scale = 1/local_scale
    local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(B, 1, 1)  #B,3,3
    local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
    local_translation = (local_translation).unsqueeze(-1)
    local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
    
    local_grid = F.affine_grid(local_rotation, (B,1,1,H,W)).squeeze(1)
    local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0],local_grid[:,-1,-1]))
    
    'Plot'
    if False:
        local_batch = local_batch.squeeze(1).detach().cpu().numpy()
        grid = grid.cpu()
        
        for i in range(1,B+1):
            img_slice = F.grid_sample(local_vol.unsqueeze(1), grid[i-1:i].unsqueeze(1))
            img_slice = img_slice.squeeze().detach().cpu().numpy()
            plt.subplot(5,10,i)
            plt.imshow(img_slice)
        plt.figure()
        for i in range(1,B+1):
            plt.subplot(5,10,i)
            plt.imshow(local_batch[i-1])
    
    'Loss'
    loss_points = model_group['criterion']['points'](pred_pts*(H/2), local_pts*(H/2))
    
    
    # loss_pose = model_group['criterion']['points'](quat, local_quat)+model_group['criterion']['points'](scale, local_scale)+model_group['criterion']['points'](translation, local_translation)
    
    loss = loss_points
    
    if train:
        loss.backward()
        model_group['optimizer'].step()
        model_group['optimizer'].zero_grad()
    
    
    model_group['loss']['overall'].append(loss.item())
    model_group['loss']['points'].append(loss_points.item())
    
def run(model_group, local_batch, local_y, local_scale, local_rotation, local_translation,  train=None):
    if train:
        model_group['model'] = model_group['model'].train()
        model_group['optimizer'].zero_grad()
        
    'predict'
    B,C,H,W = local_batch.size()
    _, scale, quat, translation, salpha = model_group['model'](local_batch)
    
    'Transform pred points'
    quat = F.normalize(quat, p=2, dim=-1)
    # quat = eulerAnglesToRotationMatrix_torch(quat.unsqueeze(1)).squeeze(1)
    matrix_rot = quaternion_to_matrix(quat)
    matrix_scale = scale.unsqueeze(-1)*torch.eye(3, dtype=scale.dtype, device=scale.device, requires_grad=False).repeat(B, 1, 1)  #B,3,3
    matrix_rot = torch.einsum('bij,bjk->bik', matrix_scale, matrix_rot)
    matrix_translation = (translation).unsqueeze(-1)#(translation/(H/2)).unsqueeze(-1)
    rotation_translation = torch.cat((matrix_rot, matrix_translation), dim=-1)
    
    grid = F.affine_grid(rotation_translation, (B,1,1,H,W)).squeeze(1)
    pred_pts = torch.stack((grid[:,0,0],grid[:,0,-1],grid[:,-1,0],grid[:,-1,-1]))
    
    'Transform groundtruth points'
    local_scale = 1/local_scale
    local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(B, 1, 1)  #B,3,3
    local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
    local_translation = (local_translation).unsqueeze(-1)
    local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
    
    local_grid = F.affine_grid(local_rotation, (B,1,1,H,W)).squeeze(1)
    local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0],local_grid[:,-1,-1]))

    
    'Loss' 
    loss_points = model_group['criterion']['points'](pred_pts*(H/2), local_pts*(H/2))
    
    # loss_pose = model_group['criterion']['points'](quat, local_quat)+model_group['criterion']['points'](scale, local_scale)+model_group['criterion']['points'](translation, local_translation)
    
    loss = loss_points
    
    
    if train:
        loss.backward()
        model_group['optimizer'].step()
        model_group['optimizer'].zero_grad()
    
    
    model_group['loss']['overall'].append(loss.item())
    model_group['loss']['points'].append(loss_points.item())
    


if __name__ ==  '__main__':

    dropoutrate = float(sys.argv[1])
    weeksage = int(sys.argv[2])

    torch.manual_seed(101)
    torch.cuda.manual_seed(101)
    np.random.seed(101)
    random.seed(101)

    'Parameters setting'
    batch_size = 49
    params_dataset =  {'sample_num':batch_size,#50
                      'mode':'training'}
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 0,
              'drop_last':True}     # for data generator
    num_classes = [3,3,3]   # xyz for 3 points
    learning_rate = 0.0001
    patience = 20
    momentum = 0.9
    loss_weight = 1.0      # more wieght for the center point than the two corners
    max_epochs = 5000
    paras_model = {'patience':patience,
                   'learning_rate':learning_rate,
                   }
    age_group = weeksage
    side = 'left'
    save_path = os.path.join(os.getcwd(), '/')#os.path.join(os.getcwd(),str(age_group)+'_'+side)


    img_path = 'train'.format(weeksage) #Please contact authors for this sensitive data#

    files_selected = glob.glob(os.path.join(img_path, '*.mat')) 
    
    

    num_files = len(files_selected)
    files_training = files_selected[0:int(0.7*num_files)]+files_selected[int(0.8*num_files):]
    files_validation = files_selected[int(0.7*num_files):int(0.8*num_files)]
    
    
    
    'CUDA for PyTorch'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    'Initialize the models'
    model1 = create_model(VGGROTGNLL(make_layers_instance_norm(), num_classes=num_classes, fc_size = 512, dropout=dropoutrate), 'vgg16', save_path, paras_model, dropoutrate, weeksage, relative=False)
    # model2 = create_model(ResNet18_instance(), 'resnet', save_path, paras_model, relative=False)
    # model1 = create_model(No_name(device=device), 'no_name', save_path, paras_model, relative=False)
    # model2 = create_model(Baseline_simple_vgg(make_layers_instance_norm(), num_classes=num_classes, fc_size = 512, dropout=0), 'Baseline_vgg', save_path, paras_model, relative=True)
    # model3 = create_model(Baseline_vgg(make_layers_instance_norm(), num_classes=num_classes, fc_size = 512, dropout=0), 'Baseline_vgg', save_path, paras_model, relative=True)
    # model2 = create_model(Proposed_vgg(make_layers_instance_norm(), num_classes=num_classes, fc_size = 512, dropout=0), 'Proposed_vgg', save_path, paras_model)
    # model3 = create_model(Transformer_vgg(make_layers_instance_norm(), num_classes=num_classes, fc_size = 512, dropout=0), 'Transformer_vgg', save_path, paras_model)
    models_list = [model1]
    parameter_values = model1.values()

    # Print the parameter values
    for param in parameter_values:
        print(param)

    
    'Log'
    writer = SummaryWriter(os.path.join(save_path, 'log'))

    'Early stop when loss plateu'
    def stop_early(best, current, count):
        stop = False
        if current<=best:
            count=0
        else:
            count+=1
            if count>=2*patience+1:
                stop=True

        return stop, count
    
    'evaluation metrics'
    count = 0
    
    'Start training and testing'
    for epoch in range(max_epochs):
#        if epoch<=10:
#            continue

        'Training'
        if epoch>0:
            shuffle(files_training)
        print(save_path)
        
        # #TRAINING SET SAVE FOR ALL MODELS SO THEY CAN BE COMPARED FAIRLY
        # training_set = Dataset_end_to_end(files_training, **params_dataset)
        # training_set.shuffle_list()
        # training_generator = data.DataLoader(training_set, **params)
        
        # 'Reference points'
        # ref_pts = training_set.reference_points()
        # ref_pts = torch.from_numpy(ref_pts).to(device=device, dtype=torch.float) #N,3
        
        
         
        # saved_info = {
        #     "local_batch": [], 
        #     "local_y": [], 
        #     "local_scale": [], 
        #     "local_rotation": [], 
        #     "local_translation": [], 
        #     "local_vol": [],
        #     "ref_pts": ref_pts
        # }

        # # Loop through the training_generator
        # for i, (local_batch, local_y, local_scale, local_rotation, local_translation, local_vol) in enumerate(training_generator):
        #     # Append the tensor values to the dictionaries
        #     saved_info["local_batch"].append(local_batch)
        #     saved_info["local_y"].append(local_y)
        #     saved_info["local_scale"].append(local_scale)
        #     saved_info["local_rotation"].append(local_rotation)
        #     saved_info["local_translation"].append(local_translation)
        #     saved_info["local_vol"].append(local_vol)

        #     # Your other calculations or processes here

        # # Convert lists of tensors to tensors
        # saved_info["local_batch"] = torch.stack(saved_info["local_batch"])
        # saved_info["local_y"] = torch.stack(saved_info["local_y"])
        # saved_info["local_scale"] = torch.stack(saved_info["local_scale"])
        # saved_info["local_rotation"] = torch.stack(saved_info["local_rotation"])
        # saved_info["local_translation"] = torch.stack(saved_info["local_translation"])
        # saved_info["local_vol"] = torch.stack(saved_info["local_vol"])

        # # Save the dictionaries
        # torch.save(saved_info, 'training_{}.pth'.format(batch_size))

    

        loaded_info = torch.load('training_{}.pth'.format(batch_size))

        # Access the saved tensors from the loaded dictionary
        local_batch_list = loaded_info["local_batch"]
        local_y_list = loaded_info["local_y"]
        local_scale_list = loaded_info["local_scale"]
        local_rotation_list = loaded_info["local_rotation"]
        local_translation_list = loaded_info["local_translation"]
        local_vol_list = loaded_info["local_vol"]

        # Simulate the behavior of the original training_generator loop
        for i in range(len(local_batch_list)):
            local_batch = local_batch_list[i]
            local_y = local_y_list[i]
            local_scale =local_scale_list[i]
            local_rotation = local_rotation_list[i]
            local_translation = local_translation_list[i]
            local_vol = local_vol_list[i]
  
            count+=1
            # Transfer to GPU
            local_batch = local_batch.to(device=device, dtype=torch.float).squeeze(0)
            local_y = torch.squeeze(local_y).to(device=device, dtype=torch.float)
            local_scale = local_scale.squeeze(0).to(device=device, dtype=torch.float)
            local_rotation = local_rotation.squeeze(0).to(device=device, dtype=torch.float)
            local_translation = local_translation.squeeze(0).to(device=device, dtype=torch.float)
            
            if epoch==0 and i==0:
                print(local_batch.size(), local_y.size())
            
            for mg in models_list:
                run_points(mg, local_batch, local_y, local_scale, local_rotation, local_translation, train=True)
                
                if epoch==0 :
                    print(mg['name'], '[%d, %5d]'%(epoch + 1, i + 1), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in mg['loss'].items()]))
                    
         
                    # print(mg['name'], '[%d, %5d] loss: %.3f y1: %.3f y2: %.3f y3: %.3f' %
                    #                   (epoch + 1, i + 1, np.mean(mg['loss']['overall']), np.mean(mg['loss']['y1']), np.mean(mg['loss']['y2']), np.mean(mg['loss']['y3'])))
        
            'Display training results'
            if (i+1) % 100 == 0:    # print every 100 mini-batches
                dict_loss = {}
                dict_lr = {}
                for mg in models_list:
                    for param_group in mg['optimizer'].param_groups:
                        current_lr = param_group['lr']
                    print(mg['name'], '[%d, %5d]'%(epoch + 1, i + 1), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in mg['loss'].items()]), 'learning rate: %.4e' %(current_lr))
                    # print(mg['name'], '[%d, %5d] loss: %.3f y1: %.3f y2: %.3f y3: %.3f learning rate: %.4e' %
                    #                   (epoch + 1, i + 1, np.mean(mg['loss']['overall']), np.mean(mg['loss']['y1']), np.mean(mg['loss']['y2']), np.mean(mg['loss']['y3']), current_lr))
                    dict_loss.update({mg['name']:np.mean(mg['loss']['overall'])})
                    dict_lr.update({mg['name']:current_lr})
                # write to log
                writer.add_scalars('training/total_loss', dict_loss, count)
                writer.add_scalars('training/lr', dict_lr, count)
               
        'After 1 epoch of training'
        dict_loss = {}
        dict_lr = {}
        for mg in models_list:
            for param_group in mg['optimizer'].param_groups:
                current_lr = param_group['lr']
            print(mg['name'], '[%d, %5d]'%(epoch + 1, i + 1), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in mg['loss'].items()]), 'learning rate: %.4e' %(current_lr))
            dict_loss.update({mg['name']:np.mean(mg['loss']['overall'])})
            dict_lr.update({mg['name']:current_lr})
            torch.save(mg['model'].state_dict(), mg['current_model'])
            for key in mg['loss']:
                mg['loss'].update({key:[]})
        writer.add_scalars('training/total_loss', dict_loss, count)
        writer.add_scalars('training/lr', dict_lr, count)
        
        
        'Validation'
        validation_set = Dataset_end_to_end(files_validation, **params_dataset)
        validation_generator = data.DataLoader(validation_set, **params)
        for mg in models_list:
            mg['optimizer'].zero_grad()
            mg['model'] = mg['model'].eval()
        
        with torch.set_grad_enabled(False):
            for i, (local_batch, local_y, local_scale, local_rotation, local_translation, local_vol) in enumerate(validation_generator):
                # Transfer to GPU
                local_batch = local_batch.to(device=device, dtype=torch.float).squeeze(0)
                local_y = torch.squeeze(local_y).to(device=device, dtype=torch.float)
                local_scale = local_scale.squeeze(0).to(device=device, dtype=torch.float)
                local_rotation = local_rotation.squeeze(0).to(device=device, dtype=torch.float)
                local_translation = local_translation.squeeze(0).to(device=device, dtype=torch.float)

                for mg in models_list:
                    run_points(mg, local_batch, local_y, local_scale, local_rotation, local_translation, train=False)

                
            'Save and display results after test'
            for mg in models_list:
                for param_group in mg['optimizer'].param_groups:
                    current_lr = param_group['lr']
                loss_save={}
                for key in mg['loss']:
                    loss_save.update({key:np.mean(mg['loss'][key])})
                loss_save.update({'lr': current_lr})
                # loss_save.update({'overall':np.mean(mg['loss']['overall']),
                #                   'y1':np.mean(mg['loss']['y1']),
                #                   'y2':np.mean(mg['loss']['y2']),
                #                   'y3':np.mean(mg['loss']['y3']),
                #                   'lr': current_lr})
                mg['loss_on_test'].update({epoch+1:loss_save})
                
                # Display every validation loss
                keys = [k for k in mg['loss_on_test']]
                keys.sort()
                for key in keys:
                    print(mg['name'], '(validation %d)'%(key), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in mg['loss_on_test'][key].items()]), mg['loss_on_test'][key]['lr'])
                    # print(mg['name'], '(validation %d) loss: %.3f y1: %.3f y2: %.3f y3: %.3f learning rate: %.4e' %
                    #                   (key, np.mean(mg['loss_on_test'][key]['overall']), np.mean(mg['loss_on_test'][key]['y1']), np.mean(mg['loss_on_test'][key]['y2']), np.mean(mg['loss_on_test'][key]['y3']), current_lr))
                    
                dict_loss.update({mg['name']:np.mean(mg['loss']['overall'])})
                dict_lr.update({mg['name']:current_lr})
            
            
                # Save the model if the loss is the lowest
                current_loss = np.mean(mg['loss']['overall'])
                if  current_loss < mg['best']:
                    mg['best'] =  current_loss
                    print("  **")
                    torch.save(mg['model'].state_dict(), mg['saved_model'])
                mg['scheduler'].step(torch.tensor([current_loss]).to(device=device, dtype=torch.float))
                # mg['scheduler'].step()
                
                # Stop early
                mg['stop'], mg['stop_count'] = stop_early(mg['best'], current_loss, mg['stop_count'])
                
                
            writer.add_scalars('validation/total_loss', dict_loss, epoch+1)
            writer.add_scalars('validation/lr', dict_lr, epoch+1)
            
            
        'Stop early'
        stop_overall = True
        for mg in models_list:
            stop_overall*=mg['stop']
        if stop_overall:
            break