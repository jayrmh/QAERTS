import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile, asksaveasfilename
import glob
import os
from PIL import Image, ImageTk
from tabulate import tabulate
import pydicom as dicom
import numpy as np
from torch.utils import data
import torch
from torch import nn
import torch.nn.functional as F
import random
from random import uniform
import scipy.io as sio
from skimage.transform import resize
import math
from scipy.interpolate import interpn
import torch.nn.functional as F
from models import *
from model_end_to_end_base import Baseline_simple_vgg_scaling, make_layers_instance_norm
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from dataset_end_to_end_vunc1 import Dataset_end_to_end, procrustes, rotation_matrix, EulerToRotation_Z, R_2vect, grid_translation, unit_vector
from geometry_updated import *
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import pickle
from torchsummary import summary

from skimage.util import random_noise
torch.cuda.empty_cache()

import sys
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import gridspec

from epi_models_utils import *

plt.style.use('ggplot')


def rgb2gray(rgb):

    # r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = rgb[...,0]

    return gray


def box_plot_metrics(metric, keys_to_plot, metric_name):
        # Mapping colors to keys
        color_mapping = {
        'DEQAERTS': 'darkslategray',
        'MCD_0.1': 'skyblue',
        'MCD_0.25': 'lightcoral',
        'QAERTS': 'lightgreen',
        'EDL': 'gold',
        'DE': 'sienna',
        'Base': 'mediumorchid',
        'MVE': 'darkcyan'}
        
        plt.figure(figsize=(10, 6))

        for idx, key in enumerate(keys_to_plot, start=1):
            data = metric.get(key, [])  # Get the data for the key
            if data:
                data = np.array(data).reshape(-1)
                print('Data shape', data.shape)
                # Plotting box plots for the data
                box = plt.boxplot(data, positions=[idx], widths=0.6, patch_artist=True, showmeans=True, showfliers=False)

                # Change boxplot color
                for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
                    plt.setp(box[element], color='black', alpha=0.5)  # Adjust transparency level here
                for patch in box['boxes']:
                    patch.set(facecolor=color_mapping[key], alpha=0.4)  # Adjust transparency level here


                # Scatter plot for individual means
                jitter = np.random.normal(0, 0.08, size=len(data))  # Generate jitter for x-axis
                scatter = plt.scatter([idx] * len(data) + jitter, data, alpha=0.9, color=color_mapping[key])


        plt.xticks(range(1, len(keys_to_plot) + 1), [key for key in keys_to_plot], fontsize = 14)# Set x-axis labels# Set x-axis labels
        plt.xlabel('Models', fontsize = 20)
        plt.ylabel('{}'.format(metric_name),fontsize = 20)
        plt.title('Boxplots with Individual Means for Specific Keys in {} Dictionary'.format(metric_name))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"/{save_folder_update}/Figures/{metric_name}_boxplot.png")
        plt.close()
    


def getRotAngle(P, Q):
    # Calculate the dot product of P and Q
    R = torch.matmul(P, torch.transpose(Q, 0, 1))
  
    # Calculate the trace of R
    trace_R = torch.trace(R)

    print('Trace of R', trace_R)

    # Calculate the cosine of the angle
    cos_theta = (trace_R - 1) / 2

    print('Cosine of the angle', cos_theta)

    # Calculate the angle in radians
    angle_rad = torch.acos(cos_theta)

    print('Angle in radians', angle_rad)

    # Convert the angle from radians to degrees
    angle_deg = angle_rad * 180 / math.pi

    print('Angle in degrees', angle_deg)


    return angle_deg

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data + 1e-7)


def normalized_cross_correlation(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


def extract_ensemble_predictions(pred_pts, atlas):
     
    predicted_grids = []
    predicted_slices = []
    predicted_procrustes = []
    
     
    for j in range(pred_pts.shape[0]):
        
        y1_pred = pred_pts[j][0].detach().squeeze().cpu().numpy()
        y2_pred = pred_pts[j][1].detach().squeeze().cpu().numpy()
        y3_pred = pred_pts[j][2].detach().squeeze().cpu().numpy()

        ensemble_grid = []
        ensemble_slice = []
        ensemble_proc = []

        for i in range(y1_pred.shape[0]):
            grid_pred, atlas_pred, scale, quat, translation, proc_pred = extract_prediction(y1_pred[i], y2_pred[i], y3_pred[i], atlas)

            ensemble_grid.append(grid_pred)
            ensemble_slice.append(atlas_pred)
            ensemble_proc.append(proc_pred)
    
        predicted_grids.append(np.stack(ensemble_grid))
        predicted_slices.append(np.stack(ensemble_slice))
        predicted_procrustes.append(np.stack(ensemble_proc))
    
    return np.stack(predicted_grids), np.stack(predicted_slices), np.stack(predicted_procrustes)




def extract_prediction(y1_pred, y2_pred, y3_pred, vol):
    matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=torch.float).unsqueeze(0).repeat(1, 1, 1)
    grid_temp = F.affine_grid(matrix_temp, (1,1,1,160,160), align_corners=False).squeeze().detach().cpu().numpy()
    
    sample_pt1 = grid_temp[0,0]
    sample_pt2 = grid_temp[0,-1]
    sample_pt3 = grid_temp[-1,0]

    sample = np.stack((sample_pt1,sample_pt2,sample_pt3), axis=0)
    target = np.stack((y1_pred,y2_pred,y3_pred), axis=0)
    
    d, tform_pts, tform_parms = procrustes(target, sample, scaling=True, reflection=False)
    
    rotation = tform_parms['rotation']
    scale = tform_parms['scale']
    translation = tform_parms['translation']
    
    rot = np.transpose(rotation)
    quat = matrix_to_quaternion(torch.from_numpy(rot))
        

    'Transform'
    grid_pred = quaternion_apply(quat.unsqueeze(0).unsqueeze(0).repeat(160,160,1), torch.from_numpy(grid_temp))
    grid_pred = grid_pred.permute(2,0,1).cpu().numpy()
    
    grid_pred = grid_pred*scale
    grid_pred[0,:,:] = grid_pred[0,:,:]+translation[0]#*scale
    grid_pred[1,:,:] = grid_pred[1,:,:]+translation[1]#*scale
    grid_pred[2,:,:] = grid_pred[2,:,:]+translation[2]#*scale
    # print(tform_pts)
    # print(grid_pred[:,0,0])
    grid_pred_torch = torch.from_numpy(grid_pred).to(dtype=vol.dtype, device=vol.device).permute(1,2,0).unsqueeze(0).unsqueeze(0)
    
    'Sample'
    slice_pred = F.grid_sample(vol.unsqueeze(0).unsqueeze(0), grid_pred_torch, align_corners=False)
    slice_pred = slice_pred.squeeze().detach().cpu().numpy()
    
    
    
    return grid_pred, slice_pred, scale, quat.cpu().numpy(), translation, d



# def extract_prediction(y1_pred, y2_pred, y3_pred, vol):
#     matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=torch.float).unsqueeze(0).repeat(1, 1, 1)
#     grid_temp = F.affine_grid(matrix_temp, (1,1,1,160,160)).squeeze().detach().cpu().numpy()
    
#     sample_pt1 = grid_temp[0,0]
#     sample_pt2 = grid_temp[0,-1]
#     sample_pt3 = grid_temp[-1,0]
    
#     sample = np.stack((sample_pt1,sample_pt2,sample_pt3), axis=0)
#     target = np.stack((y1_pred,y2_pred,y3_pred), axis=0)
    
#     proc_d, tform_pts, tform_parms = procrustes(target, sample, scaling=True, reflection=False)
    
#     'Transform'
#     grid_pred = np.einsum('mni, ij -> jmn', grid_temp, tform_parms['rotation'])
#     # grid_pred = np.einsum('ij, mnj -> imn', tform_parms['rotation'], grid_temp)
    
#     # rot = np.transpose(tform_parms['rotation'])
#     # grid_test = np.einsum('ij, mnj -> imn', rot, grid_temp)
#     # print(np.sum(grid_test-grid_pred))
    
#     rot = np.transpose(tform_parms['rotation'])
#     quat = matrix_to_quaternion(torch.from_numpy(rot))

#     grid_test = quaternion_apply(quat.unsqueeze(0).unsqueeze(0).repeat(160,160,1), torch.from_numpy(grid_temp))

#     grid_test = grid_test.permute(2,0,1).cpu().numpy()
#     # print((grid_test-grid_pred).max(), (grid_test-grid_pred).min())
#     grid_pred=grid_test

    
#     grid_pred = grid_pred*tform_parms['scale']

#     grid_pred[0,:,:] = grid_pred[0,:,:]+tform_parms['translation'][0]#*tform_parms['scale']
#     grid_pred[1,:,:] = grid_pred[1,:,:]+tform_parms['translation'][1]#*tform_parms['scale']
#     grid_pred[2,:,:] = grid_pred[2,:,:]+tform_parms['translation'][2]#*tform_parms['scale']
#     # print(tform_pts)
#     # print(grid_pred[:,0,0])
#     grid_pred_torch = torch.from_numpy(grid_pred).to(dtype=vol.dtype, device=vol.device).permute(1,2,0).unsqueeze(0).unsqueeze(0)
    
#     'Sample'
#     slice_pred = F.grid_sample(vol.unsqueeze(0).unsqueeze(0), grid_pred_torch)
#     slice_pred = slice_pred.squeeze().detach().cpu().numpy()
    
#     return grid_pred, slice_pred, tform_parms['scale'], quat.cpu().numpy(), tform_parms['translation']
def matrix_distance(pts):
    reference = np.zeros((pts.shape[0],3))  #N, 3
        
    return np.sqrt(np.sum((pts-reference)**2, axis=-1, keepdims=False))
    

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_sampled_images(pred_pts, local_pts, atlas):
        

        print(pred_pts.shape, 'pred_pts')
        print(local_pts.shape, 'local_pts')

        y1_pred = pred_pts[0].detach().squeeze().cpu().numpy()
        y2_pred = pred_pts[1].detach().squeeze().cpu().numpy()
        y3_pred = pred_pts[2].detach().squeeze().cpu().numpy()

        y1_real = local_pts[0].detach().squeeze().cpu().numpy()
        y2_real = local_pts[1].detach().squeeze().cpu().numpy()
        y3_real = local_pts[2].detach().squeeze().cpu().numpy()

        'Generate grid and atlas slice'
        slice_atlas = []
        slice_atlas_real = []
        ssim_diff = []
        ncc = []
        plane_angles = [] 
        gri_pred = []
        gri_real = []
        
        trans_real = []
        trans_pred = []
        scale_real = []
        scale_pred = []
        quat_real = []
        quat_pred = []

        procrustes_d = []
        mid_pt = []

        ed_metric = []
        mse_planes_metric = []
 
        for i in range(y1_real.shape[0]):
            grid_pred, atlas_pred, scale, quat, translation, proc_pred = extract_prediction(y1_pred[i], y2_pred[i], y3_pred[i], atlas)
            slice_atlas.append(atlas_pred)
            trans_pred.append(torch.tensor(translation))
            scale_pred.append(torch.tensor(scale).unsqueeze(0))
            quat_pred.append(torch.tensor(quat))
            procrustes_d.append(proc_pred)
            mid_pt.append(grid_pred[:,80,80])

            grid_real, atlas_real, scale_r, quat_r, translation_r, proc_real = extract_prediction(y1_real[i], y2_real[i], y3_real[i], atlas)
            scale_real.append(torch.tensor(scale_r).unsqueeze(0))
            trans_real.append(torch.tensor(translation_r))
            quat_real.append(torch.tensor(quat_r))
            slice_atlas_real.append(atlas_real)
            score = calculate_ssim_groundtruth(atlas_real, atlas_pred)
            ssim_diff.append(score)
            ncc.append(normalized_cross_correlation(atlas_real, atlas_pred))
            plane_angles.append(calculate_plane_angle_planes(grid_pred, grid_real))
            gri_pred.append(grid_pred)
            gri_real.append(grid_real)

            ed_metric.append(calculate_euclidean_distance_planes(torch.from_numpy(grid_pred), torch.from_numpy(grid_real)).numpy())
            mse_planes_metric.append(calculate_mse_planes(torch.from_numpy(grid_pred), torch.from_numpy(grid_real)).numpy())

        return np.stack(slice_atlas), np.stack(slice_atlas_real), np.stack(ssim_diff), np.stack(ncc), np.stack(plane_angles), np.stack(gri_pred), np.stack(gri_real), torch.stack(trans_pred), torch.stack(scale_pred), torch.stack(quat_pred), torch.stack(trans_real), torch.stack(scale_real), torch.stack(quat_real), np.stack(ed_metric), np.stack(mse_planes_metric), np.stack(procrustes_d), matrix_distance(np.array(mid_pt))


def plot_ensembles(slices, grids, epi_mean, epi_plane_mean, name, method, patientID=0):
    num_images = slices.shape[1]
    fig = plt.figure(figsize=(20, 2*num_images))
    gs = gridspec.GridSpec(num_images,8,figure=fig, width_ratios=[1,1,1,1,1,1,1,2])#hspace=0.2, wspace=0.2

    mean_image = np.mean(slices, axis=0)

    epi_mean =  epi_mean.detach().cpu().numpy()
    epi_plane_mean =  epi_plane_mean.detach().cpu().numpy()

    epi_mean = (epi_mean  - np.min(epi_mean)) / (np.max(epi_mean) - np.min(epi_mean))
    epi_plane_mean = (epi_plane_mean  - np.min(epi_plane_mean)) / (np.max(epi_plane_mean) - np.min(epi_plane_mean))

    idx = 0
    for i in range(num_images):
        
        for ij in range(slices.shape[0]):
            ax_img = plt.subplot(gs[i, ij])
            ax_img.imshow(slices[ij][i], cmap="gray")
            ax_img.set_title(f"Prediction {ij}")
            ax_img.grid(False)
            idx = ij

        
        ax_img = plt.subplot(gs[i, idx+1])
        ax_img.imshow(mean_image[i], cmap="gray")
        ax_img.set_title(f"Mean Prediction")
        ax_img.grid(False)

        ax_bar = plt.subplot(gs[i, idx+2])
        bar_positions = np.arange(2)
        ax_bar.bar(bar_positions, [epi_mean[i], epi_plane_mean[i]], tick_label=['Ref↓','Planes↓'])
        ax_bar.set_ylim(0, 1)

        ax_img_3d = plt.subplot(gs[i, idx+3], projection='3d')
        ax_img_3d.set_xlim((0,160))
        ax_img_3d.set_ylim((0,160))
        ax_img_3d.set_zlim3d((0,160))
        ax_img_3d.set_ylabel('voxels')
        ax_img_3d.set_xlabel('voxels')
        ax_img_3d.set_zlabel('voxels')
        ax_img_3d.set_title(f"Prediction Variability {i}")

        colors=["red", "blue", "purple", "yellow", "orange"]

        for ik in range(slices.shape[0]):    

            point_grid_pred = grids[ik][i]*80 +80 
            ax_img_3d.plot_surface(point_grid_pred[0,:,:], point_grid_pred[1,:,:], point_grid_pred[2,:,:],alpha=0.1, color=colors[ik])    
            ax_img_3d.scatter(point_grid_pred[0,0,0],point_grid_pred[1,0,0], point_grid_pred[2,0,0],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,159,159],point_grid_pred[1,159,159], point_grid_pred[2,159,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,0,159],point_grid_pred[1,0,159], point_grid_pred[2,0,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,159,0],point_grid_pred[1,159,0], point_grid_pred[2,159,0],s=10, c='black')
        

       
    plt.tight_layout()
    plt.savefig(f"/{save_folder_update}/Figures/{patientID}_{method}_{name}_ensemble.png")           
    plt.close()



def displaygroundtruth(local_pts, atlas, name="", method="", mode=False):

    if mode:
            print(local_pts.shape, 'local_pts')

            y1_real = local_pts[0].detach().squeeze().cpu().numpy()
            y2_real = local_pts[1].detach().squeeze().cpu().numpy()
            y3_real = local_pts[2].detach().squeeze().cpu().numpy()

            'Generate grid and atlas slice'

            slice_atlas_real = []
            grid_real_slice = []


            for i in range(y1_real.shape[0]):
            
            
                grid_real, atlas_real, scale_real, quat_real, translation_real, pro_d  = extract_prediction(y1_real[i], y2_real[i], y3_real[i], atlas)
                
                slice_atlas_real.append(atlas_real)
                grid_real_slice.append(grid_real)

            slice_atlas_real = np.stack(slice_atlas_real)
            grid_real_slice = np.stack(grid_real_slice)

            sampling_xrange = np.arange(-80,80)
            sampling_yrange = np.arange(-80,80)
            X, Y = np.meshgrid(sampling_xrange, sampling_yrange)
            print('X-shape', X.shape)
            grid = np.dstack([X, Y])
            grid = np.concatenate((grid,np.zeros([160,160,1])),axis=-1)

            grid_rot = np.einsum('ji, mni -> jmn', np.identity(3), grid)
            original_grid, _ = grid_translation(grid_rot, 0, 0, 0)

            original_grid = original_grid+80
    

            fig = plt.figure(figsize=(20, 4*len(slice_atlas_real)))
            gs = gridspec.GridSpec(len(slice_atlas_real), 2, figure=fig, width_ratios=[1,1])#hspace=0.2, wspace=0.2

            for i in range(len(slice_atlas_real)): 
                ax_img = plt.subplot(gs[i, 0])
                ax_img.imshow(slice_atlas_real[i], cmap="gray")
                ax_img.set_title(f"Predicted {i}")
                ax_img.grid(False)

                point_grid_pred = grid_real_slice[i]*80+80 

                ax_img_3d = plt.subplot(gs[i, 1], projection='3d')
                ax_img_3d.set_xlim((0,160))
                ax_img_3d.set_ylim((0,160))
                ax_img_3d.set_zlim3d((0,160))
                ax_img_3d.set_ylabel('voxels')
                ax_img_3d.set_xlabel('voxels')
                ax_img_3d.set_zlabel('voxels')
                ax_img_3d.set_title(f"Relative to Midplane {i}")

                ax_img_3d.plot_surface(original_grid[0,:,:], original_grid[1,:,:], original_grid[2,:,:],alpha=.5)
                ax_img_3d.scatter(original_grid[0,80,80],original_grid[1,80,80], original_grid[2,80,80])
                ax_img_3d.scatter(original_grid[0,0,0],original_grid[1,0,0], original_grid[2,0,0])
                ax_img_3d.scatter(original_grid[0,159,159],original_grid[1,159,159], original_grid[2,159,159])
                ax_img_3d.scatter(original_grid[0,0,159],original_grid[1,0,159], original_grid[2,0,159])
                ax_img_3d.scatter(original_grid[0,159,0],original_grid[1,159,0], original_grid[2,159,0])
                ax_img_3d.plot_surface(point_grid_pred[0,:,:], point_grid_pred[1,:,:], point_grid_pred[2,:,:],alpha=.5)
                                                   
    plt.tight_layout()
    plt.savefig(f"/{save_folder_update}/Figures/{method}_{name}_ground_truth.png")
    plt.close()
            
def plot_all_images_and_stats(pred_poses, ground_truth, mse_loss, epi_loss, atlas, name="MCD", standardize=True, method="ED", quality=0, original=[], plot=False, patientID='', std_points=[], brain_boundaries_old=[],  brain_boundaries_new=[], planes=True, videos=False):
    
    lossflag = True
    if epi_loss is None:
        epi_std = torch.zeros(128)
        epi_loss = torch.zeros(128)
        lossflag = False
    
    epi_std = epi_loss

    print('pred_poses', pred_poses.shape)
    
    original = original.detach().squeeze(1).cpu().numpy()

    if videos:
        slice_atlas, slice_atlas_real, ssim_diff, ncc, pa_grid, pred_grid, gt_grid, t_pred, s_pred, q_pred, t_real, s_real, q_real, ed_metrics, mse_planes_metrics, procrustes_d, mid_points = get_sampled_images(pred_poses, pred_poses, atlas)
    else:
        slice_atlas, slice_atlas_real, ssim_diff, ncc, pa_grid, pred_grid, gt_grid, t_pred, s_pred, q_pred, t_real, s_real, q_real, ed_metrics, mse_planes_metrics, procrustes_d, mid_points = get_sampled_images(pred_poses, ground_truth, atlas)
        print('MSE_loss: ', mse_loss.shape, ';', 'Epi_loss: ', epi_std.shape, ';', 'SSIM:', ssim_diff.shape, ';' ,'NCC: ', ncc.shape, ';', 'PA: ', pa_grid.shape, ';', 'ED: ', ed_metrics.shape, ';', 'Procrustes: ', procrustes_d.shape)
        if plot == False:
            return mse_loss.detach().cpu().numpy(), ssim_diff, ncc, pa_grid, ed_metrics, epi_std.detach().cpu().numpy()

        mse_loss = mse_loss.detach().cpu().numpy()
        if torch.is_tensor(epi_std):
            epi_std = epi_std.detach().cpu().numpy()

        'Tabulating the results'
        mean_std_data = [
        ["MSE_loss", np.mean(mse_loss), np.std(mse_loss)],
        ["MSE Planes", np.mean(mse_planes_metrics), np.std(mse_planes_metrics)],
        ["Epi_loss", np.mean(epi_std), np.std(epi_std)],
        ["SSIM", np.mean(ssim_diff), np.std(ssim_diff)],
        ["NCC", np.mean(ncc), np.std(ncc)],
        ["PA", np.mean(pa_grid), np.std(pa_grid)],
        ["ED", np.mean(ed_metrics), np.std(ed_metrics)],
    ]

        # Create a table with headers
        table = tabulate(mean_std_data, headers=["Name", "Mean", "Std"], tablefmt="pretty")
        print(table)

        if standardize == True: 
            # Normalize mse_loss
            mse_loss = (mse_loss - np.min(mse_loss)) / (np.max(mse_loss) - np.min(mse_loss))
            mse_planes_metrics = (mse_planes_metrics - np.min(mse_planes_metrics)) / (np.max(mse_planes_metrics) - np.min(mse_planes_metrics))
            # # Normalize epi_loss
            # epi_loss = (epi_loss - np.min(epi_loss)) / (np.max(epi_loss) - np.min(epi_loss))
            # Normalize epi_std
            epi_std = (epi_std - np.min(epi_std)) / (np.max(epi_std) - np.min(epi_std))
            # Normalize ed_metrics
            ed_metrics = (ed_metrics - np.min(ed_metrics)) / (np.max(ed_metrics) - np.min(ed_metrics))
            # Normalize ssim_diff
            ssim_diff = (ssim_diff - np.min(ssim_diff)) / (np.max(ssim_diff) - np.min(ssim_diff))
            # Normalize ncc
            ncc = (ncc - np.min(ncc)) / (np.max(ncc) - np.min(ncc))
            # Normalize pa_grid
            pa_grid = (pa_grid - np.min(pa_grid)) / (np.max(pa_grid) - np.min(pa_grid))

        # Reshape y1_mean, y2_mean, and y3_mean
        y1_mean = np.reshape(pred_poses[0].detach().squeeze().cpu().numpy(), (3, -1))*80 
        y2_mean = np.reshape(pred_poses[1].detach().squeeze().cpu().numpy(), (3, -1))*80
        y3_mean = np.reshape(pred_poses[2].detach().squeeze().cpu().numpy(), (3, -1))*80 

        # Reshape y1_gt, y2_gt, and y3_gt
        y1_gt = np.reshape(ground_truth[0].detach().squeeze().cpu().numpy(), (3, -1))*80 
        y2_gt = np.reshape(ground_truth[1].detach().squeeze().cpu().numpy(), (3, -1))*80  
        y3_gt = np.reshape(ground_truth[2].detach().squeeze().cpu().numpy(), (3, -1))*80 

        errorbar = True

        if std_points is not None:
            # Reshape y1_std, y2_std, and y3_std
            y1_std = np.reshape(std_points[0].detach().squeeze().cpu().numpy(), (3, -1))*80
            y2_std = np.reshape(std_points[1].detach().squeeze().cpu().numpy(), (3, -1))*80 
            y3_std = np.reshape(std_points[2].detach().squeeze().cpu().numpy(), (3, -1))*80 
        else:
            errorbar = False

        if planes:
            mse_loss = mse_planes_metrics

    num_images = len(slice_atlas)
    fig = plt.figure(figsize=(20, 4*num_images))
    gs = gridspec.GridSpec(num_images, 6, figure=fig, width_ratios=[1,1,1,3,3,2])#hspace=0.2, wspace=0.2

    for i in range(num_images):
        ax_img = plt.subplot(gs[i, 0])
        ax_img.imshow(slice_atlas[i], cmap="gray")
        ax_img.set_title(f"Predicted {i}")
        ax_img.grid(False)

        ax_img = plt.subplot(gs[i, 1])
        ax_img.imshow(slice_atlas_real[i], cmap="gray")
        ax_img.set_title(f"Ground Truth {i}")
        ax_img.grid(False)

        ax_img = plt.subplot(gs[i, 2])
        ax_img.imshow(original[i], cmap="gray")
        ax_img.set_title(f"Input Image {i}")
        ax_img.grid(False)


        if videos: 
            ax_img_3d = plt.subplot(gs[i, 3], projection='3d')
            ax_img_3d.set_xlim((0,160))
            ax_img_3d.set_ylim((0,160))
            ax_img_3d.set_zlim3d((0,160))
            ax_img_3d.set_ylabel('voxels')
            ax_img_3d.set_xlabel('voxels')
            ax_img_3d.set_zlabel('voxels')
            ax_img_3d.set_title(f"Predicted vs Real Location {i}")

            point_grid_pred = pred_grid[i]*80 +80 

            # print(point_grid_pred.shape, 'point_grid_pred')
    
            ax_img_3d.plot_surface(point_grid_pred[0,:,:], point_grid_pred[1,:,:], point_grid_pred[2,:,:],alpha=0.3, color="red")    
            ax_img_3d.scatter(point_grid_pred[0,0,0],point_grid_pred[1,0,0], point_grid_pred[2,0,0],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,159,159],point_grid_pred[1,159,159], point_grid_pred[2,159,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,0,159],point_grid_pred[1,0,159], point_grid_pred[2,0,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,159,0],point_grid_pred[1,159,0], point_grid_pred[2,159,0],s=10, c='black')

        else:
        
            ax_bar = plt.subplot(gs[i, 3])
            bar_positions = np.arange(5)
            ax_bar.bar(bar_positions, [mse_loss[i], ed_metrics[i], pa_grid[i], ncc[i], ssim_diff[i]], tick_label=['MSE↓', 'ED↓', 'PA↓', 'NCC↑', 'SSIM↑'])
            ax_bar.set_ylabel('Differences')
            ax_bar.set_title(f"Image {i} Stats")

            # Set y-axis ticks and limits
            if standardize == True:
                ax_bar.set_yticks(np.arange(0, 1.1, 0.1))
                ax_bar.set_ylim(0, 1)
            else:
                ax_bar.set_yticks(np.arange(0, 350, 50))
                ax_bar.set_ylim(0, 300)

            ax_img_3d = plt.subplot(gs[i, 4], projection='3d')
            ax_img_3d.set_xlim((0,160))
            ax_img_3d.set_ylim((0,160))
            ax_img_3d.set_zlim3d((0,160))
            ax_img_3d.set_ylabel('voxels')
            ax_img_3d.set_xlabel('voxels')
            ax_img_3d.set_zlabel('voxels')
            ax_img_3d.set_title(f"Predicted vs Real Location {i}")

            point_grid_real = gt_grid[i]*80 +80 
            point_grid_pred = pred_grid[i]*80 +80 

            # print(point_grid_pred.shape, 'point_grid_pred')
            
        

            ax_img_3d.plot_surface(point_grid_real[0,:,:], point_grid_real[1,:,:], point_grid_real[2,:,:],alpha=0.3, color="green")    
            ax_img_3d.scatter(point_grid_real[0,0,0],point_grid_real[1,0,0], point_grid_real[2,0,0],s=10, c='black')
            ax_img_3d.scatter(point_grid_real[0,159,159],point_grid_real[1,159,159], point_grid_real[2,159,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_real[0,0,159],point_grid_real[1,0,159], point_grid_real[2,0,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_real[0,159,0],point_grid_real[1,159,0], point_grid_real[2,159,0],s=10, c='black')

            ax_img_3d.plot_surface(point_grid_pred[0,:,:], point_grid_pred[1,:,:], point_grid_pred[2,:,:],alpha=0.3, color="red")    
            ax_img_3d.scatter(point_grid_pred[0,0,0],point_grid_pred[1,0,0], point_grid_pred[2,0,0],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,159,159],point_grid_pred[1,159,159], point_grid_pred[2,159,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,0,159],point_grid_pred[1,0,159], point_grid_pred[2,0,159],s=10, c='black')
            ax_img_3d.scatter(point_grid_pred[0,159,0],point_grid_pred[1,159,0], point_grid_pred[2,159,0],s=10, c='black')
            
            # ax_img_3d.scatter(brain_boundaries_old[0], brain_boundaries_old[1], brain_boundaries_old[2], s=0.5, alpha=0.5, c='lightgray')
            # ax_img_3d.scatter(brain_boundaries_new[0], brain_boundaries_new[1], brain_boundaries_new[2], s=0.5, alpha=0.5, c='lightgray')

            #hugo approach

            # point_grid_real = gt_grid[i]*80+80 
            # point_grid_pred = pred_grid[i]*80+80 

            # ax_img_3d.plot_surface(point_grid_real[...,0], point_grid_real[...,1], point_grid_real[...,2],alpha=0.2, color="green")    
            # # plt3d.scatter(original_grid[0,80,80],original_grid[1,80,80], original_grid[2,80,80],s=,c=black)
            # # ax_img_3d.scatter(point_grid_real[0,0,0],point_grid_real[1,0,0], point_grid_real[2,0,0],s=10, c='black')
            # # ax_img_3d.scatter(point_grid_real[0,159,159],point_grid_real[1,159,159], point_grid_real[2,159,159],s=10, c='black')
            # # ax_img_3d.scatter(point_grid_real[0,0,159],point_grid_real[1,0,159], point_grid_real[2,0,159],s=10, c='black')
            # # ax_img_3d.scatter(point_grid_real[0,159,0],point_grid_real[1,159,0], point_grid_real[2,159,0],s=10, c='black')

            # ax_img_3d.plot_surface(point_grid_pred[...,0], point_grid_pred[...,1], point_grid_pred[...,2],alpha=0.2, color="red")    
            # # ax_img_3d.scatter(sampling_grid[i][0,80,80],sampling_grid[i][1,80,80], sampling_grid[i][2,80,80], s=5)
            # # ax_img_3d.scatter(point_grid_pred[0,0,0],point_grid_pred[1,0,0], point_grid_pred[2,0,0],s=10, c='black')
            # # ax_img_3d.scatter(point_grid_pred[0,159,159],point_grid_pred[1,159,159], point_grid_pred[2,159,159],s=10, c='black')
            # # ax_img_3d.scatter(point_grid_pred[0,0,159],point_grid_pred[1,0,159], point_grid_pred[2,0,159],s=10, c='black')
            # # ax_img_3d.scatter(point_grid_pred[0,159,0],point_grid_pred[1,159,0], point_grid_pred[2,159,0],s=10, c='black')

            # ax_img_3d.scatter(brain_boundaries_old[0], brain_boundaries_old[1], brain_boundaries_old[2], s=0.5, alpha=0.5, c='lightgray')
            # ax_img_3d.scatter(brain_boundaries_new[0], brain_boundaries_new[1], brain_boundaries_new[2], s=0.5, alpha=0.5, c='lightgray')
    
            
            ax_img = plt.subplot(gs[i, 5])
            ax_img.set_title(f"Variability {i}")
            ax_img.grid(False)

            # # Transpose the arrays to have the same shape
            y1_mean_transposed = y1_mean.T
            y2_mean_transposed = y2_mean.T
            y3_mean_transposed = y3_mean.T

            y1_gt_transposed = y1_gt.T
            y2_gt_transposed = y2_gt.T
            y3_gt_transposed = y3_gt.T

            if std_points is not None:
                y1_std_transposed = y1_std.T
                y2_std_transposed = y2_std.T
                y3_std_transposed = y3_std.T
            else:
                errorbar = False

            bar_positions = np.array([0, 0.4, 0.8])
            bar_width = 0.2  # Width of each bar
            bar_group_width = bar_width # Width of each group

            # Create bars for Group 1 (y1)
            ax_img.bar(bar_positions, y1_mean_transposed[i], bar_width, label="Mean", color="#d2d2d2", yerr=(y1_std_transposed[i] if errorbar else None), capsize=5,  edgecolor='black')
            ax_img.bar(bar_positions + bar_group_width, y1_gt_transposed[i], bar_width, label="GT", color="#000000", alpha=0.5,  edgecolor='black')
            
            # Create bars for Group 2 (y2)
            bar_positions = np.array([1.2, 1.6, 2.0])  # Add 1.2 to create a separation between groups
            ax_img.bar(bar_positions, y2_mean_transposed[i], bar_width, color="#d2d2d2", yerr=(y2_std_transposed[i] if errorbar else None), capsize=5,  edgecolor='black')
            ax_img.bar(bar_positions + bar_group_width, y2_gt_transposed[i], bar_width,color="#000000", alpha=0.5, edgecolor='black')

            # Create bars for Group 3 (y3)
            bar_positions = np.array([2.4, 2.8, 3.2])  # Add 1.2 to create a separation between groups
            ax_img.bar(bar_positions, y3_mean_transposed[i], bar_width, color="#d2d2d2", yerr=(y3_std_transposed[i] if errorbar else None), capsize=5, edgecolor='black')
            ax_img.bar(bar_positions + bar_group_width, y3_gt_transposed[i], bar_width,color="#000000", alpha=0.5, edgecolor='black')
            # Set x-axis labels and legend
            ax_img.set_xticks([0.75, 1.75, 2.75])
            ax_img.axhline(0, color='black', linewidth=0.5)
            ax_img.set_xticklabels(['Ref 1', 'Ref 2', 'Ref 3'])
            ax_img.legend(loc='upper right', prop={'size': 5})
            

                                           
    plt.tight_layout()
    plt.savefig(f"/{save_folder_update}/Figures/{patientID}_{quality}_{method}_{name}_images_and_stats.png")
    plt.close()

    print('plotting other metrics', procrustes_d.shape, s_pred.shape, mid_points.shape)

    fig =  plt.figure(figsize=(10, 3))
    plot1 = fig.add_subplot(131)
    plot2 = fig.add_subplot(132)
    plot3 = fig.add_subplot(133)

    plot1.set_title('Procrustes Distance')
    plot2.set_title('Scale')
    plot3.set_title('Mid Distance')

    x = np.arange(0, procrustes_d.shape[0], 1)
    plot1.scatter(x, procrustes_d)
    plot2.scatter(x, mid_points)
    plot3.scatter(x, s_pred.squeeze().cpu().numpy())

     
    fig.tight_layout()
    fig.savefig(f"/{save_folder_update}/Figures/{patientID}_{quality}_{method}_{name}_othermetrics.png")
    plt.close()
    
    return mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes_metrics, procrustes_d, mid_points, s_pred.squeeze().cpu().numpy()


#metrics to evaluate variability
def calculate_euclidean_distance_planes(pred_pts, ground_truth):
    dist = torch.dist(pred_pts, ground_truth, 2)
    return dist 


def calculate_plane_angle_planes(predicted, actual):
    #calculate surface normals of predicted and actual planes
    
    direction_row = predicted[:,0,0]-predicted[:,0,159]
    direction_col = predicted[:,0,0]-predicted[:,159,0]
    normal = np.cross(direction_row, direction_col)
    direction_row = unit_vector(direction_row)
    direction_col = unit_vector(direction_col)
    n_hat = unit_vector(normal)

    direction_row = actual[:,0,0]-actual[:,0,159]
    direction_col = actual[:,0,0]-actual[:,159,0]
    normal = np.cross(direction_row, direction_col)
    direction_row = unit_vector(direction_row)
    direction_col = unit_vector(direction_col)
    n = unit_vector(normal)
    
    pa = np.arccos(np.matmul(n_hat, n))
    return pa

def calculate_mse_planes(predicted, actual):
    return F.mse_loss(predicted, actual)
 
def calculate_ncc_groundtruth(predicted, actual): #this is taking the mean of predictions, and sampling one image from the ground truth
    return normalized_cross_correlation(predicted, actual)

def calculate_ssim_groundtruth(predicted, actual): #this is taking the mean of predictions, and sampling one image from the ground truth
    return structural_similarity(predicted, actual, data_range=1.0)

def calculate_variability_localization(predicted, planes=False):
    if planes:
        std_points = torch.std(torch.from_numpy(predicted), dim=0)
        scale_epistemic = torch.reshape(std_points, (std_points.shape[0], 160*160*3)) 
        scale_variance_sum_epistemic = torch.mean(scale_epistemic, dim=1)
    else:
        std_points = torch.std(predicted, dim=0)
        scale_epistemic = torch.reshape(std_points, (std_points.shape[1],9)) 
        scale_variance_sum_epistemic = torch.mean(scale_epistemic, dim=1)

    return scale_variance_sum_epistemic

def make_spread(y_data, ed_metrics, pa_grid, ncc, ssim_diff, mse_loss, pro_d, mid_d, scale_d, method="ED"):
    # Create a figure with 5 subplots

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 5 subplots in 1 row with a total size of (20, 4)


    # List of labels for x-axes
    y_labels = ["ED", "PA", "NCC", "SSIM", "MSE", "Procrustes", "Mid Distance", "Scale"]

    # Data for y-axis (epi_std in this case)

    # ed_metrics = (ed_metrics - np.min(ed_metrics)) / (np.max(ed_metrics) - np.min(ed_metrics))
    # pa_grid = (pa_grid - np.min(pa_grid)) / (np.max(pa_grid) - np.min(pa_grid))
    # ncc = (ncc - np.min(ncc)) / (np.max(ncc) - np.min(ncc))
    # ssim_diff = (ssim_diff - np.min(ssim_diff)) / (np.max(ssim_diff) - np.min(ssim_diff))
    # mse_loss = (mse_loss - np.min(mse_loss)) / (np.max(mse_loss) - np.min(mse_loss))
    # y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

    for i, ax in enumerate(axes):
        ax.set_aspect('equal')  # Ensure the aspect ratio is equal
        # ax.set_xlim(0, 20)  # Set x-axis limit to 0-20
        # ax.set_ylim(0, 20)  # Set y-axis limit to 0-20

        x_data = [ed_metrics, pa_grid, ncc, ssim_diff, mse_loss, pro_d, mid_d, scale_d][i]
        
        x_data = np.squeeze(x_data).flatten()
        y_data = np.squeeze(y_data).flatten()

        print(x_data.shape, y_data.shape, 'x_data.shape, y_data.shape')

        # Use circular markers for points before size (32*7), and triangle markers for points after
        mask_circular = np.arange(len(y_data)) < 224
        mask_triangle = ~mask_circular

        print(mask_circular.shape, mask_triangle.shape, 'mask_circular.shape, mask_triangle.shape')

        ax.scatter(np.array(y_data)[mask_circular], np.array(x_data)[mask_circular], c="r", marker='o')
        ax.scatter(np.array(y_data)[mask_triangle], np.array(x_data)[mask_triangle], c="b", marker='^')


        ax.set_ylabel(y_labels[i])
        ax.set_xlabel("Uncertainty".format(method))
        ax.set_title("{} vs {}".format(method, y_labels[i]))

    # Save the figure
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"/{save_folder_update}/Figures/{method}_subplots.png")

    # Close the figure
    plt.close()
   


class Predictor:
    def __init__(self, weeksage, dropoutrate):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")


        #Initialize Base
        model_path_baseline = 'best_model_base_{}_{}.pth'.format(weeksage, float(0))
        self.model_base = Baseline_simple_vgg_backbone(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, dropout=dropoutrate).to(self.device)
        self.model_base.load_state_dict(torch.load(model_path_baseline), strict=True)

        #Initialize MVE
        model_path_mve = 'best_model_deepen_{}_{}.pth'.format(weeksage, float(6))
        self.model_mve = MVE(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, dropout=dropoutrate).to(self.device)
        self.model_mve.load_state_dict(torch.load(model_path_mve), strict=True)


        #Initialize MCD
        model_path_mcd = 'best_model_ttdfullreg_{}_{}.pth'.format(weeksage, float(0.1))
        self.model_mcd = MCD(make_layers_instance_norm_dropout(norm=True, dropout=float(0.1)), num_classes=[3,3,3], fc_size = 512, dropout=float(0.1)).to(self.device)
        self.model_mcd.load_state_dict(torch.load(model_path_mcd), strict=True)

        model_path_mcd2 = 'best_model_ttdfullreg_{}_{}.pth'.format(weeksage, float(0.25))
        self.model_mcd2 = MCD(make_layers_instance_norm_dropout(norm=True, dropout=float(0.25)), num_classes=[3,3,3], fc_size = 512, dropout=float(0.25)).to(self.device)
        self.model_mcd2.load_state_dict(torch.load(model_path_mcd2), strict=True)

        #Initialize evidential model

        model_path_evident = 'best_model_mtl_edl_{}_{}.pth'.format(weeksage, float(0))
        self.model_evident = Evidential(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, dropout=dropoutrate).to(self.device)
        self.model_evident.load_state_dict(torch.load(model_path_evident), strict=True)



        #Initialize DE
        model_path_de = 'best_model_deepen_{}_{}.pth'.format(weeksage, float(0))
        self.model_de = DeepEnsemble(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, dropout=dropoutrate).to(self.device)
        self.model_de.load_state_dict(torch.load(model_path_de), strict=True)

        #Initialize DEQAERTS
        model_path_deqaerts = 'best_model_deepenquaerts_{}_{}.pth'.format(weeksage, float(0))
        self.model_deqaerts = DeepEnsembleQAERTS(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, dropout=dropoutrate).to(self.device)
        self.model_deqaerts.load_state_dict(torch.load(model_path_deqaerts), strict=True)
        
        #Initialize VGGGNL model
        model_path_gnll = 'best_model_mtl_allgnll_{}_{}.pth'.format(weeksage, float(0))
        self.model_gnll = VGGROTGNLL(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, dropout=dropoutrate).to(self.device)
        self.model_gnll.load_state_dict(torch.load(model_path_gnll), strict=True)

        # 'Initialize atlas'
        atlas_path = 'atlas.mat' #Please contact authors for this sensitive data#
        mat = sio.loadmat(atlas_path)
        self.atlas = mat['atlas']#*mat['img_brain_mask']
        
        print('Atlas shape', self.atlas.shape)
        'To torch'
        self.atlas = torch.from_numpy(self.atlas).to(self.device).squeeze()
        self.weeksage = weeksage
        self.dropoutrate = dropoutrate
        
        #plotting the brain
        self.mask_3d = np.zeros((160,160,160), dtype=np.uint8)
        self.brain_mask_path =  'mask_ana_v2_21.pkl'#Please contact authors for this sensitive data#
        self.masked = False
        with open(self.brain_mask_path, 'rb') as f:
            mask_dict = pickle.load(f)
        self.brain_mask = torch.from_numpy(mask_dict['brain_mask']).to(dtype=torch.float32, device=self.model.device).unsqueeze(0).unsqueeze(0) if self.masked else None    #1,1,H,W,D
        self.brain_boun = mask_dict['brain_boun']
        self.points_new = self.calculate_3d_points(self.brain_boun, self.mask_3d==1)
        self.points_old = self.calculate_3d_points(self.brain_boun, self.mask_3d==0)

    def preprocess_video(self, video):
        self.video = []  #N, H, H
        
        'Scale to 160'
        for i in range(video.shape[0]):
            self.video.append(resize(video[i], (160,160)))
        
        'Normalize'
        video_max = np.stack(self.video).max()
        for i in range(len(self.video)):
            self.video[i] = self.video[i]/video_max
        
        self.video = np.stack(self.video) #N, 160, 160
        self.video = torch.from_numpy(self.video).unsqueeze(1).to(device=self.device, dtype=torch.float) #N, 1, 160, 160
        
        return self.video

    def calculate_3d_points(self, mask, region):
        points = np.nonzero(mask*region)
        ls = [i for i in range(points[0].shape[0])]
        ls = random.sample(ls, points[0].shape[0]//3)
        
        points = [ps[ls] for ps in points]
        
        return points
    
    def predict_video(self, file_names):
    

        for k in range(self.model_de.ensemble):
            mope = getattr(self.model_de, 'model'+str(k))
            mope.eval()

        for k in range(self.model_deqaerts.ensemble):
            mope = getattr(self.model_deqaerts, 'model'+str(k))
            mope.eval()

        self.model_gnll.eval()
        self.model_base.eval()
        self.model_mve.eval()
        self.model_evident.eval()
        self.model_mcd.eval()
        self.model_mcd2.eval()

        # 'Initialize atlas'
        atlas_path = 'atlas.mat' #Please contact authors for this sensitive data#

        mat = sio.loadmat(atlas_path)
        self.atlas = mat['atlas']#*mat['img_brain_mask']
        
        print('Atlas shape', self.atlas.shape)
        'To torch'
        self.atlas = torch.from_numpy(self.atlas).to(self.device).squeeze()
                
        ed_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        pa_dict =  {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        mse_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        mse_ref = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        ncc_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        ssim_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        epi_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        procrustes_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        midpoint_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}
        scale_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}

        for i in range(len(file_names)):
            

            params_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[]}


            quality = "un"

            flagged = True

            print(params_dict)


            ds = dicom.dcmread((file_names[i].replace('.pkl', '')))
            video = ds.pixel_array
            video = rgb2gray(video)
            # video = np.rot90(video, k=1, axes=(1,2))
            video = np.flip(video, (1,))
            self.video = self.preprocess_video(video)
            local_batch = self.video

            print(local_batch.shape, 'Video Batch')

            if(local_batch.shape[0] > 120):
                    sys.exit('Shape too big')
            #MODEL DEFINITIONS START HERE!!!!!!!!!!

                                    
            " 0 - Gaussian Likelihood with mean and variance (no shared scale and translation)" 

            params = sum(p.numel() for p in self.model_gnll.parameters() if p.requires_grad)
            params_dict['QAERTS'].append(params)


            # #Multiple Gaussian Likelihood

            # x, mean_og, variances, mean_compute, sigma_compute = self.model_gnll(local_batch)

            # mean_output = mean_og
            # print(mean_output.shape, 'mean_output')
            # mean_poses = torch.mean(mean_output, dim=0)
            # print(mean_poses.shape, 'mean_poses')
            # mu_arr = [i *(160/2) for i in mean_output]
            # mu_arr = torch.stack(mu_arr)   
            # print('TTD/A Samples', mu_arr.shape)
            # temp = torch.mean(mu_arr, dim=0) 
            # loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            # loss = torch.reshape(loss, (loss.shape[1], 9))
            # loss = torch.mean(loss, dim=1)  


            # #Single Gaussian Likelihood

            B = local_batch.shape[0]
            
            x, mean_og, variance, ens = self.model_gnll(local_batch)
            
            mean = mean_og.view(B,3,3).permute(1,0,2)
            mean_output = ens
            variance = variance.view(B,3,3).permute(1,0,2)
            mean_poses = mean

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_p= extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="QAERTS", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, None, None, epistemic_variance_mean_planes, self.atlas, name="QAERTS", standardize=True, method="SumVarEpiPlanes", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)
            mse_dict['QAERTS'].append(mse_planes)
            ed_dict['QAERTS'].append(ed_metrics)
            pa_dict['QAERTS'].append(pa_grid)
            ncc_dict['QAERTS'].append(ncc)
            ssim_dict['QAERTS'].append(ssim_diff)
            epi_dict['QAERTS'].append(epi_std)
            procrustes_dict['QAERTS'].append(procrustes_d)
            midpoint_dict['QAERTS'].append(mid_points)
            scale_dict['QAERTS'].append(s_pred)
            mse_ref['QAERTS'].append(mse_loss)
                   
            print('QAERTS------------------------------------------------------------------------------------------------------------------------***')

            '''1 - Baseline MSE'''
            _, output = self.model_base(local_batch)
            output = output.view(B,3,3).permute(1,0,2)

            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, None, None,  epi_loss=None, atlas=self.atlas, name="Base", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)

            mse_dict['Base'].append(mse_planes)
            ed_dict['Base'].append(ed_metrics)
            pa_dict['Base'].append(pa_grid)
            ncc_dict['Base'].append(ncc)
            ssim_dict['Base'].append(ssim_diff)
            epi_dict['Base'].append(epi_std)
            procrustes_dict['Base'].append(procrustes_d)
            midpoint_dict['Base'].append(mid_points)
            scale_dict['Base'].append(s_pred)
            mse_ref['Base'].append(mse_loss)

            params = sum(p.numel() for p in self.model_base.parameters() if p.requires_grad)
            params_dict['Base'].append(params)

            print('BASE------------------------------------------------------------------------------------------------------------------------***')


            '2-Mean Variance Estimator - MVE'''
            _, output, var = self.model_mve(local_batch)
            output = output.view(B,3,3).permute(1,0,2)

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, None, None,  epi_loss=None, atlas=self.atlas, name="MVE", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)
            
            mse_dict['MVE'].append(mse_planes)
            ed_dict['MVE'].append(ed_metrics)
            pa_dict['MVE'].append(pa_grid)
            ncc_dict['MVE'].append(ncc)
            ssim_dict['MVE'].append(ssim_diff)
            epi_dict['MVE'].append(epi_std)
            procrustes_dict['MVE'].append(procrustes_d)
            midpoint_dict['MVE'].append(mid_points)
            scale_dict['MVE'].append(s_pred)
            mse_ref['MVE'].append(mse_loss)
            

            params = sum(p.numel() for p in self.model_mve.parameters() if p.requires_grad)
            params_dict['MVE'].append(params)

            print('MVE------------------------------------------------------------------------------------------------------------------------***')

            '3-Evidential Deep Learning'

            _, pred_pts = self.model_evident(local_batch)
            gamma, v, alpha, beta = torch.split(pred_pts, int(pred_pts.shape[-1]/4), dim=-1)
            gamma = gamma.view(B,3,3).permute(1,0,2)

            edlvar = beta / (v * (alpha - 1))

            print(edlvar.shape, 'edlvar')

            epistemic_variance_mean = torch.mean(edlvar, dim=1)


            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(gamma, None, None, epistemic_variance_mean, self.atlas, name="EDL", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=edlvar.view(B,3,3).permute(1,0,2)/(160/2), brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)

            
            mse_dict['EDL'].append(mse_planes)
            ed_dict['EDL'].append(ed_metrics)
            pa_dict['EDL'].append(pa_grid)
            ncc_dict['EDL'].append(ncc)
            ssim_dict['EDL'].append(ssim_diff)
            epi_dict['EDL'].append(epi_std)
            procrustes_dict['EDL'].append(procrustes_d)
            midpoint_dict['EDL'].append(mid_points)
            scale_dict['EDL'].append(s_pred)
            mse_ref['EDL'].append(mse_loss)

            params = sum(p.numel() for p in self.model_evident.parameters() if p.requires_grad)
            params_dict['EDL'].append(params)

            print('EDL------------------------------------------------------------------------------------------------------------------------***')


            '4 - Predictions for MCD'

            mc_samples=5
            mean_output = []
            si_output = []
            for ind in range(mc_samples):
                self.model_mcd.eval()
                enable_dropout(self.model_mcd)
                with torch.no_grad():
                    _, mu, si = self.model_mcd(local_batch)
                    mu = mu.view(B,3,3).permute(1,0,2)
                    si = si.view(B,3,3).permute(1,0,2)
                    mean_output.append(mu)
                    si_output.append(si)

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            # print('TTD/A Samples', mu_arr.shape)
            # temp = torch.mean(mu_arr, dim=0) 
            # loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            # loss = torch.reshape(loss, (loss.shape[1], 9))
            # loss = torch.mean(loss, dim=1)  
            
            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)
            
            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="MCD0.1", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')

            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, None, None, epistemic_variance_mean_planes, self.atlas, name="MCD_0.1", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)

            mse_dict['MCD_0.1'].append(mse_planes)
            ed_dict['MCD_0.1'].append(ed_metrics)
            pa_dict['MCD_0.1'].append(pa_grid)
            ncc_dict['MCD_0.1'].append(ncc)
            ssim_dict['MCD_0.1'].append(ssim_diff)
            epi_dict['MCD_0.1'].append(epi_std)
            procrustes_dict['MCD_0.1'].append(procrustes_d)
            midpoint_dict['MCD_0.1'].append(mid_points)
            scale_dict['MCD_0.1'].append(s_pred)
            mse_ref['MCD_0.1'].append(mse_loss)


            params = sum(p.numel() for p in self.model_mcd.parameters() if p.requires_grad)
            params_dict['MCD_0.1'].append(params)


            print('MCD 0.1---------------------------------------------------------------------------------------------------------------')

            '5 - MCD 0.25'  
            mean_output = []
            si_output = []
            for ind in range(mc_samples):
                self.model_mcd2.eval()
                enable_dropout(self.model_mcd2)
                with torch.no_grad():
                    _, mu, si = self.model_mcd2(local_batch)
                    mu = mu.view(B,3,3).permute(1,0,2)
                    si = si.view(B,3,3).permute(1,0,2)
                    mean_output.append(mu)
                    si_output.append(si)

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i*(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            # print('TTD/A Samples', mu_arr.shape)
            # temp = torch.mean(mu_arr, dim=0) 
            # loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            # loss = torch.reshape(loss, (loss.shape[1], 9))
            # loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)
            
            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="MCD0.25", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
                    
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, None, None, epistemic_variance_mean_planes, self.atlas, name="MCD_0.25", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)

            mse_dict['MCD_0.25'].append(mse_planes)
            ed_dict['MCD_0.25'].append(ed_metrics)
            pa_dict['MCD_0.25'].append(pa_grid)
            ncc_dict['MCD_0.25'].append(ncc)
            ssim_dict['MCD_0.25'].append(ssim_diff)
            epi_dict['MCD_0.25'].append(epi_std)
            procrustes_dict['MCD_0.25'].append(procrustes_d)
            midpoint_dict['MCD_0.25'].append(mid_points)
            scale_dict['MCD_0.25'].append(s_pred)
            mse_ref['MCD_0.25'].append(mse_loss)


            params = sum(p.numel() for p in self.model_mcd2.parameters() if p.requires_grad)
            params_dict['MCD_0.25'].append(params)
            


            print('MCD 0.25---------------------------------------------------------------------------------------------------------------')

            '''6 - Deep Ensemble - DE-----------------------------'''
            mean_output = []
            si_output = []
            params = 0
            for ind in range(self.model_de.ensemble):
                print(ind, 'ind')
                model_me = getattr(self.model_de, 'model'+str(ind))
                _, mu, si = model_me(local_batch) #additional ensembles for qaerts
                mu = mu.view(B,3,3).permute(1,0,2)
                si = si.view(B,3,3).permute(1,0,2)
                mean_output.append(mu)
                si_output.append(torch.exp(si))
                params += sum(p.numel() for p in model_me.parameters() if p.requires_grad)
            
        
            params = sum(p.numel() for p in self.model_de.parameters() if p.requires_grad)
            params_dict['DE'].append(params)
        

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            # print('TTD/A Samples', mu_arr.shape)
            # temp = torch.mean(mu_arr, dim=0) 
            # loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            # loss = torch.reshape(loss, (loss.shape[1], 9))
            # loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="DE", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, None, None, epistemic_variance_mean_planes, self.atlas, name="DE", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)

            mse_dict['DE'].append(mse_planes)
            ed_dict['DE'].append(ed_metrics)
            pa_dict['DE'].append(pa_grid)
            ncc_dict['DE'].append(ncc)
            ssim_dict['DE'].append(ssim_diff)
            epi_dict['DE'].append(epi_std)
            procrustes_dict['DE'].append(procrustes_d)
            midpoint_dict['DE'].append(mid_points)
            scale_dict['DE'].append(s_pred)
            mse_ref['DE'].append(mse_loss)


            print('DEEP ENSEMBLE------------------------------------------------------------------------------------------------------------------------***')


            ''' 7 - Deep Ensemble with QAERTS - DEQAERTS-----------------------------'''
         
            mean_output = []
            si_output = []
            params = 0
            for ind in range(self.model_deqaerts.ensemble):
                print(ind, 'ind')
                model_me = getattr(self.model_deqaerts, 'model'+str(ind))
                _, mu, si, _ = model_me(local_batch) #additional ensembles for qaerts
                mu = mu.view(B,3,3).permute(1,0,2)
                si = si.view(B,3,3).permute(1,0,2)
                mean_output.append(mu)
                si_output.append(torch.exp(si))
                params += sum(p.numel() for p in model_me.parameters() if p.requires_grad)
            
        
            params = sum(p.numel() for p in self.model_deqaerts.parameters() if p.requires_grad)
            params_dict['DEQAERTS'].append(params)
        

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            # print('TTD/A Samples', mu_arr.shape)
            # temp = torch.mean(mu_arr, dim=0) 
            # loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            # loss = torch.reshape(loss, (loss.shape[1], 9))
            # loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)
            
            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)
            
            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="DEQAERTS", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, None, None, epistemic_variance_mean_planes, self.atlas, name="DEQAERTS", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new, videos=True)

            mse_dict['DEQAERTS'].append(mse_planes)
            ed_dict['DEQAERTS'].append(ed_metrics)
            pa_dict['DEQAERTS'].append(pa_grid)
            ncc_dict['DEQAERTS'].append(ncc)
            ssim_dict['DEQAERTS'].append(ssim_diff)
            epi_dict['DEQAERTS'].append(epi_std)
            procrustes_dict['DEQAERTS'].append(procrustes_d)
            midpoint_dict['DEQAERTS'].append(mid_points)
            scale_dict['DEQAERTS'].append(s_pred)
            mse_ref['DEQAERTS'].append(mse_loss)

            print('DEEP ENSEMBLE with QAERTS -----------------------------------------------------------------------------------------------------------------------***')




            results = []
            for key in params_dict:

                results.append([key, params_dict[key]])

            # Create headers for the table
            headers = ["Method", "Params"]

            # Print the table using tabulate
            table = tabulate(results, headers, tablefmt="fancy_grid")

            # Print the table
            print(table)   

            if (i > 0): 
                sys.exit('Testing Qualitatively Patient')


    def predict(self, files_training):
        

        batch_size = int(save_folder)  
        B = batch_size
        
        print('Batch Size--------------->>', batch_size)

        params_dataset =  {'sample_num':batch_size,#50
                      'mode':'training'}
        params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0,
              'drop_last':True}     # for data generator
        
        for k in range(self.model_de.ensemble):
            mope = getattr(self.model_de, 'model'+str(k))
            mope.eval()

        for k in range(self.model_deqaerts.ensemble):
            mope = getattr(self.model_deqaerts, 'model'+str(k))
            mope.eval()

        self.model_gnll.eval()
        self.model_base.eval()
        self.model_mve.eval()
        self.model_evident.eval()
        self.model_mcd.eval()
        self.model_mcd2.eval()
        # self.model_mimoqaerts.eval()
        self.model_mimomve.eval()
        self.model_mimve.eval()
        self.model_quat.eval()
        self.model_axis.eval()
        self.model_euler.eval()
        self.model_mat.eval()
        self.model_sum.eval()
        self.model_gnll_sum.eval()

        
        dihedral_angle = []
        plane_angle = []
        SSIM = []
        PSNR = []
        NCC = []
      
        # training_set = Dataset_end_to_end_ood(files_ood, **params_dataset)
        # training_generator = data.DataLoader(training_set, **params)
        # ref_pts = training_set.reference_points()
        # ref_pts = torch.from_numpy(ref_pts).to(device=self.device, dtype=torch.float) #N,3
                
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

        #      # Transfer to GPU
        #     local_batch = local_batch.to(device=self.device, dtype=torch.float).squeeze(0)
        #     local_y = torch.squeeze(local_y).to(device=self.device, dtype=torch.float)
        #     local_scale = local_scale.squeeze(0).to(device=self.device, dtype=torch.float)
        #     local_rotation = local_rotation.squeeze(0).to(device=self.device, dtype=torch.float)
        #     local_translation = local_translation.squeeze(0).to(device=self.device, dtype=torch.float)

        #     initial_rotation = local_rotation
        #     local_scale = 1/local_scale
        #     local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(batch_size, 1, 1)  #B,3,3
        #     print('Local Rotations', local_rotation.shape)
        #     local_quat = matrix_to_quaternion(local_rotation)
        #     print('Local Quaternion', local_quat.shape)
        #     local_euler = matrix_to_euler_angles(local_rotation, convention='XYZ')
        #     print('Local Euler Angles', local_euler.shape)
        #     local_axis = matrix_to_axis_angle(local_rotation)
        #     print('Local Axis Angles shape', local_axis.shape)
        #     local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
        #     local_translation = (local_translation).unsqueeze(-1)
        #     local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
            
        #     local_grid = F.affine_grid(local_rotation, (batch_size,1,1,160,160)).squeeze(1)
        #     local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0]))

        #     print('Local pts', local_pts.shape)

        #     print('Local ref pts', local_pts[:,0,0].shape)


        #     displaygroundtruth(local_pts, self.atlas, name="GroundTruth", method="Sampling",mode=True)

        #     sys.exit(0)here

        #     # Your other calculations or processes here

        # sys.exit(0)
        # # Convert lists of tensors to tensors
        # saved_info["local_batch"] = torch.stack(saved_info["local_batch"])
        # saved_info["local_y"] = torch.stack(saved_info["local_y"])
        # saved_info["local_scale"] = torch.stack(saved_info["local_scale"])
        # saved_info["local_rotation"] = torch.stack(saved_info["local_rotation"])
        # saved_info["local_translation"] = torch.stack(saved_info["local_translation"])
        # saved_info["local_vol"] = torch.stack(saved_info["local_vol"])

        # # Save the dictionaries

    

        # torch.save(saved_info, 'saved_info_{}_{}.pth'.format(weeksage, batch_size))
        

          
        ed_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[], 'MIMOMVE':[], 'MIMOQAERTS':[], 'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[], 'MIMO5':[]}
        pa_dict =  {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        mse_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        mse_ref = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        ncc_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        ssim_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        epi_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        procrustes_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        midpoint_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}
        scale_dict = {'MCD_0.1':[], 'MCD_0.25':[],  'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[],'MIMOMVE':[],'MIMOQAERTS':[],'MIMVE':[], 'Axis':[], 'Euler':[], 'Quaternion':[], 'Matrix':[], 'Sum':[], 'Mean':[],'MIMO5':[]}

            

        my_batch = []
        loaded_info = torch.load('saved_info_{}.pth'.format(batch_size))
        # Access the saved tensors from the loaded dictionary
        local_batch_list = loaded_info["local_batch"]
        local_y_list = loaded_info["local_y"]
        local_scale_list = loaded_info["local_scale"]
        local_rotation_list = loaded_info["local_rotation"]
        local_translation_list = loaded_info["local_translation"]
        local_vol_list = loaded_info["local_vol"]
        
        print(local_batch_list.shape, 'local_batch_list')
        
        id_size = len(local_batch_list)

          
        # loaded_info_ood = torch.load('saved_info_22_{}.pth'.format(batch_size))
        # # Access the saved tensors from the loaded dictionary
        # local_batch_list_ood = loaded_info_ood["local_batch"]
        # local_y_list_ood = loaded_info_ood["local_y"]
        # local_scale_list_ood = loaded_info_ood["local_scale"]
        # local_rotation_list_ood = loaded_info_ood["local_rotation"]
        # local_translation_list_ood = loaded_info_ood["local_translation"]
        # local_vol_list_ood = loaded_info_ood["local_vol"]

        # print(local_batch_list_ood.shape, 'local_ood_list')

        # ood_size = len(local_batch_list_ood)

        # local_batch_list = torch.cat((local_batch_list, local_batch_list_ood), dim=0)
        # local_y_list = torch.cat((local_y_list, local_y_list_ood), dim=0)
        # local_scale_list = torch.cat((local_scale_list, local_scale_list_ood), dim=0)
        # local_rotation_list = torch.cat((local_rotation_list, local_rotation_list_ood), dim=0)
        # local_translation_list = torch.cat((local_translation_list, local_translation_list_ood), dim=0)
        # local_vol_list = torch.cat((local_vol_list, local_vol_list_ood), dim=0)

        # print(ood_size, 'ood_size', id_size, 'id_size')

    
        # Simulate the behavior of the original training_generator loop
        for i in range(len(local_batch_list)):
            

            # params_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'quaerts':[], 'quaertslw':[], 'qtslw':[], 'DE':[], 'MVE':[], 'Base': [], 'rpr': [], 'rprstochastic': [], 'EDL': [], 'singleensemble': [],'quaertslwNOSHARE':[], 'quaertslw9':[], 'qtslw9':[], 'QAERTS':[], 'meanMTLNOSHARE': [], 'linearMTLNOSHARE':[]}
          
            params_dict = {'MCD_0.1':[], 'MCD_0.25':[], 'DE':[], 'MVE':[], 'Base': [], 'EDL': [], 'QAERTS':[], 'DEQAERTS':[], 'MIMOMVE':[], 'MIMOQAERTS':[],'MIMVE':[]}

            local_batch = local_batch_list[i]
            local_y = local_y_list[i]
            local_scale = local_scale_list[i]
            local_rotation = local_rotation_list[i]
            local_translation = local_translation_list[i]
            local_vol = local_vol_list[i]

            H,W = 160, 160

            # Transfer to GPU
            multiple_local_batch = torch.stack([local_batch.to(device=self.device, dtype=torch.float).squeeze(0) for _ in range(2)])
            local_batch = local_batch.to(device=self.device, dtype=torch.float).squeeze(0)
            local_y = torch.squeeze(local_y).to(device=self.device, dtype=torch.float)
            local_scale = local_scale.squeeze(0).to(device=self.device, dtype=torch.float)
            local_rotation = local_rotation.squeeze(0).to(device=self.device, dtype=torch.float)
            local_translation = local_translation.squeeze(0).to(device=self.device, dtype=torch.float)

            print('Local Batch Images', local_batch.shape)
    
            initial_rotation = local_rotation
            local_scale = 1/local_scale
            local_scale = local_scale.unsqueeze(-1)*torch.eye(3, dtype=local_scale.dtype, device=local_scale.device).repeat(batch_size, 1, 1)  #B,3,3
            print('Local Rotations', local_rotation.shape)
            local_quat = matrix_to_quaternion(local_rotation)
            print('Local Quaternion', local_quat.shape)
            local_euler = matrix_to_euler_angles(local_rotation, convention='XYZ')
            print('Local Euler Angles', local_euler.shape)
            local_axis = matrix_to_axis_angle(local_rotation)
            print('Local Axis Angles shape', local_axis.shape)
            local_rotation = torch.einsum('bij,bjk->bik', local_scale, local_rotation)
            local_translation = (local_translation).unsqueeze(-1)
            local_rotation = torch.cat((local_rotation, local_translation), dim=-1)
            
            local_grid = F.affine_grid(local_rotation, (batch_size,1,1,160,160)).squeeze(1)
            local_pts = torch.stack((local_grid[:,0,0],local_grid[:,0,-1],local_grid[:,-1,0]))

            print('Local pts', local_pts.shape)

            print('Local ref pts', local_pts[:,0,0].shape)


            my_batch.append(local_pts)

            # 'Use ground truth volume if available, instead of the atlas'
            # mat = sio.loadmat(files_training[i])      
            # img_vol = np.squeeze(mat['img_brain'])
            # img_mask = np.squeeze(mat['img_brain_mask'])
            # img_mask[img_mask>0]=1
            # img_vol = img_vol/img_vol.max() 
            # self.atlas = torch.from_numpy(img_vol).to(self.device)

            quality = "un"
            flagged = True

            _, output, var = self.model_quat(local_batch)
            output = output.view(B,3,3).permute(1,0,2)
            # var = var.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9)) 
            loss3 = torch.mean(loss3, dim=1)   
            print(loss3.shape, 'loss3')

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="Quaternion", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            
            mse_dict['Quaternion'].append(mse_planes)
            ed_dict['Quaternion'].append(ed_metrics)
            pa_dict['Quaternion'].append(pa_grid)
            ncc_dict['Quaternion'].append(ncc)
            ssim_dict['Quaternion'].append(ssim_diff)
            epi_dict['Quaternion'].append(epi_std)
            procrustes_dict['Quaternion'].append(procrustes_d)
            midpoint_dict['Quaternion'].append(mid_points)
            scale_dict['Quaternion'].append(s_pred)
            mse_ref['Quaternion'].append(mse_loss)
            
                        
            print('Quaternion------------------------------------------------------------------------------------------------------------------------***')


            _, output, var = self.model_axis(local_batch)
            output = output.view(B,3,3).permute(1,0,2)
            # var = var.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9)) 
            loss3 = torch.mean(loss3, dim=1)   
            print(loss3.shape, 'loss3')

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="Axis", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            
            mse_dict['Axis'].append(mse_planes)
            ed_dict['Axis'].append(ed_metrics)
            pa_dict['Axis'].append(pa_grid)
            ncc_dict['Axis'].append(ncc)
            ssim_dict['Axis'].append(ssim_diff)
            epi_dict['Axis'].append(epi_std)
            procrustes_dict['Axis'].append(procrustes_d)
            midpoint_dict['Axis'].append(mid_points)
            scale_dict['Axis'].append(s_pred)
            mse_ref['Axis'].append(mse_loss)
            
                        
            print('Axis------------------------------------------------------------------------------------------------------------------------***')


            _, output, var = self.model_euler(local_batch)
            output = output.view(B,3,3).permute(1,0,2)
            # var = var.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9))
            loss3 = torch.mean(loss3, dim=1)
            print(loss3.shape, 'loss3')

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="Euler", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            
            mse_dict['Euler'].append(mse_planes)
            ed_dict['Euler'].append(ed_metrics)
            pa_dict['Euler'].append(pa_grid)
            ncc_dict['Euler'].append(ncc)
            ssim_dict['Euler'].append(ssim_diff)
            epi_dict['Euler'].append(epi_std)
            procrustes_dict['Euler'].append(procrustes_d)
            midpoint_dict['Euler'].append(mid_points)
            scale_dict['Euler'].append(s_pred)
            mse_ref['Euler'].append(mse_loss)
           
            print('Euler------------------------------------------------------------------------------------------------------------------------***')
      
            _, output, var = self.model_mat(local_batch)
            output = output.view(B,3,3).permute(1,0,2)
            # var = var.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9))
            loss3 = torch.mean(loss3, dim=1)
            print(loss3.shape, 'loss3')

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="Matrix", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            
            mse_dict['Matrix'].append(mse_planes)
            ed_dict['Matrix'].append(ed_metrics)
            pa_dict['Matrix'].append(pa_grid)
            ncc_dict['Matrix'].append(ncc)
            ssim_dict['Matrix'].append(ssim_diff)
            epi_dict['Matrix'].append(epi_std)
            procrustes_dict['Matrix'].append(procrustes_d)
            midpoint_dict['Matrix'].append(mid_points)
            scale_dict['Matrix'].append(s_pred)
            mse_ref['Matrix'].append(mse_loss)
           
            print('Matrix------------------------------------------------------------------------------------------------------------------------***')
    


            # #Single Gaussian Likelihood
            
            x, mean_og, variance, ens = self.model_sum(local_batch)
            
            mean_poses = mean_og
            mean_output = ens
            print(mean_poses.shape, 'mean_poses')
            loss = F.mse_loss(mean_poses*(160/2), local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_p= extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="ModSum", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="SumMod", standardize=True, method="SumVarEpiPlanes", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            mse_dict['Sum'].append(mse_planes)
            ed_dict['Sum'].append(ed_metrics)
            pa_dict['Sum'].append(pa_grid)
            ncc_dict['Sum'].append(ncc)
            ssim_dict['Sum'].append(ssim_diff)
            epi_dict['Sum'].append(epi_std)
            procrustes_dict['Sum'].append(procrustes_d)
            midpoint_dict['Sum'].append(mid_points)
            scale_dict['Sum'].append(s_pred)
            mse_ref['Sum'].append(mse_loss)

            print('Sum------------------------------------------------------------------------------------------------------------------------***')



            x, _, _, mean_og, variance = self.model_gnll_sum(local_batch)
            
            mean_poses = mean_og
            print(mean_poses.shape, 'mean_poses')
            loss = F.mse_loss(mean_poses*(160/2), local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_p= extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="ModMean", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="MeanMod", standardize=True, method="SumVarEpiPlanes", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            mse_dict['Mean'].append(mse_planes)
            ed_dict['Mean'].append(ed_metrics)
            pa_dict['Mean'].append(pa_grid)
            ncc_dict['Mean'].append(ncc)
            ssim_dict['Mean'].append(ssim_diff)
            epi_dict['Mean'].append(epi_std)
            procrustes_dict['Mean'].append(procrustes_d)
            midpoint_dict['Mean'].append(mid_points)
            scale_dict['Mean'].append(s_pred)
            mse_ref['Mean'].append(mse_loss)

            print('Mean------------------------------------------------------------------------------------------------------------------------***')

            #MODEL DEFINITIONS START HERE!!!!!!!!!!


            params = sum(p.numel() for p in self.model_mimomve.parameters() if p.requires_grad)
            params_dict['MIMVE'].append(params)

            '2-Mi-MVEstimator - MIMVE'''
            _, output, var = self.model_mimve(multiple_local_batch)
            # var = var.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9)) 
            loss3 = torch.mean(loss3, dim=1)   
            print(loss3.shape, 'loss3')

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="MIMVE", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            
            mse_dict['MIMVE'].append(mse_planes)
            ed_dict['MIMVE'].append(ed_metrics)
            pa_dict['MIMVE'].append(pa_grid)
            ncc_dict['MIMVE'].append(ncc)
            ssim_dict['MIMVE'].append(ssim_diff)
            epi_dict['MIMVE'].append(epi_std)
            procrustes_dict['MIMVE'].append(procrustes_d)
            midpoint_dict['MIMVE'].append(mid_points)
            scale_dict['MIMVE'].append(s_pred)
            mse_ref['MIMVE'].append(mse_loss)

                               
            print('MIMVE------------------------------------------------------------------------------------------------------------------------***')



    
            " 0 - MIMOMVE" 

            params = sum(p.numel() for p in self.model_mimomve.parameters() if p.requires_grad)
            params_dict['MIMOMVE'].append(params)
            _, mean_output, sigma = self.model_mimomve(multiple_local_batch)

            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            print('TTD/A Samples', mu_arr.shape)
            temp = torch.mean(mu_arr, dim=0) 
            loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_p= extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="MIMOMVE", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="MIMOMVE", standardize=True, method="SumVarEpiPlanes", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            mse_dict['MIMOMVE'].append(mse_planes)
            ed_dict['MIMOMVE'].append(ed_metrics)
            pa_dict['MIMOMVE'].append(pa_grid)
            ncc_dict['MIMOMVE'].append(ncc)
            ssim_dict['MIMOMVE'].append(ssim_diff)
            epi_dict['MIMOMVE'].append(epi_std)
            procrustes_dict['MIMOMVE'].append(procrustes_d)
            midpoint_dict['MIMOMVE'].append(mid_points)
            scale_dict['MIMOMVE'].append(s_pred)
            mse_ref['MIMOMVE'].append(mse_loss)
                   
            print('MIMOMVE------------------------------------------------------------------------------------------------------------------------***')


     
                                 
            "0+1 - MIMOQAERTS" 

            params = sum(p.numel() for p in self.model_mimoqaerts.parameters() if p.requires_grad)
            params_dict['MIMOQAERTS'].append(params)

            _, mean_output, sigma = self.model_mimoqaerts(multiple_local_batch)
            mean_poses = torch.mean(mean_output, dim=0)
            loss = F.mse_loss(mean_poses*(160/2), local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_p = extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="MIMOQAERTS", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="MIMOQAERTS", standardize=True, method="SumVarEpiPlanes", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            mse_dict['MIMOQAERTS'].append(mse_planes)
            ed_dict['MIMOQAERTS'].append(ed_metrics)
            pa_dict['MIMOQAERTS'].append(pa_grid)
            ncc_dict['MIMOQAERTS'].append(ncc)
            ssim_dict['MIMOQAERTS'].append(ssim_diff)
            epi_dict['MIMOQAERTS'].append(epi_std)
            procrustes_dict['MIMOQAERTS'].append(procrustes_d)
            midpoint_dict['MIMOQAERTS'].append(mid_points)
            scale_dict['MIMOQAERTS'].append(s_pred)
            mse_ref['MIMOQAERTS'].append(mse_loss)
                   
            print('MIMOQAERTS------------------------------------------------------------------------------------------------------------------------***')
         

            # #Multiple Gaussian Likelihood

            # x, mean_og, variances, mean_compute, sigma_compute = self.model_gnll(local_batch)

            # mean_output = mean_og
            # print(mean_output.shape, 'mean_output')
            # mean_poses = torch.mean(mean_output, dim=0)
            # print(mean_poses.shape, 'mean_poses')
            # mu_arr = [i *(160/2) for i in mean_output]
            # mu_arr = torch.stack(mu_arr)   
            # print('TTD/A Samples', mu_arr.shape)
            # temp = torch.mean(mu_arr, dim=0) 
            # loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            # loss = torch.reshape(loss, (loss.shape[1], 9))
            # loss = torch.mean(loss, dim=1)  

            " 0 - Gaussian Likelihood with mean and variance (no shared scale and translation)" 

            # #Single Gaussian Likelihood
            
            x, mean_og, variance, ens = self.model_gnll(local_batch)
            
            mean = mean_og.view(B,3,3).permute(1,0,2)
            mean_output = ens
            variance = variance.view(B,3,3).permute(1,0,2)
            mean_poses = mean
            print(mean_poses.shape, 'mean_poses')
            loss = F.mse_loss(mean_poses*(160/2), local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_p= extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="QAERTS", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="QAERTS", standardize=True, method="SumVarEpiPlanes", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            mse_dict['QAERTS'].append(mse_planes)
            ed_dict['QAERTS'].append(ed_metrics)
            pa_dict['QAERTS'].append(pa_grid)
            ncc_dict['QAERTS'].append(ncc)
            ssim_dict['QAERTS'].append(ssim_diff)
            epi_dict['QAERTS'].append(epi_std)
            procrustes_dict['QAERTS'].append(procrustes_d)
            midpoint_dict['QAERTS'].append(mid_points)
            scale_dict['QAERTS'].append(s_pred)
            mse_ref['QAERTS'].append(mse_loss)

            params = sum(p.numel() for p in self.model_gnll.parameters() if p.requires_grad)
            params_dict['QAERTS'].append(params)
                   
            print('QAERTS------------------------------------------------------------------------------------------------------------------------***')


            '''1 - Baseline MSE'''
            _, output = self.model_base(local_batch)
            output = output.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9)) 
            loss3 = torch.mean(loss3, dim=1)   
            print(loss3.shape, 'loss3')

            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="Base", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)

            mse_dict['Base'].append(mse_planes)
            ed_dict['Base'].append(ed_metrics)
            pa_dict['Base'].append(pa_grid)
            ncc_dict['Base'].append(ncc)
            ssim_dict['Base'].append(ssim_diff)
            epi_dict['Base'].append(epi_std)
            procrustes_dict['Base'].append(procrustes_d)
            midpoint_dict['Base'].append(mid_points)
            scale_dict['Base'].append(s_pred)
            mse_ref['Base'].append(mse_loss)

            params = sum(p.numel() for p in self.model_base.parameters() if p.requires_grad)
            params_dict['Base'].append(params)

            print('BASE------------------------------------------------------------------------------------------------------------------------***')


            '2-Mean Variance Estimator - MVE'''
            _, output, var = self.model_mve(local_batch)
            output = output.view(B,3,3).permute(1,0,2)
            # var = var.view(B,3,3).permute(1,0,2)
            loss3 = F.mse_loss(output*(160/2), local_pts*(160/2),reduction='none')
            loss3 = torch.reshape(loss3, (loss3.shape[1], 9)) 
            loss3 = torch.mean(loss3, dim=1)   
            print(loss3.shape, 'loss3')

            var = torch.sqrt(torch.exp(var))
            print(var.shape, 'variance_points_gap')
              
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(output, local_pts, loss3,  epi_loss=None, atlas=self.atlas, name="MVE", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=None, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)
            
            mse_dict['MVE'].append(mse_planes)
            ed_dict['MVE'].append(ed_metrics)
            pa_dict['MVE'].append(pa_grid)
            ncc_dict['MVE'].append(ncc)
            ssim_dict['MVE'].append(ssim_diff)
            epi_dict['MVE'].append(epi_std)
            procrustes_dict['MVE'].append(procrustes_d)
            midpoint_dict['MVE'].append(mid_points)
            scale_dict['MVE'].append(s_pred)
            mse_ref['MVE'].append(mse_loss)
            

            params = sum(p.numel() for p in self.model_mve.parameters() if p.requires_grad)
            params_dict['MVE'].append(params)

            print('MVE------------------------------------------------------------------------------------------------------------------------***')

            '3-Evidential Deep Learning'

            _, pred_pts = self.model_evident(local_batch)
            gamma, v, alpha, beta = torch.split(pred_pts, int(pred_pts.shape[-1]/4), dim=-1)
            gamma = gamma.view(B,3,3).permute(1,0,2)

            edlvar = beta / (v * (alpha - 1))

            print(edlvar.shape, 'edlvar')

            epistemic_variance_mean = torch.mean(edlvar, dim=1)

            loss = F.mse_loss(gamma*(160/2), local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1) 

            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(gamma, local_pts, loss, epistemic_variance_mean, self.atlas, name="EDL", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i, std_points=edlvar.view(B,3,3).permute(1,0,2)/(160/2), brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)

            
            mse_dict['EDL'].append(mse_planes)
            ed_dict['EDL'].append(ed_metrics)
            pa_dict['EDL'].append(pa_grid)
            ncc_dict['EDL'].append(ncc)
            ssim_dict['EDL'].append(ssim_diff)
            epi_dict['EDL'].append(epi_std)
            procrustes_dict['EDL'].append(procrustes_d)
            midpoint_dict['EDL'].append(mid_points)
            scale_dict['EDL'].append(s_pred)
            mse_ref['EDL'].append(mse_loss)

            params = sum(p.numel() for p in self.model_evident.parameters() if p.requires_grad)
            params_dict['EDL'].append(params)

            print('EDL------------------------------------------------------------------------------------------------------------------------***')


            '4 - Predictions for MCD'

            mc_samples=5
            mean_output = []
            si_output = []
            for ind in range(mc_samples):
                self.model_mcd.eval()
                enable_dropout(self.model_mcd)
                with torch.no_grad():
                    _, mu, si = self.model_mcd(local_batch)
                    mu = mu.view(B,3,3).permute(1,0,2)
                    si = si.view(B,3,3).permute(1,0,2)
                    mean_output.append(mu)
                    si_output.append(si)

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            print('TTD/A Samples', mu_arr.shape)
            temp = torch.mean(mu_arr, dim=0) 
            loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  
            
            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)
            
            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="MCD0.1", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')

            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="MCD_0.1", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)

            mse_dict['MCD_0.1'].append(mse_planes)
            ed_dict['MCD_0.1'].append(ed_metrics)
            pa_dict['MCD_0.1'].append(pa_grid)
            ncc_dict['MCD_0.1'].append(ncc)
            ssim_dict['MCD_0.1'].append(ssim_diff)
            epi_dict['MCD_0.1'].append(epi_std)
            procrustes_dict['MCD_0.1'].append(procrustes_d)
            midpoint_dict['MCD_0.1'].append(mid_points)
            scale_dict['MCD_0.1'].append(s_pred)
            mse_ref['MCD_0.1'].append(mse_loss)


            params = sum(p.numel() for p in self.model_mcd.parameters() if p.requires_grad)
            params_dict['MCD_0.1'].append(params)


            print('MCD 0.1---------------------------------------------------------------------------------------------------------------')

            '5 - MCD 0.25'  
            mean_output = []
            si_output = []
            for ind in range(mc_samples):
                self.model_mcd2.eval()
                enable_dropout(self.model_mcd2)
                with torch.no_grad():
                    _, mu, si = self.model_mcd2(local_batch)
                    mu = mu.view(B,3,3).permute(1,0,2)
                    si = si.view(B,3,3).permute(1,0,2)
                    mean_output.append(mu)
                    si_output.append(si)

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i*(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            print('TTD/A Samples', mu_arr.shape)
            temp = torch.mean(mu_arr, dim=0) 
            loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)
            
            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="MCD0.25", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
                    
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="MCD_0.25", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)

            mse_dict['MCD_0.25'].append(mse_planes)
            ed_dict['MCD_0.25'].append(ed_metrics)
            pa_dict['MCD_0.25'].append(pa_grid)
            ncc_dict['MCD_0.25'].append(ncc)
            ssim_dict['MCD_0.25'].append(ssim_diff)
            epi_dict['MCD_0.25'].append(epi_std)
            procrustes_dict['MCD_0.25'].append(procrustes_d)
            midpoint_dict['MCD_0.25'].append(mid_points)
            scale_dict['MCD_0.25'].append(s_pred)
            mse_ref['MCD_0.25'].append(mse_loss)


            params = sum(p.numel() for p in self.model_mcd2.parameters() if p.requires_grad)
            params_dict['MCD_0.25'].append(params)
            


            print('MCD 0.25---------------------------------------------------------------------------------------------------------------')

            '''6 - Deep Ensemble - DE-----------------------------'''
            mean_output = []
            si_output = []
            params = 0
            for ind in range(self.model_de.ensemble):
                print(ind, 'ind')
                model_me = getattr(self.model_de, 'model'+str(ind))
                _, mu, si = model_me(local_batch) #additional ensembles for qaerts
                mu = mu.view(B,3,3).permute(1,0,2)
                si = si.view(B,3,3).permute(1,0,2)
                mean_output.append(mu)
                si_output.append(torch.exp(si))
                params += sum(p.numel() for p in model_me.parameters() if p.requires_grad)
            
        
            params = sum(p.numel() for p in self.model_de.parameters() if p.requires_grad)
            params_dict['DE'].append(params)
        

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            print('TTD/A Samples', mu_arr.shape)
            temp = torch.mean(mu_arr, dim=0) 
            loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)

            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)

            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="DE", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="DE", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)

            mse_dict['DE'].append(mse_planes)
            ed_dict['DE'].append(ed_metrics)
            pa_dict['DE'].append(pa_grid)
            ncc_dict['DE'].append(ncc)
            ssim_dict['DE'].append(ssim_diff)
            epi_dict['DE'].append(epi_std)
            procrustes_dict['DE'].append(procrustes_d)
            midpoint_dict['DE'].append(mid_points)
            scale_dict['DE'].append(s_pred)
            mse_ref['DE'].append(mse_loss)


            print('DEEP ENSEMBLE------------------------------------------------------------------------------------------------------------------------***')


            ''' 7 - Deep Ensemble with QAERTS - DEQAERTS-----------------------------'''
         
            mean_output = []
            si_output = []
            params = 0
            for ind in range(self.model_deqaerts.ensemble):
                print(ind, 'ind')
                model_me = getattr(self.model_deqaerts, 'model'+str(ind))
                _, mu, si, _ = model_me(local_batch) #additional ensembles for qaerts
                mu = mu.view(B,3,3).permute(1,0,2)
                si = si.view(B,3,3).permute(1,0,2)
                mean_output.append(mu)
                si_output.append(torch.exp(si))
                params += sum(p.numel() for p in model_me.parameters() if p.requires_grad)
            
        
            params = sum(p.numel() for p in self.model_deqaerts.parameters() if p.requires_grad)
            params_dict['DEQAERTS'].append(params)
        

            mean_output = torch.stack(mean_output)
            print(mean_output.shape, 'mean_output')
            mean_poses = torch.mean(mean_output, dim=0)
            print(mean_poses.shape, 'mean_poses')
            mu_arr = [i *(160/2) for i in mean_output]
            mu_arr = torch.stack(mu_arr)   
            print('TTD/A Samples', mu_arr.shape)
            temp = torch.mean(mu_arr, dim=0) 
            loss = F.mse_loss(temp, local_pts*(160/2),reduction='none')
            loss = torch.reshape(loss, (loss.shape[1], 9))
            loss = torch.mean(loss, dim=1)  

            epistemic_variance_mean = calculate_variability_localization(mean_output, planes=False)

            enemble_grid, ensemble_slices, ensemble_procrustes = extract_ensemble_predictions(mean_output, self.atlas)
            
            epistemic_variance_mean_planes = calculate_variability_localization(enemble_grid, planes=True)
            
            plot_ensembles(ensemble_slices, enemble_grid, epistemic_variance_mean, epistemic_variance_mean_planes, name="DEQAERTS", method="SumSTDEpi")

            print(epistemic_variance_mean_planes.shape, 'epistemic_variance_mean_planes')

            std_points = torch.std(mean_output, dim=0)
            print(std_points.shape, 'std_points')
        
            mse_loss, ssim_diff, ncc, pa_grid, ed_metrics, epi_std, mse_planes, procrustes_d, mid_points, s_pred = plot_all_images_and_stats(mean_poses, local_pts, loss, epistemic_variance_mean_planes, self.atlas, name="DEQAERTS", standardize=True, method="SumVarEpiPlane", quality=quality, original=local_batch, plot=flagged, patientID=i,  std_points=std_points, brain_boundaries_old=self.points_old, brain_boundaries_new=self.points_new)

            mse_dict['DEQAERTS'].append(mse_planes)
            ed_dict['DEQAERTS'].append(ed_metrics)
            pa_dict['DEQAERTS'].append(pa_grid)
            ncc_dict['DEQAERTS'].append(ncc)
            ssim_dict['DEQAERTS'].append(ssim_diff)
            epi_dict['DEQAERTS'].append(epi_std)
            procrustes_dict['DEQAERTS'].append(procrustes_d)
            midpoint_dict['DEQAERTS'].append(mid_points)
            scale_dict['DEQAERTS'].append(s_pred)
            mse_ref['DEQAERTS'].append(mse_loss)

            print('DEEP ENSEMBLE with QAERTS -----------------------------------------------------------------------------------------------------------------------***')




            results = []
            for key in params_dict:

                results.append([key, params_dict[key]])

            # Create headers for the table
            headers = ["Method", "Params"]

            # Print the table using tabulate
            table = tabulate(results, headers, tablefmt="fancy_grid")

            # Print the table
            print(table)   

        results = []

        for key in ed_dict:
            mean_ed = np.mean(ed_dict[key])
            std_ed = np.std(ed_dict[key])
            
            mean_pa = np.mean(pa_dict[key])
            std_pa = np.std(pa_dict[key])
            
            mean_mse = np.mean(mse_dict[key])
            std_mse = np.std(mse_dict[key])
            
            mean_ncc = np.mean(ncc_dict[key])
            std_ncc = np.std(ncc_dict[key])
            
            mean_ssim = np.mean(ssim_dict[key])
            std_ssim = np.std(ssim_dict[key])

            mean_proc = np.mean(procrustes_dict[key])
            std_proc = np.std(procrustes_dict[key])

            mean_mid = np.mean(midpoint_dict[key])
            std_mid = np.std(midpoint_dict[key])

            mean_scale = np.mean(scale_dict[key])
            std_scale = np.std(scale_dict[key])

            mean_ref = np.mean(mse_ref[key])
            std_ref = np.std(mse_ref[key])

            results.append([key, mean_ed, std_ed, mean_pa, std_pa, mean_mse, std_mse, mean_ncc, std_ncc, mean_ssim, std_ssim, mean_proc, std_proc, mean_mid, std_mid, mean_scale, std_scale, mean_ref, std_ref])

        # Create headers for the table
        headers = ["Method", "Mean ED", "Std ED", "Mean PA", "Std PA", "Mean MSE", "Std MSE", "Mean NCC", "Std NCC", "Mean SSIM", "Std SSIM", "Mean Procrustes", "Std Procrustes", "Mean Midpoint", "Std Midpoint", "Mean Scale", "Std Scale", "Mean MSE-Ref", "Std MSE-Ref"]

       
        # Print the table using tabulate
        table = tabulate(results, headers, tablefmt="fancy_grid")

        # Print the table
        # print(table)    

        # keys_to_plot = ['Base', 'MVE', 'QAERTS', 'EDL', 'MCD_0.1', 'MCD_0.25', 'DE', 'DEQAERTS']

        # box_plot_metrics(mse_dict, keys_to_plot, 'MSE')
        # box_plot_metrics(ssim_dict, keys_to_plot, 'SSIM')
        # box_plot_metrics(ncc_dict, keys_to_plot, 'NCC')
        # box_plot_metrics(ed_dict, keys_to_plot, 'ED')
        # box_plot_metrics(pa_dict, keys_to_plot, 'PA')
        # box_plot_metrics(procrustes_dict, keys_to_plot, 'Procrustes')
        # box_plot_metrics(midpoint_dict, keys_to_plot, 'Midpoint')
        # box_plot_metrics(scale_dict, keys_to_plot, 'Scale')

        # for key in ed_dict:
        #     method_name = key
        #     try:
        #         make_spread(epi_dict[key], ed_dict[key], pa_dict[key], ncc_dict[key], ssim_dict[key], mse_dict[key], procrustes_dict[key], midpoint_dict[key], scale_dict[key], method_name)
        #     except:
        #             print('Error in making spread for {}'.format(key))


    def matrix_distance(self, pts):
        reference = np.zeros((pts.shape[0],3))  #N, 3
        
        return np.sqrt(np.sum((pts-reference)**2, axis=-1, keepdims=False))
    


#weeksage
dropoutrate = float(sys.argv[1])
weeksage = int(sys.argv[2])
save_folder = str(sys.argv[3])
save_folder_update = save_folder + "_stuff"
os.makedirs(f"{save_folder_update}/Figures", exist_ok=True)

predictor = Predictor(weeksage=weeksage,dropoutrate=dropoutrate)

img_path = 'test'.format(weeksage) #Please contact authors for this sensitive data#
files_selected = glob.glob(os.path.join(img_path, '*.mat'))
predictor.predict(files_selected)




