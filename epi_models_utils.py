import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.parametrizations import spectral_norm
from geometry import *


# from transformer import TransformerEncoderLayer
# from vit import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
from spectral_norm_fc import spectral_norm_fc

FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]


SNPARAMS = {
    'coeff': 0.95,
    'n_power_iterations':1
}


class VGGAXISGNLL(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGAXISGNLL, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        

        self.axis = nn.Linear(fc_size,3+3+1+9)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x



        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))
        
        variance = s4.view(B,3,3).permute(1,0,2)
        variance = F.softplus(variance) + 1e-6
        
        return x, axis_pts, variance
    
class VGGQUATGNLL(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGQUATGNLL, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+9)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 9], dim=1) #quat
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        variance = s2.view(B,3,3).permute(1,0,2)
        variance = F.softplus(variance) + 1e-6
        
        return x, quat_pts, variance

class VGGMATGNLL(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGMATGNLL, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)

        self.s03 = nn.Linear(fc_size,9+3+1+9)

                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1, 9], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        variance = s3.view(B,3,3).permute(1,0,2)
        variance = F.softplus(variance) + 1e-6
        
        return x, rot_pts, variance

class VGGEULERGNLL(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGEULERGNLL, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.euler = nn.Linear(fc_size,3+3+1+9)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x



        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        
        variance = s5.view(B,3,3).permute(1,0,2)
        variance = F.softplus(variance) + 1e-6
        
        return x, euler_pts, variance

class GNLLMI(nn.Module):
       
    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(GNLLMI, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.out1 = nn.Linear(fc_size,18)
    
    def forward(self, input):
        ensemble_num,B,C,H,W = input.size()

        input = input.transpose(1, 0).view(
            B, ensemble_num, H, W
        )  
        feat = self.features(input)

        # print('feat', feat.shape)

        x = self.avgpool(feat)

        # print(x.shape, 'average pooling')

        x = x.view(x.size(0), -1)
        
        x = self.feature_fc(x)
        
        # print('output x', x.shape)    

        mean1, variance1 = torch.split(self.out1(x), 9, dim=1)
        variance1 = F.softplus(variance1) + 1e-6
        
        return feat, mean1.view(B,3,3).permute(1,0,2), variance1.view(B,3,3).permute(1,0,2)


class GNLLMIMO5(nn.Module):
       
    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(GNLLMIMO5, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.out1 = nn.Linear(fc_size,18)
        self.out2 = nn.Linear(fc_size,18)
        self.out3 = nn.Linear(fc_size,18)


    def forward(self, input):
        ensemble_num,B,C,H,W = input.size()

        input = input.transpose(1, 0).view(
            B, ensemble_num, H, W
        )  
        feat = self.features(input)

        # print('feat', feat.shape)

        x = self.avgpool(feat)

        # print(x.shape, 'average pooling')

        x = x.view(x.size(0), -1)
        
        x = self.feature_fc(x)
        
        # print('output x', x.shape)    

        mean1, variance1 = torch.split(self.out1(x), 9, dim=1)
        mean2, variance2 = torch.split(self.out2(x), 9, dim=1)
        mean3, variance3 = torch.split(self.out3(x), 9, dim=1)
        # mean4, variance4 = torch.split(self.out4(x), [9, 9], dim=1)
        # mean5, variance5 = torch.split(self.out5(x), [9, 9], dim=1)

        
        stacked_points = torch.stack([mean1.view(B,3,3).permute(1,0,2), mean2.view(B,3,3).permute(1,0,2),mean3.view(B,3,3).permute(1,0,2)])
        stacked_variances = torch.stack([variance1.view(B,3,3).permute(1,0,2), variance2.view(B,3,3).permute(1,0,2),variance3.view(B,3,3).permute(1,0,2)])

        # stacked_points = torch.stack([mean1.view(B,3,3).permute(1,0,2), mean2.view(B,3,3).permute(1,0,2),mean3.view(B,3,3).permute(1,0,2),mean4.view(B,3,3).permute(1,0,2), mean5.view(B,3,3).permute(1,0,2)])
        # stacked_variances = torch.stack([variance1.view(B,3,3).permute(1,0,2), variance2.view(B,3,3).permute(1,0,2),variance3.view(B,3,3).permute(1,0,2),variance4.view(B,3,3).permute(1,0,2),variance5.view(B,3,3).permute(1,0,2)])
        stacked_variances = F.softplus(stacked_variances) + 1e-6
        
        return feat, stacked_points, stacked_variances

class VGGROTGNLLMIMO(nn.Module):
       
    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGROTGNLLMIMO, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,(8+9))
        self.s03 = nn.Linear(fc_size,(9+3+1+9))
        self.axis = nn.Linear(fc_size,(3+3+1+9))
        self.euler = nn.Linear(fc_size,(3+3+1+9))
        self.scale = nn.Linear(fc_size, (9+9))
                
    def forward(self, input):
        ensemble_num,B,C,H,W = input.size()

        input = input.transpose(1, 0).view(
            B, ensemble_num, H, W
        )  

        # print('input', input.shape)

        feat = self.features(input)

        # print('features', feat.shape)
        
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)


        ref_pts, s1 =  torch.split(self.scale(x), [9,9], dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 9], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1, 9], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        stacked_points = torch.stack([ref_pts.view(B,3,3).permute(1,0,2), quat_pts, rot_pts, axis_pts, euler_pts])
        stacked_variances = torch.stack([s1.view(B,3,3).permute(1,0,2), s2.view(B,3,3).permute(1,0,2), s3.view(B,3,3).permute(1,0,2), s4.view(B,3,3).permute(1,0,2), s5.view(B,3,3).permute(1,0,2)])
        stacked_variances = F.softplus(stacked_variances) + 1e-6
        
        return feat, stacked_points, stacked_variances

def make_layers_instance_norm_mimo(norm=True):
    layers = []
    in_channels = 2
    for v in FILTER_SIZE:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=1)
            if norm:
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class GNLLMIMO(nn.Module):
       
    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(GNLLMIMO, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.out1 = nn.Linear(fc_size,18)
        self.out2 = nn.Linear(fc_size,18)

    
    def forward(self, input):
        ensemble_num,B,C,H,W = input.size()

        input = input.transpose(1, 0).view(
            B, ensemble_num, H, W
        )  
        feat = self.features(input)

        # print('feat', feat.shape)

        x = self.avgpool(feat)

        # print(x.shape, 'average pooling')

        x = x.view(x.size(0), -1)
        
        x = self.feature_fc(x)
        
        # print('output x', x.shape)    

        mean1, variance1 = torch.split(self.out1(x), 9, dim=1)
        mean2, variance2 = torch.split(self.out2(x), 9, dim=1)

        # mean4, variance4 = torch.split(self.out4(x), [9, 9], dim=1)
        # mean5, variance5 = torch.split(self.out5(x), [9, 9], dim=1)

        stacked_points = torch.stack([mean1.view(B,3,3).permute(1,0,2), mean2.view(B,3,3).permute(1,0,2)])
        stacked_variances = torch.stack([variance1.view(B,3,3).permute(1,0,2), variance2.view(B,3,3).permute(1,0,2)])

        # stacked_points = torch.stack([mean1.view(B,3,3).permute(1,0,2), mean2.view(B,3,3).permute(1,0,2),mean3.view(B,3,3).permute(1,0,2),mean4.view(B,3,3).permute(1,0,2), mean5.view(B,3,3).permute(1,0,2)])
        # stacked_variances = torch.stack([variance1.view(B,3,3).permute(1,0,2), variance2.view(B,3,3).permute(1,0,2),variance3.view(B,3,3).permute(1,0,2),variance4.view(B,3,3).permute(1,0,2),variance5.view(B,3,3).permute(1,0,2)])
        stacked_variances = F.softplus(stacked_variances) + 1e-6
        
        return feat, stacked_points, stacked_variances

class VGGROTGNLLSumMean(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGROTGNLLSumMean, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+9)
        self.s03 = nn.Linear(fc_size,9+3+1+9)
        self.axis = nn.Linear(fc_size,3+3+1+9)
        self.euler = nn.Linear(fc_size,3+3+1+9)
        self.scale = nn.Linear(fc_size, 9+9)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        ref_pts, s1 =  torch.split(self.scale(x), [9,9], dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 9], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1, 9], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        stacked_points = torch.stack([ref_pts.view(B,3,3).permute(1,0,2), quat_pts,rot_pts, axis_pts, euler_pts])
        stacked_variances = torch.stack([s1.view(B,3,3).permute(1,0,2), s2.view(B,3,3).permute(1,0,2), s3.view(B,3,3).permute(1,0,2), s4.view(B,3,3).permute(1,0,2), s5.view(B,3,3).permute(1,0,2)])
        stacked_variances = F.softplus(stacked_variances) + 1e-6
         
        mean = torch.mean(stacked_points, dim=0)
        variance = (stacked_points + stacked_points**2).mean(dim=0) - mean**2
        variance = F.softplus(variance) + 1e-6


        return x, stacked_points, stacked_variances, mean, variance



class VGGROTGNLLSum(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGROTGNLLSum, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+9)
        self.s03 = nn.Linear(fc_size,9+3+1+9)
        self.axis = nn.Linear(fc_size,3+3+1+9)
        self.euler = nn.Linear(fc_size,3+3+1+9)
        self.scale = nn.Linear(fc_size, 9+9)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        ref_pts, s1 =  torch.split(self.scale(x), [9,9], dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 9], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1, 9], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        mean_out = torch.stack([ref_pts.view(B,3,3).permute(1,0,2), quat_pts,rot_pts, axis_pts, euler_pts])
        variances = torch.stack([s1.view(B,3,3).permute(1,0,2), s2.view(B,3,3).permute(1,0,2), s3.view(B,3,3).permute(1,0,2), s4.view(B,3,3).permute(1,0,2), s5.view(B,3,3).permute(1,0,2)])
        variances = F.softplus(variances) + 1e-6

        means = torch.mean(mean_out, dim=0)

        return x, means, variances, mean_out



class VGGLWQUAT9(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGLWQUAT9, self).__init__()
        
        self.unc = 9
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+self.unc)
        self.scale = nn.Linear(fc_size, 9+self.unc)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        scale, s1 =  torch.split(self.scale(x), [9, self.unc], dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, self.unc], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

       
        # un = self.globalavgpool(feat)
        # s1 = F.softplus(s1) + 1e-6
        # s2 = F.softplus(s2) + 1e-6
        
        return x, scale, quat_pts, s1, s2
    
class VGGLWROT9(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGLWROT9, self).__init__()

        self.unc = 9
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+self.unc)
        self.s03 = nn.Linear(fc_size,9+3+1+self.unc)
        self.axis = nn.Linear(fc_size,3+3+1+self.unc)
        self.euler = nn.Linear(fc_size,3+3+1+self.unc)
        self.scale = nn.Linear(fc_size, 9+self.unc)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        ref_pts, s1 =  torch.split(self.scale(x), 9, dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, self.unc], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1,self.unc], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, self.unc], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, self.unc], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))
        
        return x, ref_pts, quat_pts, rot_pts, axis_pts, euler_pts, s1, s2, s3, s4, s5
    


class VGGLWQUAT(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGLWQUAT, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+1)
        self.scale = nn.Linear(fc_size, 9+1)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        scale, s1 =  torch.split(self.scale(x), 9, dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 1], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

       
        # un = self.globalavgpool(feat)
        # s1 = F.softplus(s1) + 1e-6
        # s2 = F.softplus(s2) + 1e-6
        
        return x, scale, quat_pts, s1, s2
    

class VGGROT(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGROT, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,4)
        self.s03 = nn.Linear(fc_size,9)
        self.axis = nn.Linear(fc_size,3)
        self.euler = nn.Linear(fc_size,3)
        self.scale = nn.Linear(fc_size, 9)
        
        self.translation = nn.Linear(fc_size,3)
        self.sc = nn.Linear(fc_size, 1)

                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        ref_pts = self.scale(x) #ref
        translation_og = self.translation(x)
        scale_og = self.sc(x)


        quat = self.quat(x) #quat
        euler_angles = self.euler(x) #euler
        axis_angles = self.axis(x) #axis
        rot_matrix_pred = self.s03(x) #rot

        
       
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
        scale = scale_og.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation_og.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=scale_og.dtype, device=scale_og.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)

        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = scale*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        return x, ref_pts, quat_pts, rot_pts, axis_pts, euler_pts
    
class VGGLWROT(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGLWROT, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+1)
        self.s03 = nn.Linear(fc_size,9+3+1+1)
        self.axis = nn.Linear(fc_size,3+3+1+1)
        self.euler = nn.Linear(fc_size,3+3+1+1)
        self.scale = nn.Linear(fc_size, 9+1)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        ref_pts, s1 =  torch.split(self.scale(x), 9, dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 1], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1, 1], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, 1], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, 1], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))


        
        return x, ref_pts, quat_pts, rot_pts, axis_pts, euler_pts, s1, s2, s3, s4, s5
    

class VGGROTGNLL(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(VGGROTGNLL, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.quat = nn.Linear(fc_size,8+9)
        self.s03 = nn.Linear(fc_size,9+3+1+9)
        self.axis = nn.Linear(fc_size,3+3+1+9)
        self.euler = nn.Linear(fc_size,3+3+1+9)
        self.scale = nn.Linear(fc_size, 9+9)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x

        ref_pts, s1 =  torch.split(self.scale(x), [9,9], dim=1) #ref
        quat, translation, alpha, s2 = torch.split(self.quat(x), [4, 3, 1, 9], dim=1) #quat
        
        #Quaternion
        quat = F.normalize(quat, p=2, dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        alpha = alpha.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        translation = translation.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        matrix_temp = torch.tensor(((1,0,0,0),(0,1,0,0),(0,0,1,0)), dtype=alpha.dtype, device=alpha.device).unsqueeze(0).repeat(B, 1, 1)
        grid_temp = F.affine_grid(matrix_temp, (B,1,1,H,W)).squeeze(1)
        my_grid = quaternion_apply(quat, grid_temp)
        my_grid = alpha*my_grid+translation
        quat_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Rotation
        rot_matrix_pred, translation, alpha, s3 = torch.split(self.s03(x), [9, 3, 1, 9], dim=1) #quat
        rot_matrix_pred = rot_matrix_pred.view(B,3,3).permute(0,1,2)
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Rotation', rot_matrix_pred.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot_matrix_pred)
        # print('Pred Rotation w/ scale', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        rot_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Axis
        axes, translation, alpha, s4 = torch.split(self.axis(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Axis Angles', axis_angles.shape)
        rot = axisAngletoRotation_torch_nerf(axes)
        # print('Axis Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)
        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        axis_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        #Euler
        euler_angles, translation, alpha, s5 = torch.split(self.euler(x), [3, 3, 1, 9], dim=1) #quat
        alpha = alpha.unsqueeze(-1)*torch.eye(3, dtype=alpha.dtype, device=alpha.device).repeat(B, 1, 1)  #B,3,3
        # print('Pred Scale Angles', scale.shape)
        # print('Euler Angles', euler_angles.shape)
        rot = eulerAnglesToRotationMatrix_torch_jay(euler_angles)
        # print('Euler Matrix', rot.shape)
        pred_rotation = torch.einsum('bij,bjk->bik', alpha, rot)
        # print('Pred Rotation', pred_rotation.shape)
        translation = (translation).unsqueeze(-1)
        # print('PredTranslation', translation.shape)
        pred_rotation = torch.cat((pred_rotation, translation), dim=-1)
        # print('Pred Rotation Updated', pred_rotation.shape)

        my_grid = F.affine_grid(pred_rotation, (B,1,1,H,W)).squeeze(1)
        euler_pts = torch.stack((my_grid[:,0,0],my_grid[:,0,-1],my_grid[:,-1,0]))

        ens = torch.stack([ref_pts.view(B,3,3).permute(1,0,2),quat_pts,rot_pts, axis_pts, euler_pts])
        means = torch.stack([ref_pts, torch.reshape(quat_pts, (quat_pts.shape[1], 9)), torch.reshape(rot_pts, (rot_pts.shape[1], 9)), torch.reshape(axis_pts, (axis_pts.shape[1], 9)), torch.reshape(euler_pts, (euler_pts.shape[1], 9))])
        mean = torch.mean(means, dim=0)
        variances = torch.stack([s1, s2, s3, s4, s5])
        variance = (variances + means**2).mean(dim=0) - mean**2
        
        variance = F.softplus(variance) + 1e-6
        
        return x, mean, variance, ens


class DeepEnsembleQAERTS(nn.Module):
    def __init__(self, features, num_classes=9, fc_size=512, init_weights=True, device='', dropout=0, num_ensemble=5):
        super(DeepEnsembleQAERTS, self).__init__()

        self.ensemble = num_ensemble
        for i in range(num_ensemble):
            setattr(self, 'model'+str(i), VGGROTGNLL(features, num_classes, fc_size, init_weights, device, dropout))

    def forward(self, x):
        B,C,H,W = x.size()
        means = []
        variances = []
        feat_combined = []
        for i in range(self.ensemble):
            model = getattr(self, 'model'+str(i))
            feat, mean, variance, _ = model(x)
            feat_combined.append(feat)
            means.append(mean)
            variances.append(variance)
        feat_combined = torch.stack(feat_combined)
        means = torch.stack(means)
        mean = torch.mean(means, dim=0)
        variances = torch.stack(variances)
        variance = (variances + means**2).mean(dim=0) - mean**2
        return feat_combined, mean, variance


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    # elif isinstance(m, nn.BatchNorm1d):
    #     nn.init.normal_(m.weight.data, mean=1, std=0.02)
    #     nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class RPRMC_VGG(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(RPRMC_VGG, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        #Two head branches
        self.emd = mean_var(fc_size,fc_size) # embedding of z  

        self.var = nn.Sequential(nn.Linear(fc_size,fc_size), nn.BatchNorm1d(fc_size, eps=0.001, affine=False)) # variance of z

        self.scale = nn.Linear(fc_size,9)

        self.Tmax = 50

    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        

        emb = self.emd(x)
        logvar = self.var(x)
        sqrt_var = torch.exp(logvar * 0.5)
       
        preds = []
        if self.training:  # Only apply the trick during training
            rep_emb = emb[None].expand(self.Tmax, *emb.shape)
            rep_sqrt_var = sqrt_var[None].expand(self.Tmax, *sqrt_var.shape)
            norm_v = torch.randn_like(rep_sqrt_var)
            z = rep_emb + rep_sqrt_var * norm_v
            for i in range(self.Tmax):
                pred = self.scale(z[i])
                preds.append(pred)
            preds = torch.stack(preds)
        else:
            z = emb
            preds = self.scale(z)

        print('Printz', z.shape)

        return feat, preds, emb, logvar
    

class RPR_VGG(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(RPR_VGG, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        #Two head branches
        self.emd = mean_var(fc_size,fc_size) # embedding of z  

        self.var = nn.Sequential(nn.Linear(fc_size,fc_size), nn.BatchNorm1d(fc_size, eps=0.001, affine=False)) # variance of z

        self.scale = nn.Linear(fc_size,9)

    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        

        emb = self.emd(x)
        logvar = self.var(x)
        sqrt_var = torch.exp(logvar * 0.5)
       
        if self.training:  # Only apply the trick during training
            eps = torch.randn_like(sqrt_var)
            z = emb + eps * sqrt_var
        else:
            z = emb

        pred = self.scale(z)


        return feat, pred, emb, logvar
    
def mean_var(in_channel, out_channel):
    return nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(True)
        ) # embedding of z  

def make_layers_instance_norm(norm=True):
    layers = []
    in_channels = 1
    for v in FILTER_SIZE:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm:
                # layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_instance_norm_spectral_with_residual(norm=True):
    layers = []
    in_channels = 1
    for v in FILTER_SIZE:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            residual_block = ResidualBlock(in_channels, v, stride=1, kernel_size=3)
            if norm:
                layers += [residual_block, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [residual_block, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


#Evidential Deep Learning

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = torch.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return torch.mean(reg) if reduce else reg

# Custom loss function to handle the custom regularizer coefficient
def EvidentialRegressionLoss(true, pred):
        return EvidentialRegression(true, pred, coeff=1e-2)

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = torch.split(evidential_output, int(evidential_output.shape[-1]/4), dim=-1)
    loss_nll = NIG_NLL(y_true*80, gamma*80, v, alpha, beta)
    loss_reg = NIG_Reg(y_true*80, gamma*80, v, alpha, beta)
    return loss_nll + coeff * loss_reg

class DenseNormalGamma(nn.Module):
    def __init__(self, units_in, units_out):
        super(DenseNormalGamma, self).__init__()
        self.units_in = int(units_in)
        self.units_out = int(units_out)
        self.linear = nn.Linear(units_in, 4 * units_out)

    def evidence(self, x):
        softplus = nn.Softplus(beta=1)
        return softplus(x)

    def forward(self, x):
        output = self.linear(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units_out, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat(tensors=(mu, v, alpha, beta), dim=-1)

    def compute_output_shape(self):
        return (self.units_in, 4 * self.units_out)
    


class Evidential(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(Evidential, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.scale = DenseNormalGamma(fc_size, 9)
        # self.representation = None
                
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        scale = self.scale(x)

    
        return x, scale


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            spectral_norm_fc(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False), coeff=SNPARAMS['coeff'], n_power_iterations=SNPARAMS['n_power_iterations']),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            spectral_norm_fc(nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False), coeff=SNPARAMS['coeff'], n_power_iterations=SNPARAMS['n_power_iterations']),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                spectral_norm_fc(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),coeff=SNPARAMS['coeff'], n_power_iterations=SNPARAMS['n_power_iterations']),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

def make_layers_instance_norm_dropout(norm=True, dropout=0):
    layers = []
    in_channels = 1
    for v in FILTER_SIZE:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm:
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True), nn.Dropout(p=dropout)]
                # layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout(p=dropout)]
                # layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def feature_fc_dropout(in_channel, out_channel, dropout=0):

    return nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.Dropout(p=dropout),
                nn.ReLU(True),
                nn.Linear(out_channel, out_channel),
                nn.Dropout(p=dropout),
                nn.ReLU(True),
                )


def feature_fc(in_channel, out_channel, dropout=0):

    return nn.Sequential(
                nn.Linear(in_channel, out_channel),
                # nn.Dropout(p=dropout),
                nn.ReLU(True),
                nn.Linear(out_channel, out_channel),
                # nn.Dropout(p=dropout),
                nn.ReLU(True),
                )

#Multi-task models
class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, loss_list, device):
        super(WeightedMultiTaskLoss, self).__init__()
        self._loss_list  = loss_list
        self._sigmas_sq = []
        for i in range(len(self._loss_list)):
            param = nn.Parameter(torch.empty(1))
            nn.init.uniform_(param, 0.2, 1.0)
            param = param.to(device)
            self._sigmas_sq.append(param)
    
    def get_loss(self):
        factor = 1.0 / (2.0 * self._sigmas_sq[0])
        loss = (factor * self._loss_list[0]) + torch.log(self._sigmas_sq[0])
        for i in range(1, len(self._sigmas_sq)):
            factor = 1.0 / (2.0 * self._sigmas_sq[i])
            loss = loss + (factor * self._loss_list[i]) + torch.log(self._sigmas_sq[i])
        return loss

class MTL(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(MTL, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc_dropout(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.translation = nn.Linear(fc_size,3)
        self.quat = nn.Linear(fc_size,4)
        self.scale = nn.Linear(fc_size, 9)
        self.sc = nn.Linear(fc_size, 1)
        self.axial = nn.Linear(fc_size, 3)
        self.euler = nn.Linear(fc_size, 3)
        self.rotation = nn.Linear(fc_size,9)
        # self.representation = None
                
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x)
        
        translation = self.translation(x)
        quat = self.quat(x)
        sc = self.sc(x)
        scale = self.scale(x)
        axial = self.axial(x)
        euler = self.euler(x)
        rotation = self.rotation(x)
       

        return x, scale, quat, translation, axial, euler, rotation, sc

#MCD
class MCD(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(MCD, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc_dropout(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)

        self.scale = nn.Linear(fc_size, 18)

                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # x_single = self.single_fc(x)
        x = self.scale(x)
        mean, variance = torch.split(x, 9, dim=1)
        variance = F.softplus(variance) + 1e-6
        return feat, mean, variance


class MVE(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(MVE, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        # self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)

        self.scale = nn.Linear(fc_size,18)

    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # x_single = self.single_fc(x)
        x = self.scale(x)
        mean, variance = torch.split(x, 9, dim=1)

        variance = F.softplus(variance) + 1e-6
        return feat, mean, variance
    
class DeepEnsemble(nn.Module):
    def __init__(self, features, num_classes=9, fc_size=512, init_weights=True, device='', dropout=0, num_ensemble=5):
        super(DeepEnsemble, self).__init__()

        self.ensemble = num_ensemble
        for i in range(num_ensemble):
            setattr(self, 'model'+str(i), MVE(features, num_classes, fc_size, init_weights, device, dropout))

    def forward(self, x):
        B,C,H,W = x.size()
        means = []
        variances = []
        feat_combined = []
        for i in range(self.ensemble):
            model = getattr(self, 'model'+str(i))
            feat, mean, variance = model(x)
            feat_combined.append(feat)
            means.append(mean)
            variances.append(variance)
        feat_combined = torch.stack(feat_combined)
        means = torch.stack(means)
        mean = mean.mean(dim=0)
        variances = torch.stack(variances)
        variance = (variances + means**2).mean(dim=0) - mean**2
        return feat_combined, mean, variance
    
class Simple_VGG_GAP(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(Simple_VGG_GAP, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.scale = nn.Linear(fc_size,9)
        
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        scale = self.scale(x)
        un = self.globalavgpool(feat)
        un = un.view(un.size(0), -1)

        return feat, scale, un
    


class Baseline_simple_vgg_backbone(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(Baseline_simple_vgg_backbone, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.scale = nn.Linear(fc_size, 9)
        # self.representation = None
                
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        

        
        scale = self.scale(x)

       

        return x, scale


class MTL_Learned(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(MTL_Learned, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc_dropout(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.translation = nn.Linear(fc_size,3)
        self.quat = nn.Linear(fc_size,4+1)
        self.scale = nn.Linear(fc_size, 9+1)
        self.sc = nn.Linear(fc_size, 1)
        self.axial = nn.Linear(fc_size, 3+1)
        self.euler = nn.Linear(fc_size, 3+1)
        self.rotation = nn.Linear(fc_size,9+1)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x)
        
        translation = self.translation(x)
      
        sc = self.sc(x) 
        scale, s1 =  torch.split(self.scale(x), 9, dim=1) #ref
        quat, s2 = torch.split(self.quat(x), 4, dim=1) #quat
        rotation, s3 = torch.split(self.rotation(x), 9, dim=1) #rot
        axial, s4 =  torch.split(self.axial(x), 3, dim=1) #axis
        euler, s5 = torch.split(self.euler(x), 3, dim=1) #euler

        # un = self.globalavgpool(feat)
        s1 = F.softplus(s1) + 1e-6
        s2 = F.softplus(s2) + 1e-6
        s3 = F.softplus(s3) + 1e-6
        s4 = F.softplus(s4) + 1e-6
        s5 = F.softplus(s5) + 1e-6    
        
        return x, scale, quat, translation, axial, euler, rotation, sc, s1, s2, s3, s4, s5


class Baseline_simple_vgg_learned_weights_mtquat(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(Baseline_simple_vgg_learned_weights_mtquat, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.translation = nn.Linear(fc_size,3)
        self.sc = nn.Linear(fc_size, 1)


        self.quat = nn.Linear(fc_size,4+1)
        self.scale = nn.Linear(fc_size, 9+1)
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # self.representation = x.clone().detach()
        # # x_single = self.single_fc(x)
        
        translation = self.translation(x)
      
        sc = self.sc(x) 

        scale, s1 =  torch.split(self.scale(x), 9, dim=1) #ref
        quat, s2 = torch.split(self.quat(x), 4, dim=1) #quat
        # un = self.globalavgpool(feat)
        # s1 = F.softplus(s1) + 1e-6
        # s2 = F.softplus(s2) + 1e-6
        
        return x, scale, quat, translation, sc, s1, s2
