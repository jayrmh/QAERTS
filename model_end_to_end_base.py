import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from geometry import *

# from transformer import TransformerEncoderLayer
# from vit import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


#FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]

       
class Baseline_simple_vgg_scaling(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(Baseline_simple_vgg_scaling, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.translation = nn.Linear(fc_size,3)
        self.quat = nn.Linear(fc_size,4)
        self.scale = nn.Linear(fc_size, 9)
        self.sc = nn.Linear(fc_size, 1)
                
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # x_single = self.single_fc(x)
        
        translation = self.translation(x)
        quat = self.quat(x)
        scale = self.scale(x)
        sc = self.sc(x)

        return feat, scale, quat, translation, sc



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
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
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

def feature_fc(in_channel, out_channel, dropout=0):

    return nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.Dropout(p=dropout),
                nn.ReLU(True),
                nn.Linear(out_channel, out_channel),
                nn.Dropout(p=dropout),
                nn.ReLU(True),
                )

def feature_conv(in_channel, out_channel, dropout=0):
    
    return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.ReLU(True),
                # nn.Dropout(p=dropout),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0),
                nn.ReLU(True),
                # nn.Dropout(p=dropout),
            )

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResidualBlock_instance(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock_instance, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.InstanceNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(outchannel)
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out
    
class ResNet18_instance(nn.Module):
    def __init__(self, in_ch=1, st_ch=64):
        super(ResNet18_instance, self).__init__()
        self.inchannel = st_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, self.inchannel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(self.inchannel),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.layer1 = self.make_layer(ResidualBlock_instance, self.inchannel,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock_instance, self.inchannel*2, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock_instance, self.inchannel*2, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock_instance, self.inchannel*2, 2, stride=2)
        
        self.feature_fc = feature_fc(self.inchannel * 10 * 10, self.inchannel)
        
        self.translation = nn.Linear(self.inchannel,3)
        self.quat = nn.Linear(self.inchannel,4)
        self.scale = nn.Linear(self.inchannel, 1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        x = out.view(out.size(0), -1)
        
        x = self.feature_fc(x)
        
        # x_single = self.single_fc(x)
        
        
        translation = self.translation(x)
        quat = self.quat(x)
        scale = self.scale(x)
        
        return out, scale, quat, translation
        
    

       
class Baseline_simple_vgg(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device='', dropout=0):
        super(Baseline_simple_vgg, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))

        self.feature_fc = feature_fc(FILTER_SIZE[-2] * 10 * 10, fc_size, dropout)
        
        self.translation = nn.Linear(fc_size,3)
        self.quat = nn.Linear(fc_size,4)
        self.scale = nn.Linear(fc_size, 9)
                
                
    def forward(self, input):
        B,C,H,W = input.size()
        
        feat = self.features(input)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        
        # x_single = self.single_fc(x)
        
        
        translation = self.translation(x)
        quat = self.quat(x)
        scale = self.scale(x)
        
        return feat, scale, quat, translation


class No_name(nn.Module):

    def __init__(self, device=''):
        super(No_name, self).__init__()
        self.device = device
        self.small = 1e-7
        self.start_channel = 64
        self.bottleneck_dim = self.start_channel*8
        
        # self.pose_model = ResNet18_instance(st_ch=self.start_channel)
        self.pose_model = Baseline_simple_vgg(make_layers_instance_norm())
        self.reconstruct_model = self.conv1 = nn.Sequential(
            nn.Conv2d(self.bottleneck_dim, self.bottleneck_dim//2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.bottleneck_dim//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(self.bottleneck_dim//2, self.bottleneck_dim//4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.bottleneck_dim//4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(self.bottleneck_dim//4, self.bottleneck_dim//8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.bottleneck_dim//8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(self.bottleneck_dim//8, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
        )
        
        
        
    
    def sli_to_vol(self, feat_2d):
        '''
        feat_2d: B, C, H, W
        
        output: B, C, H, W, D
                B, 1, H, W, D
        '''
        B, C, H, W = feat_2d.size()
        
        feat_3d = feat_2d.unsqueeze(-3)
        
        zeros = torch.zeros((B, C, (H-1)//2, H, W)).to(self.device)
        zeros_mask = torch.zeros((B, 1, (H-1)//2, H, W)).to(self.device)
        ones = torch.ones((B, 1, 1, H, W)).to(self.device)
        
        feat_3d = torch.cat((zeros, feat_3d, zeros), dim=-3)
        
        mask = torch.cat((zeros_mask+self.small, ones, zeros_mask+self.small), dim=2)
        
        # feat_3d = torch.cat((feat_3d, mask), dim=1)
        
        return feat_3d, mask
    
    def transform_vol(self, feat_3d, mask, scale, rot, translation):
        '''
        feat_3d: B, C, H, W, D
        mask: B, 1, H, W, D
        scale: B, 1     grid_scale
        rot: B, 4
        translation: B, 3
        
        output: 1, C, H, W, D
        '''
        matrix_rot = quaternion_to_matrix(rot)
        # matrix_rot = eulerAnglesToRotationMatrix_torch(rot.unsqueeze(1)).squeeze(1)
        
        'get grid'
        matrix = combine_scale_rot_trans_inverse(scale.unsqueeze(1), matrix_rot.unsqueeze(1), translation.unsqueeze(1))     #B,1,3,4
        matrix = matrix.squeeze(1)
        grid = F.affine_grid(matrix, feat_3d.size())
        grid_mask = F.affine_grid(matrix, mask.size())
        # grid = torch.cat((grid[...,2:],grid[...,1:2],grid[...,0:1]), dim=-1)
        
        
        
        'transform'
        feat_transform = F.grid_sample(feat_3d, grid)   #B, C, H, W, D
        mask_transform = F.grid_sample(mask, grid_mask, padding_mode='border')   #B, 1, H, W, D
        
        'Mean'
        feat_transform = feat_transform.sum(0)
        mask_transform = mask_transform.sum(0)
        
        feat_transform = (feat_transform/mask_transform).unsqueeze(0)   #1, C, H, W, D
        
        return feat_transform
        
        
    def sample_slice(self, cost_volume, scale, rot, translation):
        '''
        cost_volume: 1, C, H, W, D
        scale: B, 1
        rot: B, 4
        translation: B, 3
        '''
        _, C, H, W, D = cost_volume.size()
        B,_ = scale.size()
        
        matrix_rot = quaternion_to_matrix(rot)
        # matrix_rot = eulerAnglesToRotationMatrix_torch(rot.unsqueeze(1)).squeeze(1)
        
        'get grid'
        matrix = combine_scale_rot_trans(scale.unsqueeze(1), matrix_rot.unsqueeze(1), translation.unsqueeze(1))     #B,1,3,4
        matrix = matrix.squeeze(1)  #B,3,4
        grid = F.affine_grid(matrix, (B,C,1,H,W))   
        
        'sample slice features'
        slices = [F.grid_sample(cost_volume, grid[s:s+1]) for s in range(B)]
        slices = torch.cat(slices, dim=0)   #B, C, 1, H, W
        
        return slices.squeeze(2)
    
         
    
    def forward(self, x, finetune = False):
        ''
        B, C, H, W = x.size()
        
        # if finetune:
        #     B=int(B*2)
        
        feat_2d, scale, quat, translation = self.pose_model(x)  #grid_scale
        feat_3d, mask = self.sli_to_vol(feat_2d[0:B//2])
        
        cost_volume = self.transform_vol(feat_3d, mask, scale[0:B//2], quat[0:B//2], translation[0:B//2]/(H//2))
        
        slices = self.sample_slice(cost_volume, scale, quat, translation/(H//2))
        
        slices = self.reconstruct_model(slices)
        
        return slices, scale, quat, translation
        
        