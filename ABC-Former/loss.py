import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
import math
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class L2_histo(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, y):
#         # input has dims: (Batch x Bins)
#         bins = x.size(1)
#         r = torch.arange(bins)
#         s, t = torch.meshgrid(r, r)
#         tt = t >= s
#         tt = tt.to(device)

#         cdf_x = torch.matmul(x, tt.float())
#         cdf_y = torch.matmul(y, tt.float())

#         return torch.sum(torch.square(cdf_x - cdf_y), dim=1)

def L2_histo(x, y):
    bins = x.size(1)
    r = torch.arange(bins)
    s, t = torch.meshgrid(r, r)
    tt = t >= s
    tt = tt.to(device)

    cdf_x = torch.matmul(x, tt.float())
    cdf_y = torch.matmul(y, tt.float())

    return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space """
    
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)

import torch


class mae_loss():
    def compute(output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
class VGG19_Content(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_Content, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'1': 'relu1_1', '3': 'relu1_2', '6': 'relu2_1', '8': 'relu2_2'} # may add other layers    
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def forward(self, pred, true, layer):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)

        return torch.mean((true_f[layer]-pred_f[layer])**2)
    
def calc_mae(source, target):

  source1 = np.reshape(source, [-1,1]).astype(np.float32)
  target1 = np.reshape(target, [-1,1]).astype(np.float32)
  
  if len(source1)%3 == 1:
     source1=np.append(source1,[0])
     source1=np.append(source1,[0])
  if len(source1)%3 == 2:
     source1=np.append(source1,[0])
     
  if len(target1)%3 == 1:
     target1=np.append(target1,[0])
     target1=np.append(target1,[0])
  if len(target1)%3 == 2:
     target1=np.append(target1,[0])
     

  source1 = np.reshape(source1, [-1, 3]).astype(np.float32)
  target1 = np.reshape(target1, [-1, 3]).astype(np.float32)
  source_norm = np.sqrt(np.sum(np.power(source1, 2), 1))
  target_norm = np.sqrt(np.sum(np.power(target1, 2), 1))
  norm = source_norm * target_norm
  L = np.shape(norm)[0]
  inds = norm != 0
  angles = np.sum(source1[inds, :] * target1[inds, :], 1) / norm[inds]

  angles[angles > 1] = 1
  f = np.arccos(angles)
  f[np.isnan(f)] = 0
  f = f * (180 / np.pi)
  safe_v = 0.999999
  vec1 = torch.from_numpy(source1)
  vec2 = torch.from_numpy(target1)

  illum_normalized1 = torch.nn.functional.normalize(vec1,dim=1)
  illum_normalized2 = torch.nn.functional.normalize(vec2,dim=1)

  dot = torch.sum(illum_normalized1*illum_normalized2,dim=1)
  dot = torch.clamp(dot, -safe_v, safe_v)
  angle = torch.acos(dot)*(180/math.pi)

  loss = torch.mean(angle)
  return sum(f)/L 


def calc_mse(image1, image2):

    mse = np.mean((image1/1.0 - image2/1.0) ** 2)

    return mse




# def _convert_output_type_range(img, dst_type):
#     if dst_type not in (np.uint8, np.float32):
#         raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
#     if dst_type == np.uint8:
#         img = img.round()
#     else:
#         img /= 255.
#     return img.astype(dst_type)

# def _convert_input_type_range(img):
#     img_type = img.dtype
#     img = img.astype(np.float32)
#     if img_type == np.float32:
#         pass
#     elif img_type == np.uint8:
#         img /= 255.
#     else:
#         raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
#     return img

# def rgb2ycbcr(img, y_only=False):
#     img_type = img.dtype
#     img = _convert_input_type_range(img)
#     if y_only:
#         out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
#     else:
#         out_img = np.matmul(
#             img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],[24.966, 112.0, -18.214]]) + [16, 128, 128]
#     out_img = _convert_output_type_range(out_img, img_type)
#     return out_img

# def to_y_channel(img):
#     img = img.astype(np.float32) / 255.
#     if img.ndim == 3 and img.shape[2] == 3:
#         img = rgb2ycbcr(img, y_only=True)
#         img = img[..., None]
#     else:
#         raise ValueError(f'Wrong image shape [2]: {img.shape[2]}.')
#     return img * 255.

# def reorder_image(img, input_order='HWC'):
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
#     if len(img.shape) == 2:
#         img = img[..., None]
#     if input_order == 'CHW':
#         img = img.transpose(1, 2, 0)
#     return img
# def calculate_mse(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
#     assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img1 = reorder_image(img1, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)

#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

#     if test_y_channel:
#         img1 = to_y_channel(img1)
#         img2 = to_y_channel(img2)

#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return mse