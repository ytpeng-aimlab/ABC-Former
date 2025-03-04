import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import random

def histogram_loaderpatch_BGR(image):
    R_hist, R_bins = np.histogram(image[:, :, 2], bins=256, range=(0, 256)) # R_hist.shape = (256,)
    G_hist, G_bins = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(image[:, :, 0], bins=256, range=(0, 256))

    R_pdf = R_hist/sum(R_hist)
    G_pdf = G_hist/sum(G_hist)
    B_pdf = B_hist/sum(B_hist)
    BGR = np.vstack((B_pdf,G_pdf,R_pdf))
    return BGR

def RGBtoLab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_channel, a_channel, b_channel = cv2.split(lab_image)
    norm_L_channel = L_channel
    norm_a_channel = a_channel
    norm_b_channel = b_channel
    return cv2.merge((norm_L_channel, norm_a_channel, norm_b_channel)) #hwc


class BasicDataset_BGR(Dataset):
    def __init__(self, imgs_dir, fold=0, patch_size=128, patch_num_per_image=1, max_trdata=12000):

        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num_per_image = patch_num_per_image
        # get selected training data based on the current fold

        if fold == 0:
            logging.info(f'Training process will use {max_trdata} training images randomly selected from all training data')
            logging.info('Loading training images information...')
            self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir)
                        if not file.startswith('.')]
        else:
            logging.info(f'There is no fold {fold}! Training process will use all training data.')

        if max_trdata != 0 and len(self.imgfiles) > max_trdata:
            print('>12000')
            random.shuffle(self.imgfiles)
            self.imgfiles = self.imgfiles[0:max_trdata]
            for i in range(len(self.imgfiles)):
                with open(os.path.join('./checkpoints/mywork1', 'paperdataset.txt'), 'a') as f:
                   f.write(str(self.imgfiles[i])+'\n')
                   f.close()
        logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):
        if flip_op == 1:
            pil_img = np.flip(pil_img, axis=1)
        elif flip_op == 2:
            pil_img = np.flip(pil_img,axis=0)

        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans

        return img_trans

    def __getitem__(self, i):
        gt_ext = ('G_AS.png','G_AS.png')
        img_file = self.imgfiles[i]
        in_img = cv2.imread(img_file)
        # get image size
        w, h = in_img.shape[1],in_img.shape[0]
        # get ground truth images
        parts = img_file.split('_')
        base_name = ''
        for i in range(len(parts) - 2):
            base_name = base_name + parts[i] + '_'
        gt_awb_file = base_name + gt_ext[0]
        parts = gt_awb_file.split('/')
        parts[-2] = 'gt'
        gt_awb_file = '/'.join(parts)
        gt_img = cv2.imread(gt_awb_file)
        # get flipping option
        flip_op = np.random.randint(3)
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op) #chw
        img_patches = in_img_patches.transpose((1,2,0)) #hwc
        img_patches_hist = histogram_loaderpatch_BGR(img_patches)
        img_patches_lab = histogram_loaderpatch_BGR(RGBtoLab(img_patches))
        gt_img_patches = self.preprocess(gt_img, self.patch_size, (patch_x, patch_y), flip_op)
        gt_patches = gt_img_patches.transpose((1,2,0))
        gt_patches_hist = histogram_loaderpatch_BGR(gt_patches)
        gt_patches_lab = histogram_loaderpatch_BGR(RGBtoLab(gt_patches))


        for j in range(self.patch_num_per_image - 1):
            # get flipping option
            flip_op = np.random.randint(3)
            # get random patch coord
            patch_x = np.random.randint(0, high=w - self.patch_size)
            patch_y = np.random.randint(0, high=h - self.patch_size)
            temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op) #chw
            temp_img = temp.transpose((1,2,0)) #hwc
            temp_hist = histogram_loaderpatch_BGR(temp_img)
            temp_lab = histogram_loaderpatch_BGR(RGBtoLab(temp_img))
            in_img_patches = np.append(in_img_patches, temp, axis=0)
            img_patches_hist = np.append(img_patches_hist, temp_hist, axis=0)
            img_patches_lab = np.append(img_patches_lab, temp_lab, axis=0)
            temp = self.preprocess(gt_img, self.patch_size, (patch_x, patch_y), flip_op)
            temp_gt = temp.transpose((1,2,0))
            temp_gt_hist = histogram_loaderpatch_BGR(temp_gt)
            temp_gt_lab = histogram_loaderpatch_BGR(RGBtoLab(temp_gt))
            gt_img_patches = np.append(gt_img_patches, temp, axis=0)
            gt_patches_hist = np.append(gt_patches_hist, temp_gt_hist, axis=0)
            gt_patches_lab = np.append(gt_patches_lab, temp_gt_lab, axis=0)

        return {'image':(in_img_patches), 'gt_AWB':(gt_img_patches),'image_hist':(img_patches_hist),'gt_hist':(gt_patches_hist), 'image_lab':(img_patches_lab), 'gt_lab':(gt_patches_lab)}


