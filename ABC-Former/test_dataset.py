import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import io, transform

from torch.utils.data import Dataset, DataLoader
from os import path
from PIL import Image


## test cube data
test_cube_path = '/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/others_cube/output/'
test_cube_img_files = os.listdir(test_cube_path) #所有圖片的檔名
test_cube_img_path = [os.path.join("/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/others_cube/output/", i) for i in test_cube_img_files]
test_cube_gt_path = '/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/others_cube/gt/'

## test set1 data
test_set1_path = '/mnt/disk2/cyc202/awbformer/datasets/test/renderset1/input/Set1_with_cc21046/'
test_set1_img_files = os.listdir(test_set1_path) #所有圖片的檔名
test_set1_img_path = [os.path.join("/mnt/disk2/cyc202/awbformer/datasets/test/renderset1/input/Set1_with_cc21046/", i) for i in test_set1_img_files]
test_set1_gt_path = '/mnt/disk2/cyc202/awbformer/datasets/test/renderset1/gt/'

## test set2 data
set2_path = '/mnt/disk2/cyc202/awbformer/datasets/test/renderset2/images/'
set2_img_files = os.listdir(set2_path) #所有圖片的檔名
set2_img_path = [os.path.join("/mnt/disk2/cyc202/awbformer/datasets/test/renderset2/images/",i) for i in set2_img_files]
set2_gt_path = '/mnt/disk2/cyc202/awbformer/datasets/test/renderset2/gt/'

## test mixedWB data
mixedWB_path = '/mnt/disk2/roger/mywork3_W/data/mixed/images/'
mixedWB_img_files = os.listdir(mixedWB_path) #所有圖片的檔名
mixedWB_img_path = [os.path.join("/mnt/disk2/roger/mywork3_W/data/mixed/images/",i) for i in mixedWB_img_files]
mixedWB_gt_path = '/mnt/disk2/roger/mywork3_W/data/mixed/gt/'


def histogram_loader_BGRimage(image):
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
 

def get_testset_cube():
	test_data  = testset_cube()
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	return testloader

def get_test_set1():
	test_data  = testset_set1()
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	return testloader

def get_test_set2():
	test_data  = testset_set2()
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	return testloader

def get_testset_mix():
	test_data  = testsetmix()
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	return testloader


class testset_cube(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader_BGRimage
        self.lab_loader = RGBtoLab
        self.transform = transforms.ToTensor()

        self.images = test_cube_img_path
        self.label = test_cube_gt_path

    def __getitem__(self, index):
        single_img_np = self.images[index]        
        img_name = single_img_np.split('/')[-1].split('_')[0]
        single_label_np = self.label+img_name+'.JPG'
        input_img = cv2.imread(single_img_np) 
        gt_img =  cv2.imread(single_label_np)

        img_hist = self.histogram_loader(input_img)
        # gt_hist = self.histogram_loader(gt_img)
        img_lab = self.histogram_loader(self.lab_loader(input_img))
        # label_lab = self.lab_loader(gt_img)

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        return img_hist, img_lab, input_img, gt_img, single_img_np

    def __len__(self):
        return len(self.images)


class testset_set1(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader_BGRimage
        self.lab_loader = RGBtoLab
        self.transform = transforms.ToTensor()

        self.images = test_set1_img_path
        # self.label = test_set1_gt_path

    def __getitem__(self, index):
        gt_ext = ('G_AS.png','G_AS.png')
        single_img = self.images[index]
        parts = single_img.split('_')
        base_name = ''
        for i in range(len(parts) - 2):
            base_name = base_name + parts[i] + '_'
        gt_awb_file = base_name + gt_ext[0]
        parts = gt_awb_file.split('/')
        parts.remove('input')
        parts[-2] = 'gt'
        gt_awb_file = '/'.join(parts)
        single_label = gt_awb_file

        input_img = cv2.imread(single_img) 
        gt_img =  cv2.imread(single_label)

        img_hist = self.histogram_loader(input_img)
        img_lab = self.histogram_loader(self.lab_loader(input_img))

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        return img_hist, img_lab, input_img, gt_img, single_img

    def __len__(self):
        return len(self.images)


class testset_set2(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader_BGRimage
        self.lab_loader = RGBtoLab
        self.transform = transforms.ToTensor()

        self.images = set2_img_path
        self.label = set2_gt_path

    def __getitem__(self, index):
        single_img_np = self.images[index]        
        img_name = single_img_np.split('/')[-1]
        single_label_np = self.label+img_name
        input_img = cv2.imread(single_img_np) 
        gt_img =  cv2.imread(single_label_np)

        img_hist = self.histogram_loader(input_img)
        # gt_hist = self.histogram_loader(gt_img)
        img_lab = self.histogram_loader(self.lab_loader(input_img))
        # label_lab = self.lab_loader(gt_img)

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        return img_hist, img_lab, input_img, gt_img, single_img_np

    def __len__(self):
        return len(self.images)
    

class testsetmix(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader_BGRimage
        self.lab_loader = RGBtoLab
        self.transform = transforms.ToTensor()

        self.images = mixedWB_img_path
        self.label = mixedWB_gt_path

    def __getitem__(self, index):
        gt_ext = ('G_AS.png')
        single_img = self.images[index]
        parts = single_img.split('_')
        base_name = ''
        for i in range(len(parts) - 2):
            base_name = base_name + parts[i] + '_'
        gt_awb_file = base_name + gt_ext
        parts = gt_awb_file.split('/')
        parts[-2] = 'gt'
        gt_awb_file = '/'.join(parts)
        single_label = gt_awb_file
        
        input_img = cv2.imread(single_img) ###rgb
        gt_img =  cv2.imread(single_label)

        img_hist = self.histogram_loader(input_img)
        img_lab = self.histogram_loader(self.lab_loader(input_img))

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        return img_hist, img_lab, input_img, gt_img, single_img

    def __len__(self):
        return len(self.images) 