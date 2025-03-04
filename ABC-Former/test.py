import os
import cv2
import numpy as np
import skimage.io
from skimage import color
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from torchvision import models
from torchvision import  transforms

from PDFformer_hist import Hist_Histoformer
from PDFformer_lab import Lab_Histoformer
from sRGBformer import CAFormer

from test_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='histogram_network')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--embed_dim', type=int, default=16, help='dim of emdeding features')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='TwoDCFF', help='TwoDCFF/ffn token mlp')

parser.add_argument('--save_image_dir', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/result_noPretrained/cube/16d/epoch285/output',  help='save image dir')
parser.add_argument('--outdir', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/result_noPretrained/cube/16d/epoch285/txt',  help='outdir dir')

parser.add_argument('--hist_weight', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/hist_net/Hist_d24_epoch_285.pth',  help='hist weight')
parser.add_argument('--lab_weight', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/lab_net/Lab_d24_epoch_285.pth',  help='lab weight')
parser.add_argument('--sRGB_weight', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/sRGB_net/sRGB_d24_epoch_285.pth',  help='sRGB weight')

opt = parser.parse_args()

os.makedirs(opt.save_image_dir, exist_ok=True)
os.makedirs(opt.outdir, exist_ok=True)

def calc_mae(source, target):
  source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
  target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
  norm = source_norm * target_norm
  L = np.shape(norm)[0]
  inds = norm != 0
  angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
  angles[angles > 1] = 1
  f = np.arccos(angles)
  f[np.isnan(f)] = 0
  f = f * 180 / np.pi
  return sum(f)/ (L)

def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return calc_mae(a, b)

def calc_deltaE2000(source, target):
  source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
  target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
  source = color.rgb2lab(source)
  target = color.rgb2lab(target)
  error = np.mean(color.deltaE_ciede2000(source,target))
  return error

hist_net = Hist_Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF').cuda()
lab_net = Lab_Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF').cuda()
sRGB_net = CAFormer(embed_dim=(opt.embed_dim),token_projection='linear').cuda()

hist_net_checkpoint = torch.load(opt.hist_weight)
lab_net_checkpoint = torch.load(opt.lab_weight)
sRGB_net_checkpoint = torch.load(opt.sRGB_weight)
hist_net.load_state_dict(hist_net_checkpoint['state_dict'])
lab_net.load_state_dict(lab_net_checkpoint['state_dict'])
sRGB_net.load_state_dict(sRGB_net_checkpoint['state_dict'])

testloader = get_testset_cube()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hist_net.eval()
lab_net.eval()
sRGB_net.eval()
mse_lst, mae_lst, deltaE2000_lst = list(), list(), list()
with torch.no_grad(): 
    for i,(hist_inputs, lab_inputs, ori_img, label_img, path) in enumerate(testloader):
        total_images = len(testloader.dataset)
        img_name = path[0].split('/')[-1]

        hist_inputs = hist_inputs.to(device).to(device, dtype=torch.float32)
        lab_inputs = lab_inputs.to(device).to(device, dtype=torch.float32)
        ori_img = ori_img.to(device).to(device, dtype=torch.float32)
        label_img = label_img.to(device).to(device, dtype=torch.float32)

        pred_hist, hist_W = hist_net(hist_inputs)
        pred_lab, lab_W = lab_net(lab_inputs)
        predicted_image = sRGB_net(padding_for_unet(ori_img).float(), hist_W, lab_W)
        
        if ori_img.size() != predicted_image.size():
            ori_a = ori_img.size()[2]
            ori_b = ori_img.size()[3]
            after_a = predicted_image.size()[2]
            after_b = predicted_image.size()[3]

            if ori_a != after_a:
                predicted_image = predicted_image[:, :, :-(after_a-ori_a),:]
            if ori_b != after_b:
                predicted_image = predicted_image[:, :, :, :-(after_b-ori_b)] 
        if ori_img.size() != predicted_image.size():
            print('incorrect')

        out_img = (torch.clamp(predicted_image,0,1))
        for gt_, out_ in zip(label_img.float().permute(0, 2, 3, 1), out_img.float().permute(0, 2, 3, 1)):
            mae_and_delta_Es = [[mean_angular_error(gt_.cpu().squeeze().numpy().reshape(-1, 3), out_.cpu().squeeze().numpy().reshape(-1, 3)),
                                    np.mean(calc_deltaE2000(gt_.cpu().squeeze().numpy(),out_.cpu().squeeze().numpy()))]]
            mae, deltaE = np.mean(mae_and_delta_Es, axis=0)
            mse = (((gt_ - out_) * 255.) ** 2).mean().cpu().item()
            print("Sample {}/{}: MSE: {}, MAE: {}, DELTA_E: {}".format(i+1, total_images, mse, mae, deltaE), end="\n\n")
            sample_info = "Sample {}: MSE: {}, MAE: {}, DELTA_E: {}".format(img_name, mse, mae, deltaE)
            with open(os.path.join(opt.outdir, "sample.txt"), "a") as f:
               f.write(sample_info+'\n')
            mse_lst.append(mse)
            mae_lst.append(mae)
            deltaE2000_lst.append(deltaE)
            print("Average:\n"
            "\nMSE: {}, Q1: {}, Q2: {}, Q3: {}"
            "\nMAE: {}, Q1: {}, Q2: {}, Q3: {}"
            "\nDELTA_E: {}, Q1: {}, Q2: {}, Q3: {}".format(np.mean(mse_lst), np.quantile(mse_lst, 0.25), np.quantile(mse_lst, 0.5), np.quantile(mse_lst, 0.75),
                                                            np.mean(mae_lst), np.quantile(mae_lst, 0.25), np.quantile(mae_lst, 0.5), np.quantile(mae_lst, 0.75),
                                                            np.mean(deltaE2000_lst), np.quantile(deltaE2000_lst, 0.25), np.quantile(deltaE2000_lst, 0.5), np.quantile(deltaE2000_lst, 0.75)))

        final_info = "\nFinal Info--->  \nMSE: {}, Q1: {}, Q2: {}, Q3: {} \nMAE: {}, Q1: {}, Q2: {}, Q3: {} \nDELTA_E: {}, Q1: {}, Q2: {}, Q3: {}".format(
        np.mean(mse_lst), np.quantile(mse_lst, 0.25), np.quantile(mse_lst, 0.5), np.quantile(mse_lst, 0.75),
        np.mean(mae_lst), np.quantile(mae_lst, 0.25), np.quantile(mae_lst, 0.5), np.quantile(mae_lst, 0.75),
        np.mean(deltaE2000_lst), np.quantile(deltaE2000_lst, 0.25), np.quantile(deltaE2000_lst, 0.5), np.quantile(deltaE2000_lst, 0.75))
        if not os.path.exists(opt.outdir):
            os.makedirs(opt.outdir)
        with open(os.path.join(opt.outdir, "final.txt"), "w+") as f:
            f.write(final_info)

        # save output images
        output_img = predicted_image.cpu().numpy().transpose((0, 2, 3, 1))
        output_img = output_img[0, :, :, :]*255.0
        output_img = np.array(output_img)
        print('save:', img_name)
        cv2.imwrite(os.path.join(opt.save_image_dir, img_name), output_img) 