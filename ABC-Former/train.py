import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from torchvision import models
from torchvision import  transforms

from train_dataset import *
from utils import *
from loss import *
from PDFformer_hist import Hist_Histoformer
from PDFformer_lab import Lab_Histoformer
from sRGBformer import CAFormer

def main(opt):
    dataset = BasicDataset_BGR(opt.dir_img, fold=0, patch_size=opt.patch_size, patch_num_per_image=4, max_trdata=12000)
    trainloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    ### Checkpoint ###
#     checkpoint_hist = torch.load("/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/hist_net/Hist_d24_epoch_350.pth")
#     checkpoint_lab = torch.load("/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/lab_net/Lab_d24_epoch_350.pth")
#     checkpoint_sRGB = torch.load("/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/sRGB_net/sRGB_d24_epoch_350.pth")
    
    ### Model ###
    hist_net = Hist_Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF').cuda()
    lab_net = Lab_Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF').cuda()
    sRGB_net = CAFormer(embed_dim=(opt.embed_dim),token_projection='linear').cuda()
#     hist_net.load_state_dict(checkpoint_hist['state_dict'])
#     lab_net.load_state_dict(checkpoint_lab['state_dict'])
#     sRGB_net.load_state_dict(checkpoint_sRGB['state_dict'])

    ### Cuda/GPU ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False
  
    ### Optimizer ###
    hist_optimizer = torch.optim.Adam(hist_net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lab_optimizer = torch.optim.Adam(lab_net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    sRGB_optimizer = torch.optim.Adam(sRGB_net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     hist_optimizer.load_state_dict(checkpoint_hist['optimizer'])
#     lab_optimizer.load_state_dict(checkpoint_lab['optimizer'])
#     sRGB_optimizer.load_state_dict(checkpoint_sRGB['optimizer'])

    ### Patch number ###
    patchnum=4
    
    ### Loss ###
    criterionL1 = nn.L1Loss().cuda()

    import sys
    for e in range(350, opt.epochs):
        torch.cuda.empty_cache()
        hist_net.train()
        lab_net.train()
        sRGB_net.train()
        for ik,(batch) in enumerate(trainloader):
            imgs_ = batch['image']
            awb_gt_ = batch['gt_AWB']
            imgs_hist = batch['image_hist']
            gts_hist = batch['gt_hist']
            lab_imgs = batch['image_lab']
            lab_gt = batch['gt_lab']

            for j in range(patchnum):
                input_img = imgs_[:, (j * 3): 3 + (j * 3), :, :]  #(1, 3, 128, 128) #bgr
                label_img= awb_gt_[:, (j * 3): 3 + (j * 3), :, :] #bgr
                input_hist = imgs_hist[:, (j * 3): 3 + (j * 3), :] #bgr
                label_hist = gts_hist[:, (j * 3): 3 + (j * 3), :] #bgr
                input_lab_img = lab_imgs[:, (j * 3): 3 + (j * 3), :]
                label_lab_img = lab_gt[:, (j * 3): 3 + (j * 3), :]
                
                input_img = input_img.to(torch.float).to(device)
                label_img = label_img.to(torch.float).to(device)
                input_hist = input_hist.to(torch.float).to(device)
                label_hist = label_hist.to(torch.float).to(device)
                input_lab_img = input_lab_img.to(torch.float).to(device)
                label_lab_img = label_lab_img.to(torch.float).to(device)

                input_img = (input_img/255.0).cuda()
                gt = (label_img/255.0).cuda()   

                # Histoformer
                pred_hist, hist_W = hist_net(input_hist)

                R_out = pred_hist[:,2]
                G_out = pred_hist[:,1]
                B_out = pred_hist[:,0]
                R_labels = label_hist[:,2]
                G_labels = label_hist[:,1]
                B_labels = label_hist[:,0]

                hist_optimizer.zero_grad()
                R_loss = L2_histo(R_out, R_labels)
                G_loss = L2_histo(G_out, G_labels)
                B_loss = L2_histo(B_out, B_labels)
                RGB_loss = (1*R_loss)+(1*G_loss)+(1*B_loss)
                hist_loss = torch.mean(RGB_loss)
                hist_loss.backward(retain_graph=True)
                hist_optimizer.step() 

                # Labformer
                pred_lab_img, lab_W  = lab_net(input_lab_img)

                lab_R_out = pred_lab_img[:,2]
                lab_G_out = pred_lab_img[:,1]
                lab_B_out = pred_lab_img[:,0]
                lab_R_labels = label_lab_img[:,2]
                lab_G_labels = label_lab_img[:,1]
                lab_B_labels = label_lab_img[:,0]

                lab_optimizer.zero_grad()
                lab_R_loss = L2_histo(lab_R_out, lab_R_labels)
                lab_G_loss = L2_histo(lab_G_out, lab_G_labels)
                lab_B_loss = L2_histo(lab_B_out, lab_B_labels)
                RGB_loss = lab_R_loss+lab_G_loss+lab_B_loss
                lab_loss = torch.mean(RGB_loss)
                lab_loss.backward(retain_graph=True) 
                lab_optimizer.step() 

                # sRGBformer
                pred_image = sRGB_net(input_img, hist_W, lab_W) 

                sRGB_optimizer.zero_grad()        
                sRGB_mae_loss = criterionL1(pred_image, gt)
                sRGB_mae_loss.backward(retain_graph=True)
                sRGB_optimizer.step()
                        
            if ik %20==0:
                print('epoch: {}, batch: {}, hist_loss: {}, lab_mae_loss: {}, sRGB_mae_loss: {}'.format(e+1, ik+1, hist_loss.data, lab_loss.data, sRGB_mae_loss.data))


        torch.save({'epoch': e+1, 
                    'state_dict': hist_net.state_dict(),
                    'optimizer' : hist_optimizer.state_dict()
                    }, os.path.join(opt.save_hist_dir, "Hist_d24_last.pth"))

        torch.save({'epoch': e+1, 
                'state_dict': lab_net.state_dict(),
                'optimizer' : lab_optimizer.state_dict()
                }, os.path.join(opt.save_lab_dir, "Lab_d24_last.pth"))

        torch.save({'epoch': e+1, 
                'state_dict': sRGB_net.state_dict(),
                'optimizer' : sRGB_optimizer.state_dict()
                }, os.path.join(opt.save_sRGB_dir,"sRGB_d24_last.pth"))
        
        ### Save model ###
        if (e+1)>200 or (e+1)%10==0 or e==0:
            torch.save({'epoch': e+1, 
                    'state_dict': hist_net.state_dict(),
                    'optimizer' : hist_optimizer.state_dict()
                    }, os.path.join(opt.save_hist_dir, "Hist_d24_epoch_{}.pth".format(e+1)))

            torch.save({'epoch': e+1, 
                    'state_dict': lab_net.state_dict(),
                    'optimizer' : lab_optimizer.state_dict()
                    }, os.path.join(opt.save_lab_dir, "Lab_d24_epoch_{}.pth".format(e+1)))

            torch.save({'epoch': e+1, 
                    'state_dict': sRGB_net.state_dict(),
                    'optimizer' : sRGB_optimizer.state_dict()
                    }, os.path.join(opt.save_sRGB_dir,"sRGB_d24_epoch_{}.pth".format(e+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='histogram_network')
    # global settings
    parser.add_argument('--batch_size', type=int, default=120, help='training batch size')
    parser.add_argument('--epochs', type=int, default=500, help='the starting epoch count')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--patch_size', type=int, default=128, help='training patch_size')
    parser.add_argument('--embed_dim', type=int, default=16, help='dim of emdeding features')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='TwoDCFF', help='TwoDCFF/ffn token mlp')
    parser.add_argument('--save_hist_dir', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/hist_net',  help='save dir')
    parser.add_argument('--save_lab_dir', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/lab_net',  help='save dir')
    parser.add_argument('--save_sRGB_dir', type=str, default ='/mnt/disk2/cyc202/awbformer/Hist_PDFLab_sRGB/noPretrained_checkpoints/16d/sRGB_net',  help='save dir')
    parser.add_argument('--dir_img', type=str, default =r'/mnt/disk2/roger/awb_mywork/datasets/train/input',  help='read dir_img path')

    opt = parser.parse_args()
    os.makedirs(opt.save_hist_dir, exist_ok=True)
    os.makedirs(opt.save_lab_dir, exist_ok=True)
    os.makedirs(opt.save_sRGB_dir, exist_ok=True)
    print (opt)
    main(opt)