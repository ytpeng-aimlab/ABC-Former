import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src import ABCFormerM
import random
from mixedillWB.src import ops
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tb = True
except ImportError:
    use_tb = False

# use_tb = False    # TB file too large after a point
from src import dataset
from torch.utils.data import DataLoader


def train_net(net_h, net_pl, net_s, device, data_dir, epochs=140,
              batch_size=32, lr=0.001, l2reg=0.00001, grad_clip_value=0,
              chkpoint_period=10, smooth_weight=0.01,
              multiscale=False, wb_settings=None, shuffle_order=True,
              patch_number=12, optimizer_algo='Adam', max_tr_files=0,
              patch_size=128, model_name='WB_model',
              save_cp=True):
    """ Trains a network and saves the trained model in harddisk. """

    dir_checkpoint = 'checkpoints_model/Hist_PDFLab_sRGB/16d/DSTFC_64'  # check points directory
    os.makedirs(dir_checkpoint, exist_ok=True)

    SMOOTHNESS_WEIGHT = smooth_weight

    input_files = dataset.Data.load_files(data_dir)
    random.shuffle(input_files)

    if max_tr_files > 0:
        if max_tr_files < len(input_files):
            input_files = input_files[:max_tr_files]

    dataset.Data.assert_files(input_files, wb_settings=wb_settings)

    train_set = dataset.Data(input_files, patch_size=patch_size,
                             patch_number=patch_number, multiscale=multiscale,
                             shuffle_order=shuffle_order, wb_settings=wb_settings)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True)


    if use_tb:  # if TensorBoard is used
        writer = SummaryWriter(log_dir='runs/' + model_name,
                               comment=f'LR_{lr}_BS_{batch_size}')
    else:
        writer = None
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:                {epochs}
        WB Settings:           {wb_settings}
        Batch size:            {batch_size}
        Patch per image:       {patch_number}
        Patch size:            {patch_size} x {patch_size}
        Learning rate:         {lr}
        L2 reg. weight:        {l2reg}
        Smooth weight:         {smooth_weight}
        Grad. clipping:        {grad_clip_value}
        Optimizer:             {optimizer_algo}
        Checkpoints:           {save_cp}
        Device:                {device.type}
        TensorBoard:           {use_tb}
  ''')
    

    ### Checkpoint ###
    # checkpoint_hist = torch.load("/mnt/disk2/cyc202/awbformer/HVDualformerW_code/checkpoints_model/DST_64/Hist_PDFLab_sRGB/16d/Histformer_64p_16d_epoch_296.pth")
    # checkpoint_PDFlab = torch.load("/mnt/disk2/cyc202/awbformer/HVDualformerW_code/checkpoints_model/DST_64/Hist_PDFLab_sRGB/16d/PDFLabformer_64p_16d_epoch_296.pth")
    # checkpoint_sRGB = torch.load("/mnt/disk2/cyc202/awbformer/HVDualformerW_code/checkpoints_model/DST_64/Hist_PDFLab_sRGB/16d/sRGBformer_64p_16d_epoch_296.pth")
    
    ### Model ###
    # net_h.load_state_dict(checkpoint_hist['state_dict'])
    # net_pl.load_state_dict(checkpoint_PDFlab['state_dict'])
    # net_s.load_state_dict(checkpoint_sRGB['state_dict'])

    ### Optimizer ###
    # optimizer_h = torch.optim.AdamW(net_h.parameters(), lr=lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
    optimizer_h = torch.optim.Adam(net_h.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2reg)
    # optimizer_h.load_state_dict(checkpoint_hist['optimizer'])
    optimizer_pl = torch.optim.Adam(net_pl.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2reg)
    # optimizer_pl.load_state_dict(checkpoint_PDFlab['optimizer'])
    optimizer_s = torch.optim.Adam(net_s.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=l2reg)
    # optimizer_s.load_state_dict(checkpoint_sRGB['optimizer'])

    x_kernel, y_kernel = ops.get_sobel_kernel(device, chnls=len(wb_settings))

    for epoch in range(epochs):
        net_h.train()
        net_pl.train()
        net_s.train()
        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_smoothness_loss = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1} / {epochs}',
                  unit='img') as pbar:
            for ik,(batch) in enumerate(train_loader):
                img = batch['image']
                img = img.to(device=device, dtype=torch.float32)
                gt = batch['gt']
                gt = gt.to(device=device, dtype=torch.float32)
                imghist = batch['hist']
                imghist = imghist.to(device=device, dtype=torch.float32)
                gthist = batch['gthist']
                gthist = gthist.to(device=device, dtype=torch.float32)
                imgPDFLab = batch['PDFLab']
                imgPDFLab = imgPDFLab.to(device=device, dtype=torch.float32)
                gtPDFLab = batch['gtPDFLab']
                gtPDFLab = gtPDFLab.to(device=device, dtype=torch.float32)

                
                RGB_loss=0
                PDFLab_loss=0
                rec_loss = 0
                smoothness_loss = 0
                for p in range(img.shape[1]):
                    patch = img[:, p, :, :, :]
                    gt_patch = gt[:, p, :, :, :]
                    patch_hist = imghist[:, p, :, :]
                    gt_patch_hist = gthist[:, p, :, :]
                    patch_PDFLab = imgPDFLab[:, p, :, :]
                    gt_patch_PDFLab = gtPDFLab[:, p, :, :]
                    
                    hist_result, hist_W = net_h(patch_hist)
                    R_loss = ops.L2_histo(hist_result[:,0], gt_patch_hist[:,0])
                    G_loss = ops.L2_histo(hist_result[:,1], gt_patch_hist[:,1])
                    B_loss = ops.L2_histo(hist_result[:,2], gt_patch_hist[:,2])
                    RGB_loss += torch.mean(R_loss+G_loss+B_loss)

                    PDFLab_result, PDFlab_W  = net_pl(patch_PDFLab)
                    PDFlab_B_loss = ops.L2_histo(PDFLab_result[:,0], gt_patch_PDFLab[:,0])
                    PDFlab_G_loss = ops.L2_histo(PDFLab_result[:,1], gt_patch_PDFLab[:,1])
                    PDFlab_R_loss = ops.L2_histo(PDFLab_result[:,2], gt_patch_PDFLab[:,2])
                    PDFLab_loss += torch.mean(PDFlab_B_loss+PDFlab_G_loss+PDFlab_R_loss)

                    result, weights = net_s(patch, hist_W, PDFlab_W)
                    rec_loss += ops.compute_loss(result, gt_patch)

                    smoothness_loss += SMOOTHNESS_WEIGHT * (
                            torch.sum(F.conv2d(weights, x_kernel, stride=1) ** 2) +
                            torch.sum(F.conv2d(weights, y_kernel, stride=1) ** 2))

                RGB_loss = (RGB_loss / img.shape[1])
                optimizer_h.zero_grad()
                RGB_loss.backward()
                optimizer_h.step()

                PDFLab_loss = (PDFLab_loss / img.shape[1])
                optimizer_pl.zero_grad()
                PDFLab_loss.backward()
                optimizer_pl.step()

                rec_loss = rec_loss / img.shape[1]
                smoothness_loss = smoothness_loss / img.shape[1]
                loss = rec_loss + smoothness_loss

                py_loss = loss.item()
                py_rec_loss = rec_loss.item()

                py_smoothness_loss = smoothness_loss.item()
                epoch_smoothness_loss += py_smoothness_loss

                epoch_rec_loss += py_rec_loss
                epoch_loss += py_loss

                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()

                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(net_s.parameters(), grad_clip_value)
                if ik %20==0:
                   print('epoch: {} ,batch: {}, Hist_loss: {}, PDFLab_loss: {}, sRGB_loss: {}'.format(epoch + 1, ik + 1, RGB_loss.data, PDFLab_loss.data, loss.data))

                pbar.set_postfix(**{'Total loss (batch)': py_loss},
                                 **{'Rec. loss (batch)': py_rec_loss},
                                 **{'Smoothness loss (batch)': py_smoothness_loss}
                                 )

                global_step += 1

        epoch_loss = epoch_loss / (len(train_loader))
        epoch_rec_loss = epoch_rec_loss / (len(train_loader))
        epoch_smoothness_loss = epoch_smoothness_loss / (len(train_loader))
        logging.info(f'{model_name} - Epoch loss: = {epoch_loss}, '
                     f'Rec. loss = {epoch_rec_loss}, '
                     f'Smoothness loss = {epoch_smoothness_loss}')

        if ((epoch + 1) % chkpoint_period == 0) or ((epoch + 1) >100):
            # if not os.path.exists(dir_checkpoint):
            #     os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')

            torch.save({'state_dict': net_h.state_dict(),
                        'optimizer' : optimizer_h.state_dict()
                        }, os.path.join(dir_checkpoint,"Histformer_64p_16d_epoch_{}.pth".format(epoch+1)))
            torch.save({'state_dict': net_pl.state_dict(),
                        'optimizer' : optimizer_pl.state_dict()
                        }, os.path.join(dir_checkpoint,"PDFLabformer_64p_16d_epoch_{}.pth".format(epoch+1)))
            torch.save({'state_dict': net_s.state_dict(),
                        'optimizer' : optimizer_s.state_dict()
                        }, os.path.join(dir_checkpoint,"sRGBformer_64p_16d_epoch_{}.pth".format(epoch+1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


    # torch.save(net.state_dict(), 'models/' + f'{model_name}.pth')
    logging.info('Saved trained model!')

    if use_tb:
        writer.close()

    logging.info('End of training')

def get_args():
    """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

    parser = argparse.ArgumentParser(description='Train WB Correction.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=350, help='Number of epochs', dest='epochs')
    parser.add_argument('-s', '--patch-size', dest='patch_size', type=int, default=64, help='Size of input training patches')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=80, help='Batch size', dest='batch_size')
    parser.add_argument('-pn', '--patch-number', type=int, default=4, help='number of patches per trainig image', dest='patch_number')
    parser.add_argument('-nrm', '--normalization', dest='norm', type=bool, default=False, help='Apply BN in network')
    parser.add_argument('-msc', '--multi-scale', dest='multiscale', type=bool, default=False, help='Multi-scale training samples')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('-mtf', '--max-tr-files', dest='max_tr_files', type=int, default=0, help='max number of training files; default is 0 which uses all files')
    parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str, default='Adam', help='Adam or SGD or AdamW')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float, nargs='?', default=0, help='L2 regularization factor', dest='l2r')
    parser.add_argument('-sw', '--smoothness-weight', dest='smoothness_weight', type=float, default=100.0, help='smoothness weight')

    parser.add_argument('-wbs', '--wb-settings', dest='wb_settings', nargs='+', default=['D', 'S', 'T', 'F', 'C'])  
    parser.add_argument('-l', '--load', dest='load', type=bool, default=False, help='Load model from a .pth file')
    parser.add_argument('-so', '--shuffle-order', dest='shuffle_order', type=bool, default=False, help='Shuffle order of WB')
    parser.add_argument('-ml', '--model-location', dest='model_location', default=None)
    parser.add_argument('-cpf', '--checkpoint-frequency', dest='cp_freq', type=int, default=10, help='Checkpoint frequency.')
    parser.add_argument('-gc', '--grad-clip-value', dest='grad_clip_value', type=float, default=0, help='Gradient clipping value; if = 0, no clipping applied')
    parser.add_argument('-trd', '--training-dir', dest='trdir', default='/mnt/disk2/roger/mywork3_W/data/images', help='Training directory')
    parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)
    parser.add_argument('-mn', '--model-name', dest='model_name', type=str, default='ifrnet', help='Model name')
    parser.add_argument('-emb', '--embed_dim', dest='embed_dim',type=int, default=16, help='embed_dim.')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training Mixed-Ill WB correction')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        torch.cuda.set_device(args.gpu)
    logging.info(f'Using device {device}')

    hist_net = ABCFormerM.HistNet(device=device, inchnls=3 * len(args.wb_settings), em_dim=args.embed_dim, wbset_num=len(args.wb_settings))
    PDFlab_net = ABCFormerM.PDFLabNet(device=device, inchnls=3 * len(args.wb_settings), em_dim=args.embed_dim, wbset_num=len(args.wb_settings))
    sRGB_net = ABCFormerM.sRGBNet(device=device, inchnls=3 * len(args.wb_settings), em_dim=args.embed_dim, wbset_num=len(args.wb_settings))

    hist_net.to(device=device)
    PDFlab_net.to(device=device)
    sRGB_net.to(device=device)

    postfix = f'_p_{args.patch_size}'

    if args.norm:
        postfix += f'_w_BN'

    if args.shuffle_order:
        postfix += f'_w_shuffling'

    if args.smoothness_weight == 0:
        postfix += f'_wo_smoothing'

    for wb_setting in args.wb_settings:
        postfix += f'_{wb_setting}'

    model_name = args.model_name + postfix

    try:
        train_net(net_h=hist_net, net_pl=PDFlab_net, net_s=sRGB_net, device=device, data_dir=args.trdir,
                  patch_number=args.patch_number,
                  multiscale=args.multiscale,
                  smooth_weight=args.smoothness_weight,
                  max_tr_files=args.max_tr_files,
                  wb_settings=args.wb_settings,
                  shuffle_order=args.shuffle_order,
                  epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr,
                  l2reg=args.l2r,
                  optimizer_algo=args.optimizer,
                  grad_clip_value=args.grad_clip_value,
                  chkpoint_period=args.cp_freq, 
                  patch_size=args.patch_size,
                  model_name=model_name)

    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'wb_correction_intrrupted_check_point.pth')
        # logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
