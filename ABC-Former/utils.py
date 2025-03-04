import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))
from sklearn.linear_model import LinearRegression


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result


def hist_match(image_path,image_path_hs,out_R,out_G,out_B):   
    asli = cv2.imread(image_path)
    # print('asli',asli.shape)
    ori_img = np.array(asli)
    # print('input_image',np.array(asli).shape)
    hs = cv2.imread(image_path_hs)
    # print('hs',hs)
    result = np.copy(asli)
    R_hist, R_bins = np.histogram(asli[:, :, 2], bins=256, range=(0, 256)) 
    G_hist, G_bins = np.histogram(asli[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(asli[:, :, 0], bins=256, range=(0, 256))
    asli_B = asli[:,:,0]
    asli_G = asli[:,:,1]
    asli_R = asli[:,:,2]
    asli_shape_B = asli_B.shape    
    asli_shape_G = asli_G.shape 
    asli_shape_R = asli_R.shape 
    asli_B = asli_B.ravel()
    asli_G = asli_G.ravel()
    asli_R = asli_R.ravel()

    values = np.array(range(0,256),dtype=np.uint8)
    try:
        o_values_B, bin_idx_B, o_counts_B = np.unique(asli_B, return_inverse=True,return_counts=True)
        o_values_G, bin_idx_G, o_counts_G = np.unique(asli_G, return_inverse=True,return_counts=True)
        o_values_R, bin_idx_R, o_counts_R = np.unique(asli_R, return_inverse=True,return_counts=True)

        o_quantiles_B = np.cumsum(o_counts_B).astype(np.float64)
        o_quantiles_B /= o_quantiles_B[-1]
        o_quantiles_G = np.cumsum(o_counts_G).astype(np.float64)
        o_quantiles_G /= o_quantiles_G[-1]
        o_quantiles_R = np.cumsum(o_counts_R).astype(np.float64)
        o_quantiles_R /= o_quantiles_R[-1]
        b_quantiles = np.cumsum((out_B.squeeze(0)*sum(B_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        b_quantiles /= b_quantiles[-1]
        g_quantiles = np.cumsum((out_G.squeeze(0)*sum(G_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        g_quantiles /= g_quantiles[-1]
        r_quantiles = np.cumsum((out_R.squeeze(0)*sum(R_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        r_quantiles /= r_quantiles[-1]
        interp_t_valuesB = np.interp(o_quantiles_B, b_quantiles, values) #, b_values
        interp_t_valuesG = np.interp(o_quantiles_G, g_quantiles, values)
        interp_t_valuesR = np.interp(o_quantiles_R, r_quantiles, values)
        result[:,:,0] = interp_t_valuesB[bin_idx_B].reshape(asli_shape_B)
        result[:,:,1] = interp_t_valuesG[bin_idx_G].reshape(asli_shape_G)
        result[:,:,2] = interp_t_valuesR[bin_idx_R].reshape(asli_shape_R)

        return result, hs,ori_img
    except ValueError:
        pass

def npTOtensor(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    
    return image

def align_to_four(img):
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]

    return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def padding_for_unet(img):
    ##########往右下做padding
    # print('img.size',img.size())
    img_size=img.size()
    h ,w= img_size[2] , img_size[3]
    # print('h ,w',h ,w)
    if h%16 != 0:
        # print('改變h',h%16)
        a = h%16
        img = torch.nn.functional.pad(img, (0, 0, 0, 16-a))
        # print('16-a',16-a)
        # print('new_img.size_h_pad',img.size())
    if w%16 != 0:
        # print('改變w')
        b = w%16
        img = torch.nn.functional.pad(img, (0, 16-b, 0, 0))
        # print('16-b',16-b)
        # print('new_img.size_hw_pad',img.size())
    return img

def hist_matchpatchp(image,label,out_R,out_G,out_B):   
    
    asli = image
    ori_img = np.array(image)
    # print('input_image',np.array(asli).shape)
    label1=label
    hs = label1
    result = np.copy(asli)
    R_hist, R_bins = np.histogram(asli[:, :, 0], bins=256, range=(0, 256)) 
    G_hist, G_bins = np.histogram(asli[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(asli[:, :, 2], bins=256, range=(0, 256))
    asli_R = asli[:,:,0]
    asli_G = asli[:,:,1]
    asli_B = asli[:,:,2]
    asli_shape_B = asli_B.shape    
    asli_shape_G = asli_G.shape 
    asli_shape_R = asli_R.shape 
    asli_B = asli_B.ravel()
    asli_G = asli_G.ravel()
    asli_R = asli_R.ravel()

    values = np.array(range(0,256),dtype=np.uint8)
    try:
        o_values_B, bin_idx_B, o_counts_B = np.unique(asli_B, return_inverse=True,return_counts=True)
        o_values_G, bin_idx_G, o_counts_G = np.unique(asli_G, return_inverse=True,return_counts=True)
        o_values_R, bin_idx_R, o_counts_R = np.unique(asli_R, return_inverse=True,return_counts=True)

        o_quantiles_B = np.cumsum(o_counts_B).astype(np.float64)
        o_quantiles_B /= o_quantiles_B[-1]
        o_quantiles_G = np.cumsum(o_counts_G).astype(np.float64)
        o_quantiles_G /= o_quantiles_G[-1]
        o_quantiles_R = np.cumsum(o_counts_R).astype(np.float64)
        o_quantiles_R /= o_quantiles_R[-1]
        b_quantiles = np.cumsum((out_B.squeeze(0)*sum(B_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        b_quantiles /= b_quantiles[-1]
        g_quantiles = np.cumsum((out_G.squeeze(0)*sum(G_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        g_quantiles /= g_quantiles[-1]
        r_quantiles = np.cumsum((out_R.squeeze(0)*sum(R_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        r_quantiles /= r_quantiles[-1]
        interp_t_valuesB = np.interp(o_quantiles_B, b_quantiles, values) #, b_values
        interp_t_valuesG = np.interp(o_quantiles_G, g_quantiles, values)
        interp_t_valuesR = np.interp(o_quantiles_R, r_quantiles, values)
        result[:,:,0] = interp_t_valuesB[bin_idx_B].reshape(asli_shape_B)
        result[:,:,1] = interp_t_valuesG[bin_idx_G].reshape(asli_shape_G)
        result[:,:,2] = interp_t_valuesR[bin_idx_R].reshape(asli_shape_R)

        return result, hs,ori_img
    except ValueError:
        pass
def hist_matchpatch_batch(image,out_R,out_G,out_B):   
    
    asli = image.cpu()
    result = np.copy(asli.cpu())
    R_hist, R_bins = np.histogram(asli[:, :, 0], bins=256, range=(0, 256)) 
    G_hist, G_bins = np.histogram(asli[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(asli[:, :, 2], bins=256, range=(0, 256))
    asli_R = asli[:,:,0]
    asli_G = asli[:,:,1]
    asli_B = asli[:,:,2]
    asli_shape_B = asli_B.shape    
    asli_shape_G = asli_G.shape 
    asli_shape_R = asli_R.shape 
    asli_B = asli_B.ravel()
    asli_G = asli_G.ravel()
    asli_R = asli_R.ravel()

    values = np.array(range(0,256),dtype=np.uint8)
    try:
        o_values_B, bin_idx_B, o_counts_B = np.unique(asli_B, return_inverse=True,return_counts=True)
        o_values_G, bin_idx_G, o_counts_G = np.unique(asli_G, return_inverse=True,return_counts=True)
        o_values_R, bin_idx_R, o_counts_R = np.unique(asli_R, return_inverse=True,return_counts=True)

        o_quantiles_B = np.cumsum(o_counts_B).astype(np.float64)
        o_quantiles_B /= o_quantiles_B[-1]
        o_quantiles_G = np.cumsum(o_counts_G).astype(np.float64)
        o_quantiles_G /= o_quantiles_G[-1]
        o_quantiles_R = np.cumsum(o_counts_R).astype(np.float64)
        o_quantiles_R /= o_quantiles_R[-1]
        b_quantiles = np.cumsum((out_B.squeeze(0)*sum(B_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        b_quantiles /= b_quantiles[-1]
        g_quantiles = np.cumsum((out_G.squeeze(0)*sum(G_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        g_quantiles /= g_quantiles[-1]
        r_quantiles = np.cumsum((out_R.squeeze(0)*sum(R_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        r_quantiles /= r_quantiles[-1]
        interp_t_valuesB = np.interp(o_quantiles_B, b_quantiles, values) #, b_values
        interp_t_valuesG = np.interp(o_quantiles_G, g_quantiles, values)
        interp_t_valuesR = np.interp(o_quantiles_R, r_quantiles, values)
        result[:,:,0] = interp_t_valuesB[bin_idx_B].reshape(asli_shape_B)
        result[:,:,1] = interp_t_valuesG[bin_idx_G].reshape(asli_shape_G)
        result[:,:,2] = interp_t_valuesR[bin_idx_R].reshape(asli_shape_R)

        return result
    except ValueError:
        pass
def hist_matchpatch(image,label,out_R,out_G,out_B):   
    
    asli = image
    ori_img = np.array(image)
    # print('input_image',np.array(asli).shape)
    label1=label.copy()
    hs = label1
    print('hs',hs.shape)
    hs = hs[..., ::-1]
    result = np.copy(asli)
    R_hist, R_bins = np.histogram(asli[:, :, 0], bins=256, range=(0, 256)) 
    G_hist, G_bins = np.histogram(asli[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(asli[:, :, 2], bins=256, range=(0, 256))
    asli_R = asli[:,:,0]
    asli_G = asli[:,:,1]
    asli_B = asli[:,:,2]
    asli_shape_B = asli_B.shape    
    asli_shape_G = asli_G.shape 
    asli_shape_R = asli_R.shape 
    asli_B = asli_B.ravel()
    asli_G = asli_G.ravel()
    asli_R = asli_R.ravel()

    values = np.array(range(0,256),dtype=np.uint8)
    try:
        o_values_B, bin_idx_B, o_counts_B = np.unique(asli_B, return_inverse=True,return_counts=True)
        o_values_G, bin_idx_G, o_counts_G = np.unique(asli_G, return_inverse=True,return_counts=True)
        o_values_R, bin_idx_R, o_counts_R = np.unique(asli_R, return_inverse=True,return_counts=True)

        o_quantiles_B = np.cumsum(o_counts_B).astype(np.float64)
        o_quantiles_B /= o_quantiles_B[-1]
        o_quantiles_G = np.cumsum(o_counts_G).astype(np.float64)
        o_quantiles_G /= o_quantiles_G[-1]
        o_quantiles_R = np.cumsum(o_counts_R).astype(np.float64)
        o_quantiles_R /= o_quantiles_R[-1]
        b_quantiles = np.cumsum((out_B.squeeze(0)*sum(B_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        b_quantiles /= b_quantiles[-1]
        g_quantiles = np.cumsum((out_G.squeeze(0)*sum(G_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        g_quantiles /= g_quantiles[-1]
        r_quantiles = np.cumsum((out_R.squeeze(0)*sum(R_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        r_quantiles /= r_quantiles[-1]
        interp_t_valuesB = np.interp(o_quantiles_B, b_quantiles, values) #, b_values
        interp_t_valuesG = np.interp(o_quantiles_G, g_quantiles, values)
        interp_t_valuesR = np.interp(o_quantiles_R, r_quantiles, values)
        result[:,:,0] = interp_t_valuesB[bin_idx_B].reshape(asli_shape_B)
        result[:,:,1] = interp_t_valuesG[bin_idx_G].reshape(asli_shape_G)
        result[:,:,2] = interp_t_valuesR[bin_idx_R].reshape(asli_shape_R)

        return result, hs,ori_img
    except ValueError:
        pass
def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)
      
    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask
def histogram_loaderpatch(image):
    R_hist, R_bins = np.histogram(image[:, :, 0], bins=256, range=(0, 256)) # R_hist.shape = (256,)
    G_hist, G_bins = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(image[:, :, 2], bins=256, range=(0, 256))

    R_pdf = R_hist/sum(R_hist)
    G_pdf = G_hist/sum(G_hist)
    B_pdf = B_hist/sum(B_hist)
    RGB = np.vstack((R_pdf,G_pdf,B_pdf))
    return RGB