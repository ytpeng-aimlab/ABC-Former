import os
import cv2
# import time
import numpy as np
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from PIL import Image

from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum
from torch.nn import init
from torchvision import models 
########### Downsample/Upsample #############
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.transpose(1, 2)# (B, C, H, W)
        out = self.conv(x).transpose(1,2)# B H*W C
        return out

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        x = x.transpose(1, 2)# (B, C, H, W)
        out = self.deconv(x).transpose(1,2)# B H*W C
        return out


########### Input/Output Projection #############
class InputProjection(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        # print('InputProjection',x.shape)
        x = self.proj(x).transpose(1, 2)
        # print(x.shape)
        if self.norm is not None:
            x = self.norm(x)
        return x
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):###b c n 
        # print('CALayer',x.shape)
        y = self.avg_pool(x)
        # print('avg',y.shape)
        y = self.conv_du(y)
        # print('yconvdu',y.shape)
        return x * y, y    
class OutputProjection(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        x = x.transpose(1, 2)#(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


########### Multi-head Self-Attention #############

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = x.permute(0, 2, 1) ###b c n 
        res = self.body(x)
        res, y = self.CA(res)
        res += x
        return res, y



########### Feed-Forward Network #############
class TwoDCFF(nn.Module): #2D
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # print('2DCFF x',x.shape)        
        
        x = self.linear1(x) 
        bs, hw, c = x.size()
        x = x.reshape(bs,16,-1,c)
        x = x.permute(0,3,1,2)
        # print(' x',x.shape)        
        # exit()
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = x.reshape(bs,-1,c)

        x = self.linear2(x)

        return x


########### Transformer #############
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='TwoDCFF',se_layer=False):        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TwoDCFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        # self.intra_block = Intra_SA(dim, num_heads)
        self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())

    def forward(self, x, mask=None):
        # shortcut = x #  B N C 10 256 32
        # B_, N, C = x.shape
        # # print('shortcut',shortcut.shape)
        # x = self.norm1(x)#BNC
        # hx = self.intra_block(x)
        # hx = hx.permute(0, 2, 1)
        # # print('hx',hx.size())
        # # intra,_ = self.attn(x, mask=None)
        # # print('intra',intra.shape)
        # x = shortcut + self.drop_path(hx)
        
        shortcut_cha = x
        # print('shortcut_cha',shortcut_cha.shape)
        x = self.norm2(x)
        x, y = self.CAB(x)
        ###bcn
        # print('CAB',x.shape)
        x = x.permute(0, 2, 1)##bnc
        x = shortcut_cha + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x, y


########### Basic layer of Histoformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='TwoDCFF',se_layer=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])
         

    def forward(self, x, mask=None):
        Weight=[]
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, y = blk(x,mask)
                Weight.append(y.permute(0,2,1).detach())

        return x, Weight


########### Histoformer ################
class Hist_Histoformer(nn.Module): #[2, 2, 2, 2, 2, 2, 2, 2, 2] [1, 2, 8, 8, 8, 8, 8, 2, 1]
    def __init__(self, in_chans=3, embed_dim=32,
                 depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='TwoDCFF', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_projection = InputProjection(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_projection = OutputProjection(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        self.finalconv = nn.Conv1d(in_channels=2*embed_dim, out_channels=embed_dim, kernel_size=1, stride=1)
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            depth=depths[0],
                            num_heads=num_heads[0],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            depth=depths[1],
                            num_heads=num_heads[1],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            depth=depths[2],
                            num_heads=num_heads[2],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            depth=depths[4],
                            num_heads=num_heads[4],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim*8, embed_dim*4)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            depth=depths[5],
                            num_heads=num_heads[5],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            depth=depths[6],
                            num_heads=num_heads[6],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim*4, embed_dim*1)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            depth=depths[7],
                            num_heads=num_heads[7],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.apply(self._init_weights)
        self.softmax = nn.Softmax(2)


        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_projection(x)
        y = self.pos_drop(y)
        # print('y',y.size())
        #Encoder
        conv0,d0 = self.encoderlayer_0(y,mask=mask)
        # print('conv0',conv0.size(),d0.shape)

        pool0 = self.dowsample_0(conv0)
        conv1,d1 = self.encoderlayer_1(pool0,mask=mask)
        # print('conv1',conv1.size(),d1.shape)
        pool1 = self.dowsample_1(conv1)
        # print('pool1',pool1.size())
        conv2,d2= self.encoderlayer_2(pool1,mask=mask)
        # print('conv2',conv2.size(),d2.shape)
        pool2 = self.dowsample_2(conv2)
        # print('pool2',pool2.size())

        # Bottleneck
        conv4,d3 = self.conv(pool2, mask=mask)
        # print('conv4',conv4.size(),d3.shape)

        #Decoder
        up0 = self.upsample_0(conv4)
        # print('up0',up0.size())
        deconv0 = torch.cat([up0,conv2],-1)
        # print('deconv0',deconv0.size())
        deconv0,d4 = self.decoderlayer_0(deconv0,mask=mask)
        # print('deconv0',deconv0.size(),d4.shape)
        
        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv1],-1)
        deconv1,d5 = self.decoderlayer_1(deconv1,mask=mask)
        # print('deconv1',deconv1.size(),d5.shape)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2,conv0],-1)
        deconv2,d6 = self.decoderlayer_2(deconv2,mask=mask)
        # print('deconv2',deconv2.size(),d6.shape)
        deconv3 = self.finalconv(deconv2.permute(0,2,1))
        deconv3 = deconv3.permute(0,2,1)
        # print('deconv3',deconv3.size())

        # Output Projection
        y = self.output_projection(deconv3)
        x_y = self.softmax(x+y)

        cha_hist=[d0,d1,d2,d3,d4,d5,d6]
        hist_feaure = [deconv3.detach(),conv4.detach(),conv0.detach()]
        # print('..............')
        # print(cha_hist[0].shape,cha_hist[1].shape,cha_hist[2].shape)  
        # print(cha_hist[3].shape,cha_hist[4].shape,cha_hist[5].shape,cha_hist[6].shape)   
        # print('..............')
        return  x_y, cha_hist
        
