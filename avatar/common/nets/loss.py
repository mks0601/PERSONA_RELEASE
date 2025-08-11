import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import lpips
from utils.smpl_x import smpl_x
from utils.transforms import get_neighbor
from config import cfg

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_tgt, mask=None, bg=None, window_size=11):
        batch_size, feat_dim, img_height, img_width = img_out.shape

        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_tgt, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_tgt*img_tgt, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_tgt, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

# image perceptual loss (LPIPS. https://github.com/richzhang/PerceptualSimilarity)
class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def forward(self, img_out, img_tgt):
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_tgt = img_tgt * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_tgt)
        return loss

class ImgLoss(nn.Module):
    def __init__(self):
        super(ImgLoss, self).__init__()
        self.ssim = SSIM()
        self.lpips = LPIPS()

    def forward(self, img_out, img_tgt):
        rgb_loss = torch.abs(img_out - img_tgt) * cfg.rgb_loss_weight
        ssim_loss = (1 - self.ssim(img_out, img_tgt)) * cfg.ssim_loss_weight
        lpips_loss = self.lpips(img_out, img_tgt) * cfg.lpips_loss_weight
        loss = rgb_loss + ssim_loss + lpips_loss
        return loss

# https://github.com/facebookresearch/sapiens/blob/3e829ac27476e4a70b6a01f85e487492afe02df1/seg/mmseg/models/losses/metric_silog_loss.py
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_out, depth_tgt):
        batch_size = depth_out.shape[0]

        loss = 0
        for i in range(batch_size):
            # get foreground
            is_valid = (depth_out[i] > 0) * (depth_tgt[i] > 0)
            depth_out_i = depth_out[i][is_valid]
            depth_tgt_i = depth_tgt[i][is_valid]

            # normalize in [0,1]
            depth_out_normalized = (depth_out_i - torch.min(depth_out_i)) / (torch.max(depth_out_i) - torch.min(depth_out_i) + 1e-4)
            depth_tgt_normalized = (depth_tgt_i - torch.min(depth_tgt_i)) / (torch.max(depth_tgt_i) - torch.min(depth_tgt_i) + 1e-4)
            is_valid = (depth_out_normalized > 0) * (depth_tgt_normalized > 0)
            depth_out_normalized, depth_tgt_normalized = depth_out_normalized[is_valid], depth_tgt_normalized[is_valid]

            # compute loss
            diff_log = torch.log(depth_tgt_normalized) - torch.log(depth_out_normalized)
            diff_log_mean = torch.mean(diff_log)
            diff_log_sq_mean = torch.mean(diff_log**2)
            loss += torch.sqrt(diff_log_sq_mean - 0.5*diff_log_mean**2)
        loss = loss / batch_size
        return loss

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, normal_out, normal_tgt):
        is_valid_out = 1 - ((normal_out==0).sum(1)[:,None] == 3).float()
        is_valid_tgt = 1 - ((normal_tgt==0).sum(1)[:,None] == 3).float()
        normal_out = normal_out*2-1 # [0,1] -> [-1,1]
        normal_tgt = normal_tgt*2-1 # [0,1] -> [-1,1]
        loss = (torch.abs(normal_out - normal_tgt) + (1 - torch.sum(normal_out*normal_tgt,1)[:,None])) * is_valid_out * is_valid_tgt
        return loss

class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, seg_out, seg_tgt):
        is_fg = 1 - ((seg_tgt==0).sum(1)[:,None] == 3).float()
        loss = torch.abs(seg_out - seg_tgt) * is_fg
        return loss

class GeoLoss(nn.Module):
    def __init__(self):
        super(GeoLoss, self).__init__()
        self.depth_loss = DepthLoss()
        self.normal_loss = NormalLoss()
        self.seg_loss = SegLoss()

    def forward(self, depth_out, normal_out, seg_out, depth_tgt, normal_tgt, seg_tgt):
        depth_loss = self.depth_loss(depth_out, depth_tgt) * cfg.depth_loss_weight
        normal_loss = self.normal_loss(normal_out, normal_tgt)
        seg_loss = self.seg_loss(seg_out, seg_tgt)
        loss = depth_loss + normal_loss + seg_loss
        return loss

class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, face):
        super(LaplacianReg, self).__init__()
        neighbor_idxs, neighbor_weights = get_neighbor(vertex_num, face)
        self.neighbor_idxs, self.neighbor_weights = neighbor_idxs.cuda(), neighbor_weights.cuda()

    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        lap = x + (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        return lap

    def forward(self, out, tgt):
        if tgt is None:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_tgt = self.compute_laplacian(tgt, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_tgt) ** 2
            return loss

# total variation regularizer
class TVReg(nn.Module):
    def __init__(self):
        super(TVReg, self).__init__()

    def forward(self, out, mask=None):
        if mask is None:
            mask = torch.ones_like(out)
        w_variance = (torch.pow(out[:,:,:,:-1] - out[:,:,:,1:], 2) * mask[:,:,:,:-1] * mask[:,:,:,1:]).mean((1,2,3))
        h_variance = (torch.pow(out[:,:,:-1,:] - out[:,:,1:,:], 2) * mask[:,:,:-1,:] * mask[:,:,1:,:]).mean((1,2,3))
        loss = (h_variance + w_variance)
        return loss

