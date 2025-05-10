'''
Author: xuancx 1728321546@qq.com
Date: 2025-03-21 12:08:24
LastEditors: xuancx 1728321546@qq.com
LastEditTime: 2025-05-10 15:32:47
FilePath: /mydraggan/backend.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import copy
import os
import sys
import time

import utils
sys.path.append('stylegan2')

import cv2
import torch
from stylegan2.model import StyleGAN

class UI_Backend(object):
    def __init__(self, device='cuda'):
        self.model_path = None
        self.device = device
        self.generator = StyleGAN(self.device)
        self.mask_matrix = None
        self.beta = 10 #论文用的10
        self.reg = 0
        self.r1 = 3
        self.r2 = 12
        self.atol = 0
    
    def load_ckpt(self, model_path):
        """加载模型"""
        self.generator.load_ckpt(model_path)
        self.model_path = model_path
        
    def gen_img(self, seed):
        """调用生成器生成图像"""
        if self.model_path is not None:
            self.W = self.generator.gen_w(seed) 
            img, self.init_F = self.generator.gen_img(self.W)
            # print("self.init_F.shape", self.init_F.shape)
            return img
        else:
            # 如果模型还没加载则返回None
            # print("Please use load_ckpt first.")
            return None
        
    def prepare_to_drag(self, init_pts, mask_matrix ,lr=0.001, r1 = 3, r2 = 12, atol = 0):
        # 备份初始图像的特征图
        self.F0_resized = torch.nn.functional.interpolate(self.init_F, 
                                              size=(512, 512), 
                                              mode="bilinear",
                                              align_corners=True).detach().clone()
        # 备份初始点坐标
        temp_init_pts_0 = copy.deepcopy(init_pts)
        self.init_pts_0 = torch.from_numpy(temp_init_pts_0).float().to(self.device)
        
        # 将w向量的部分特征设置为可训练
        temp_W = self.W.cpu().numpy().copy()
        self.W = torch.from_numpy(temp_W).to(self.device).float()
        self.W.requires_grad_(False)
        
        self.W_layers_to_optimize = self.W[:, :6, :].detach().clone().requires_grad_(True)
        self.W_layers_to_fixed = self.W[:, 6:, :].detach().clone().requires_grad_(False)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam([self.W_layers_to_optimize], lr=lr)

        # 保存mask矩阵
        self.mask_matrix = mask_matrix

        # 保存r1, r2, atol
        self.r1 = r1
        self.r2 = r2
        self.atol = atol

    # 进行一次迭代优化
    def drag(self, _init_pts, _tar_pts, id = 0, radius = 360):  

        # time1 = time.time()

        init_pts = torch.from_numpy(_init_pts).float().to(self.device)
        tar_pts = torch.from_numpy(_tar_pts).float().to(self.device)

        # 如果起始点和目标点之间的像素误差足够小，则停止
        if torch.allclose(init_pts, tar_pts, atol=self.atol):
            return False, (None, None)
        
        # 将latent的0:6设置成可训练,6:设置成不可训练 See Sec3.2 in paper
        W_combined = torch.cat([self.W_layers_to_optimize, self.W_layers_to_fixed], dim=1)
        # 前向推理，为了拿到特征图
        new_img, _F = self.generator.gen_img(W_combined)

        # 计算motion supervision loss，See Sec3.2 in paper

        # time2 = time.time()
        F_resized = torch.nn.functional.interpolate(_F, size=(512, 512), mode="bilinear", align_corners=True)
        loss = self.motion_supervision(
            F_resized,
            init_pts, tar_pts)
        
        # # # 计算mask的loss（motion_supervision后的新特征图-没有运动前的原特征图）* (1-M),其中M是mask，没有mask即全1表示全部都可以修改:
        # mask_matrix_tensor = torch.from_numpy(self.mask_matrix).float().to(self.device)  # 转换为张量
        # mask_loss = torch.nn.functional.l1_loss(
        #     F_resized * (1 - mask_matrix_tensor), 
        #     self.F0_resized.detach() * (1 - mask_matrix_tensor)
        # )

        # 总损失
        # loss += self.beta * mask_loss

        # loss += self.reg * torch.nn.functional.l1_loss(W_combined, self.W)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # time3 = time.time()
        # 更新初始点 see Sec3.3 Point Tracking
        with torch.no_grad():
            # 以上过程会优化一次latent, 直接用新的latent生成图像，拿到新的特征图，用来进行point_tracking过程
            new_img, F_for_point_tracking = self.generator.gen_img(W_combined)
            F_for_point_tracking_resized = torch.nn.functional.interpolate(F_for_point_tracking, size=(512, 512), 
                                                               mode="bilinear", align_corners=True).detach()
            new_init_pts = self.point_tracking(F_for_point_tracking_resized, init_pts, tar_pts, id, radius)

        # time4 = time.time()

        # print("time1:{} time2:{} time3:{} time4:{}".format(time1, time2, time3, time4))

        return True, (new_init_pts.detach().clone().cpu().numpy(), tar_pts.detach().clone().cpu().numpy(), new_img)
    
    def motion_supervision(self, F, init_pts, tar_pts):
        n = init_pts.shape[0]
        loss = 0.0
        h, w = F.shape[2], F.shape[3]
        time_s, time_e = 0, 0
        for i in range(n):
            
            # time1 = time.time()
            
            dir_vec = tar_pts[i] - init_pts[i]
            d_i = dir_vec / (torch.norm(dir_vec) + 1e-7)
            if torch.norm(d_i) > torch.norm(dir_vec):
                d_i = dir_vec

            
            # time2 = time.time()
            mask = utils.create_circular_mask(h, w, init_pts[i].tolist(), self.r1).to(self.device)
            
            # time3 = time.time()
            
            coordinates = torch.nonzero(mask).float()
            shifted_coordinates = coordinates + d_i[None]
            F_qi = F[:, :, mask]
            
            norm_shifted_coordinates = shifted_coordinates.clone()
            norm_shifted_coordinates[:, 0] = (2.0 * shifted_coordinates[:, 0] / (w - 1)) - 1  # x
            norm_shifted_coordinates[:, 1] = (2.0 * shifted_coordinates[:, 1] / (h - 1)) - 1  # y
            norm_shifted_coordinates = norm_shifted_coordinates.unsqueeze(0).unsqueeze(0).clamp(-1, 1)
            norm_shifted_coordinates = norm_shifted_coordinates.flip(-1)
            F_qi_plus_di = torch.nn.functional.grid_sample(F, norm_shifted_coordinates, mode="bilinear", align_corners=True)
            F_qi_plus_di = F_qi_plus_di.squeeze(2)

            loss += torch.nn.functional.l1_loss(F_qi.detach(), F_qi_plus_di)

        #     time4 = time.time()
        #     time_s += time3 - time2
        #     time_e += time4 - time1
            
        # print(time_s, time_e, time_s / time_e )
        return loss
    
    # init_pts -> new init_pts -> ... -> tar_pts
    def point_tracking(self, F, init_pts, tar_pts, id = 0, radius = 360):
        n = init_pts.shape[0]
        new_init_pts = torch.zeros_like(init_pts)
        timee = 0
        for i in range(n):
            # time1 = time.time()
            # 以初始点为中心生成一个正方形mask,
            if id == 0:
                patch = utils.create_square_mask(F.shape[2], F.shape[3], 
                                                init_pts[i].tolist(), self.r2).to(self.device)
            elif id == 1:
                patch = utils.create_sector_mask(F.shape[2], F.shape[3], 
                                                  init_pts[i].tolist(), tar_pts[i].tolist(), radius, self.r2).to(self.device)
            
            # patch = utils.create_circular_mask(F.shape[2], F.shape[3], 
            #                                   init_pts[i].tolist(), r2).to(self.device)

            #得到所有不为0的索引
            patch_coordinates = torch.nonzero(patch)  # shape [num_points, 2]

            eps = 0.1
            while patch_coordinates.shape[0] < 1:
                # 如果没有mask区域，扩大mask区域
                if id == 0:
                    patch = utils.create_square_mask(F.shape[2], F.shape[3], 
                                                    init_pts[i].tolist(), self.r2 + 1).to(self.device)
                elif id == 1:
                    patch = utils.create_sector_mask(F.shape[2], F.shape[3], 
                                                      init_pts[i].tolist(), tar_pts[i].tolist(), radius + eps, self.r2 + 1).to(self.device)
                patch_coordinates = torch.nonzero(patch)
                eps += 0.5
            # time2 = time.time()
                
            # 拿到mask区域的特征
            F_qi = F[:, :, patch_coordinates[:, 0], patch_coordinates[:, 1]] # [N, C, num_points] torch.Size([1, 128, 729])
            # 旧初始点的特征
            self.init_pts_0[i][0] = min(max(self.init_pts_0[i][0], 0), F.shape[2] - 1)
            self.init_pts_0[i][1] = min(max(self.init_pts_0[i][1], 0), F.shape[3] - 1)
            f_i = self.F0_resized[:, :, self.init_pts_0[i][0].long(), self.init_pts_0[i][1].long()] # [N, C, 1]    
            # 计算mask内每个特征与老的初始点对应特征的距离
            distances = (F_qi - f_i[:, :, None]).abs().mean(1) # [N, num_points] torch.Size([1, 729])
            # distances = torch.linalg.norm(F_qi - f_i[:, :, None], dim=1)
            # 找到距离最小的，也就是老的初始点对应特征最像的特征
            # 如果
            min_index = torch.argmin(distances)
            new_init_pts[i] = patch_coordinates[min_index] # [row, col] 
        # print("timee: ", timee)
        return new_init_pts
    