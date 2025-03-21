import os
import sys
sys.path.append('stylegan2')

import torch
from stylegan2.model import StyleGAN

class UI_Backend(object):
    def __init__(self):
        self.model_path = None
        self.device = 'cuda'
        self.generator = StyleGAN(self.device)
    
    def load_ckpt(self, model_path):
        """加载模型"""
        self.generator.load_ckpt(model_path)
        self.model_path = model_path
        
    def gen_img(self, seed):
        """调用生成器生成图像"""
        if self.model_path is not None:
            self.W = self.generator.gen_w(seed) 
            img, self.init_F = self.generator.gen_img(self.W)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            print("已经获得种子为{0}图像!".format(seed))
            return img
        else:
            # 如果模型还没加载则返回None
            print("Please use load_ckpt first.")
            return None