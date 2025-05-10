'''
Author: xuancx 1728321546@qq.com
Date: 2025-03-24 13:09:08
LastEditors: xuancx 1728321546@qq.com
LastEditTime: 2025-05-10 15:29:19
FilePath: /mydraggan/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import math
from PIL import Image
import customtkinter as ctk
import numpy as np
import torch

def Draw_arrow(draw, x, y, xx, yy, r, color):
    """
    在圆心为(x,y)的圆画一个箭头指向圆心为(xx,yy)的圆
    箭头的长度为r，颜色为color,箭头的大小为sz
    """
    #得到箭头的角度
    dx = xx - x
    dy = yy - y
    angle = math.atan2(dy, dx)

    #绘制直线
    start = (x + r * math.cos(angle), y + r * math.sin(angle))
    end = (xx - r * math.cos(angle), yy - r * math.sin(angle))
    draw.line([start, end], fill=color, width=3)

    # 调整箭头头部的位置，使其在蓝色小球外部s
    arrow_base_x = xx - r * math.cos(angle)  
    arrow_base_y = yy - r * math.sin(angle)

    # 根据直线的长度定义箭头头部的大小([10,30]之间的大小)
    s,t = 10,30
    len = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    # print(len)
    sz = len * (t - s) / (1024 * math.sqrt(2))  + s

    # 箭头头部的两个点
    arrow_x1 = arrow_base_x - sz * math.cos(angle - math.pi / 6)
    arrow_y1 = arrow_base_y - sz * math.sin(angle - math.pi / 6)
    arrow_x2 = arrow_base_x - sz * math.cos(angle + math.pi / 6)
    arrow_y2 = arrow_base_y - sz * math.sin(angle + math.pi / 6)
    
    # 绘制箭头头部
    draw.polygon(
        [(arrow_base_x, arrow_base_y), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)],
        fill=color
    )
    
def create_circular_mask(h, w, center, r):
    """创建一个h*w的圆形mask区域，以起始点为中心，r为半径"""
    cy, cx = center
    
    # 预分配mask数组
    mask = np.zeros((h, w), dtype=np.bool_)
    
    # 计算边界以减少计算量
    y_min = max(int(cy - r), 0)
    y_max = min(int(cy + r) + 1, h)
    x_min = max(int(cx - r), 0)
    x_max = min(int(cx + r) + 1, w)
    
    # 生成区域内的坐标
    y = np.arange(y_min, y_max)
    x = np.arange(x_min, x_max)
    
    # 使用广播计算距离的平方
    y_diff = y - cy
    x_diff = x - cx
    yy, xx = np.meshgrid(y_diff, x_diff, indexing='ij')
    dist_squared = yy**2 + xx**2
    
    # 判断是否在圆内（避免开平方）
    mask[y_min:y_max, x_min:x_max] = dist_squared <= r**2
    
    return torch.from_numpy(mask).to(torch.bool)

def create_square_mask(h, w, center, r):
    """高效创建h*w的方形mask区域，以center为中心，边长为2*r+1"""
    # 提取中心点坐标并转换为整数
    cy, cx = map(int, center)
    r = int(r)
    
    # 计算方形的边界，确保不超出图像范围
    y_min = max(0, cy - r)
    y_max = min(h, cy + r + 1)  # +1是因为切片操作不包含结束索引
    x_min = max(0, cx - r)
    x_max = min(w, cx + r + 1)
    
    # 创建全零mask并设置方形区域为1
    mask = np.zeros((h, w), dtype=np.bool_)
    mask[y_min:y_max, x_min:x_max] = True
    
    return torch.from_numpy(mask).to(torch.bool)

def create_sector_mask(h, w, center, to, radius, r2):
    """高效创建扇形mask，保持原函数的精度和结果"""
    # 提取中心点和目标点坐标
    cy, cx = map(int, center)
    ty, tx = map(int, to)
    
    # 预计算扇形角度的一半（弧度）
    angle_rad_half = np.deg2rad(radius) / 2
    
    # 预计算目标方向的正弦和余弦值
    dx_target = tx - cx
    dy_target = ty - cy
    target_dist = np.hypot(dx_target, dy_target)
    
    # 处理中心点和目标点重合的情况
    if target_dist < 1e-7:
        # 如果两点重合，创建一个圆形mask
        y, x = np.ogrid[:h, :w]
        distance_squared = (x - cx) ** 2 + (y - cy) ** 2
        max_radius_squared = min(r2, 1.0) ** 2
        return torch.from_numpy(distance_squared <= max_radius_squared).to(torch.bool)
    
    # 归一化目标方向向量
    sin_target = dy_target / target_dist
    cos_target = dx_target / target_dist
    
    # 预计算最大距离的平方
    max_radius = min(r2, target_dist)
    max_radius_squared = max_radius ** 2
    
    # 创建坐标网格
    y, x = np.ogrid[:h, :w]
    dx = x - cx
    dy = y - cy
    
    # 高效计算角度条件 - 使用点积替代arctan2
    dot_product = dx * cos_target + dy * sin_target
    dist = np.hypot(dx, dy)
    
    # 避免除以零
    mask = dist > 0
    cos_angle = np.zeros_like(dist)
    cos_angle[mask] = dot_product[mask] / dist[mask]
    
    # 余弦值阈值（cos(angle_rad_half)）
    cos_threshold = np.cos(angle_rad_half)
    angle_cond = cos_angle >= cos_threshold
    
    # 距离条件（使用平方避免开平方运算）
    radius_cond = (dx ** 2 + dy ** 2) <= max_radius_squared
    
    # 逻辑组合生成掩码
    return torch.from_numpy(np.logical_and(angle_cond, radius_cond)).to(torch.bool)


def f_to_image(img):
    """生成的图像转换为显示在屏幕上的图像"""
    image = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = Image.fromarray(image[0].cpu().numpy(), 'RGB')
    # #如果是512*512的图像，需要转化为1024*1024的图像
    if image.size[0] != 1024:
        image = image.resize((1024, 1024), Image.LANCZOS)
    return image



def create_button(parent, text, command, font=("Segoe UI", 18, "bold"), height=40, corner_radius=8, fg_color="#3B8ED0", hover_color="#2C6BA0", padx=0, pady=0, side=None, fill=None):
    """通用按钮创建函数"""
    def on_button_hover(button, hover_color):
        """鼠标悬停时改变按钮颜色"""
        if button.cget("state") == "disabled":
            return
        button.configure(fg_color=hover_color)

    def on_button_leave(button, default_color):
        """鼠标离开时恢复按钮颜色"""
        if button.cget("state") == "disabled":
            return
        button.configure(fg_color=default_color)

    button = ctk.CTkButton(
        parent,
        text=text,
        command=command,
        font=font,
        height=height,
        corner_radius=corner_radius,
        fg_color=fg_color
    )
    button.pack(padx=padx, pady=pady, side=side, fill=fill)
    button.bind("<Enter>", lambda e: on_button_hover(button, hover_color))
    button.bind("<Leave>", lambda e: on_button_leave(button, fg_color))
    return button

def get_inverted_mask_matrix(mask_image):
    """获取反转的 mask 矩阵：mask 区域为 0，没有 mask 的区域为 1"""
    if mask_image is None:
        #返回一个全1的512*512矩阵
        return np.ones((512, 512), dtype=np.uint8)

    resized_mask_image = mask_image.resize((512, 512), Image.NEAREST)
    # 将 mask_image 转换为 NumPy 数组
    mask_array = np.array(resized_mask_image)

    # 提取 Alpha 通道（第四个通道）
    if mask_array.shape[-1] == 4:  # 确保是 RGBA 图像
        alpha_channel = mask_array[:, :, 3]

    # 生成二值化矩阵：Alpha 通道 > 0 的区域为 0，其他区域为 1
    inverted_mask_matrix = np.where(alpha_channel > 0, 0, 1).astype(np.uint8)

    return inverted_mask_matrix
