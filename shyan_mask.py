'''
Author: xuancx 1728321546@qq.com
Date: 2025-04-21 13:39:27
LastEditors: xuancx 1728321546@qq.com
LastEditTime: 2025-05-10 15:55:11
FilePath: /zijun/code/bishe/mydraggan/experiments/shiyan.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-
import random
import sys
import time
import dlib
import cv2
import os
import math
import pprint
import tkinter as tk
import numpy as np
import torch
import torchvision
from ttkthemes import ThemedTk
import customtkinter as ctk
from PIL import Image, ImageTk
from customtkinter import CTkImage
from tkinter import font
import torchvision.transforms as transforms
from pytorch_fid import fid_score

print(sys.path)
import torch # 存放人脸图片的路径
from backend import UI_Backend

def get_image():
    seed = random.randint(0, 100000)
    # print(seed)
    image = model.gen_img(seed)
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = Image.fromarray(image[0].cpu().numpy(), 'RGB')
    return image

def shiyan_get_pair_points(img1, img2):

    detector = dlib.get_frontal_face_detector() #获取人脸分类器
    predictor = dlib.shape_predictor("./experiments/shape_predictor_5_face_landmarks.dat")    # 获取人脸检测器

    img1 = np.array(img1)  # 转换为 numpy 数组
    img2 = np.array(img2)

    dets1 = detector(img1, 1) #使用detector进行人脸检测 dets为返回的结果
    dets2 = detector(img2, 1) #使用detector进行人脸检测 dets为返回的结果

    points1 = []
    points2 = []

    for index, face in enumerate(dets1):
        shape = predictor(img1, face)  # 寻找人脸的68个标定点 
        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            # print(index, pt_pos)
            points1.append(pt_pos)

    for index, face in enumerate(dets2):
        shape = predictor(img2, face)  # 寻找人脸的68个标定点 
        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            # print(index, pt_pos)
            points2.append(pt_pos)

    return points1, points2

def get_fid(radius):
    real_images_folder = "./experiments/radius_real{}".format(radius)  # 真实图像文件夹路径
    generated_images_folder = "./experiments/radius_gen{}".format(radius)  # 生成图像文件夹路径
    if os.path.exists(real_images_folder) == False:
        os.makedirs(real_images_folder)
    if os.path.exists(generated_images_folder) == False:
        os.makedirs(generated_images_folder)
    # 计算FID距离值
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], device='cuda', batch_size=8, dims=2048)
    return fid_value

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    radius_list = np.linspace(10, 360, 10) 

    # 读取图片
    model = UI_Backend(device='cuda')
    model.load_ckpt("models/stylegan2-ffhq-512x512.pkl") #利用ffhq模型进行测试
    print("加载模型完成!!")

    for radius in radius_list:
        md = 0
        fid = 0
        id = 50
        time_all = 0
        for j in range(0,id):
            print("start:  ", j)
            #随机生成两张图片
            image2 = get_image()
            image1 = get_image() #现在背后的特征图是image1的，所以image1才会动，而image2不会动

            points1, points2 = shiyan_get_pair_points(image1, image2)

            while len(points1) != len(points2) and len(points1) != 5:
                print("检测到的点数不相等，发生错误!!!重新生成图片{0}".format(j))
                print("points1: {0}, points2: {1}".format(len(points1), len(points2)))
                print("points1: {0}, points2: {1}".format(points1, points2))

                image2 = get_image()
                image1 = get_image() #现在背后的特征图是image1的，所以image1才会动，而image2不会动
                points1, points2 = shiyan_get_pair_points(image1, image2)

            init_pts = []
            tar_pts = []
            for i in range(0, len(points1)):
                init_pts.append((points1[i][0], points1[i][1]))
                tar_pts.append((points2[i][0], points2[i][1]))
            init_pts = np.vstack(init_pts)[:, ::-1].copy()
            tar_pts = np.vstack(tar_pts)[:, ::-1].copy()

            model.prepare_to_drag(init_pts, np.ones((512, 512), dtype=np.uint8), 0.001)
            drag_step = 0 #步数
            if_drag = True
            image = image1

            #开始计时
            time1 = time.time()
            while (if_drag and drag_step < 300):
                # print("第{}轮, drag step: {}".format(j, drag_step))
                status, ret = model.drag(init_pts, tar_pts, id = 1, radius=radius)
                if status:
                    init_pts, _, image = ret
                else:
                    if_drag = False
                    break
                # 异步更新图像
                drag_step += 1      
            #结束计时
            time2 = time.time()
            time_all += time2 - time1
            print("drag step: {0}, time: {1}, timeall: {2}".format(drag_step, time2 - time1, time_all))
            #完成drag后计算md和fid分数
            #md计算方法：红蓝点之间的距离和
            for i in range(len(init_pts)):
                md += math.sqrt((init_pts[i][0] - tar_pts[i][0])**2 + (init_pts[i][1] - tar_pts[i][1])**2)
            #保存到文件夹
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(image[0].cpu().numpy(), 'RGB')

            if os.path.exists("./experiments/radius_real{}".format(radius)) == False:
                os.makedirs("./experiments/radius_real{}".format(radius))
            if os.path.exists("./experiments/radius_gen{}".format(radius)) == False:
                os.makedirs("./experiments/radius_gen{}".format(radius))

            image.save("./experiments/radius_real{}/{}.png".format(radius, j + 1))
            image2.save("./experiments/radius_gen{}/{}.png".format(radius, j + 1))

            print("md: {0}".format(md))

        fid = get_fid(radius)
        time_all = time_all
        
        print("time: {0}".format(time_all))
        print("fid:{0}, md:{1}".format(fid, md))
        #对于每一个(radius)保存md,fid,time_all到文件
        with open("./experiments/radius.txt", "a") as f:
            f.write("(radius: {}), md: {}, fid: {}, time: {}\n".format(radius, md, fid, time_all))






    
    