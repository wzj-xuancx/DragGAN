'''
Author: xuancx 1728321546@qq.com
Date: 2025-04-21 13:39:27
LastEditors: xuancx 1728321546@qq.com
LastEditTime: 2025-05-10 15:34:05
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

def get_pair_points(face_path1 = "faces/face1.png", 
                    face_path2 = "faces/face2.png", 
                    predictor_path = "shape_predictor_68_face_landmarks.dat"):

    detector = dlib.get_frontal_face_detector() #获取人脸分类器
    predictor = dlib.shape_predictor(predictor_path)    # 获取人脸检测器

    img1 = cv2.imread(face_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(face_path2, cv2.IMREAD_COLOR)

    img1 = cv2.resize(img1, (1024, 1024))
    img2 = cv2.resize(img2, (1024, 1024))

    # 摘自官方文档：
    # image is a numpy ndarray containing either an 8bit grayscale or RGB image.
    # opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
    b1, g1, r1 = cv2.split(img1)    # 分离三个颜色通道
    img11 = cv2.merge([r1, g1, b1])   # 融合三个颜色通道生成新图片

    b2, g2, r2 = cv2.split(img2)    # 分离三个颜色通道
    img22 = cv2.merge([r2, g2, b2])   # 融合三个颜色通道生成新图片

    #如果图片是512*512的，要先resize成1024*1024

    dets1 = detector(img11, 1) #使用detector进行人脸检测 dets为返回的结果
    dets2 = detector(img22, 1) #使用detector进行人脸检测 dets为返回的结果

    points1 = []
    points2 = []

    for index, face in enumerate(dets1):
        shape = predictor(img1, face)  # 寻找人脸的68个标定点 
        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            print(index, pt_pos)
            points1.append(pt_pos)

    for index, face in enumerate(dets2):
        shape = predictor(img2, face)  # 寻找人脸的68个标定点 
        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            print(index, pt_pos)
            points2.append(pt_pos)

    return points1, points2

def get_image():
    seed = random.randint(0, 100000)
    # print(seed)
    image = model.gen_img(seed)
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = Image.fromarray(image[0].cpu().numpy(), 'RGB')
    return image

def shiyan_get_pair_points(img1, img2):

    detector = dlib.get_frontal_face_detector() #获取人脸分类器
    predictor = dlib.shape_predictor("./experiments/shape_predictor_68_face_landmarks.dat")    # 获取人脸检测器

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

def get_fid():
    real_images_folder = "./experiments/real_images_204radius"  # 真实图像文件夹路径
    generated_images_folder = "./experiments/generated_images_204radius"  # 生成图像文件夹路径


    # 计算FID距离值
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], device='cuda', batch_size=8, dims=2048)
    return fid_value

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # model = UI_Backend(device='cuda')
    # model.load_ckpt("models/stylegan2-ffhq-512x512.pkl") #利用ffhq模型进行测试
    # print("模型加载完成")
    # #随机生成两张图片
    # image1, image2 = get_image(), get_image()
    # image1.save("./experiments/generated_images/{}.png".format(1))
    # image2.save("./experiments/real_images/{}.png".format(1))
    # fid = get_fid()
    # print("fid:{0}".format(fid))


    # 读取图片
    model = UI_Backend(device='cuda')
    model.load_ckpt("models/stylegan2-ffhq-512x512.pkl") #利用ffhq模型进行测试
    print("加载模型完成!!")
    md = 0
    fid = 0
    id = 100
    time_all = 0
    for j in range(0,id):
        print("start:  ", j)
        #随机生成两张图片
        image2 = get_image()
        image1 = get_image() #现在背后的特征图是image1的，所以image1才会动，而image2不会动

        # image1.save("./experiments/generated_images/11.png")
        # image2.save("./experiments/real_images/11.png")

        points1, points2 = shiyan_get_pair_points(image1, image2)

        while len(points1) != len(points2) and len(points1) != 68:
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

        model.prepare_to_drag(init_pts, np.ones((512, 512), dtype=np.uint8), 0.001, atol=1, r1=7, r2=12.5)
        # model.prepare_to_drag(init_pts, np.ones((512, 512), dtype=np.uint8), 0.001)
        drag_step = 0 #步数
        if_drag = True
        image = image1

        #开始计时
        time1 = time.time()
        while (if_drag and drag_step < 300):
            # print("第{}轮，第{}步".format(j, drag_step))
            status, ret = model.drag(init_pts, tar_pts, id = 1, radius = 204)
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

        image.save("./experiments/generated_images_204radius/{}.png".format(j + 1))
        image2.save("./experiments/real_images_204radius/{}.png".format(j + 1))

        print("md: {0}".format(md))

    fid = get_fid()
    md = md / id / 68
    time_all = time_all / id
    
    print("time: {0}".format(time_all))
    print("fid:{0}, md:{1}".format(fid, md))
    #保存md和fid到文件
    with open("./experiments/md_fid.txt", "a") as f:
        f.write("md: {0}, fid: {1}\n".format(md, fid))






    
    