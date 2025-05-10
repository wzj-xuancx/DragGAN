import math
import pprint
import tkinter as tk
import numpy as np
import torch
from ttkthemes import ThemedTk
import customtkinter as ctk
from PIL import Image, ImageTk
from customtkinter import CTkImage
import sys
from tkinter import font
from utils import Draw_arrow, create_button, f_to_image, get_inverted_mask_matrix
sys.path.append('stylegan2')

# print(sys.path)
import os
import random
import threading
from backend import UI_Backend
from PIL import ImageDraw

class DragganWindow:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1200x800")
        self.model = UI_Backend(device='cuda')
        self.after_id = None  # 用于存储当前的 after 调用 ID
        self.last_points_image = []
        self.if_bind_on_image = -1 #只会在按下Strat Points/Mask的时候改变状态
        self.MAX_seed = 1000000000

        #drag过程
        self.if_drag = False #是否在拖拽，只会在按下Start Drag/Stop Drag的时候改变状态
        self.drag_step = 0
        
        #和drag后的gif有关
        self.gif_images = []
        self.gif_images_with_points = []
        self.gif_duration = 0

        #和点有关
        self.point_radius = 12
        self.points = []

        #和画笔有关
        # self.image_size = (512, 512)
        self.brush_size = 20  # 默认画笔大小
        self.brush_tool = None  # 默认画笔工具
        self.mask_image = None

        #xx 启动!
        self.setup_ui()
        # print(font.families(root))

    def create_status_bar(self):
        """创建顶部状态栏"""
        self.status_bar = ctk.CTkLabel(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 20, "bold"),
            height=40,
            anchor="center",
            fg_color="#2B2B2B",  # 背景颜色
            text_color="white"   # 文本颜色
        )
        self.status_bar.pack(fill=tk.X, side=tk.TOP)

    def update_status(self, message, d=1500):
        """更新状态栏消息"""
        
        # 取消之前的 after 调用
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)  

        self.status_var.set(message)

        self.after_id = self.root.after(d, self.reset_status) #等待d毫秒后重置状态栏

    def reset_status(self):
        """重置状态栏为默认消息"""
        self.status_var.set("WZJ_DragGAN")
        self.after_id = None  # 重置 after_id

    def setup_ui(self):
        """Material Design风格界面"""
        # 配置主题
        ctk.set_appearance_mode("System")  # 跟随系统主题
        ctk.set_default_color_theme("blue")  # 内置主题: blue, green, dark-blue

        # 初始化顶部状态栏
        self.status_var = tk.StringVar(value="WZJ_DragGAN")
        self.create_status_bar()        

        # 主框架布局
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        self.control_panel = ctk.CTkFrame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 右侧图像区域
        self.image_panel = ctk.CTkFrame(self.main_frame, width=912) #912-512=400
        self.image_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板组件
        self.create_controls()
        
        # 图像显示区域
        self.create_image_view()

    def create_controls(self):
        """具体控件的布局"""
        # 标题
        title_stylegan = ctk.CTkLabel(
            self.control_panel,
            text="StyleGAN",
            font=("Segoe UI", 20, "bold"),
            anchor="w"
        )
        title_stylegan.pack(fill=tk.X, pady=(0, 20))

        # 模型选择
        model_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        model_frame.pack(fill=tk.X, pady=5, expand=False)
        
        ctk.CTkLabel(model_frame, text="Model Weights:").pack(side=tk.LEFT)

        # 下拉菜单（如果有模型文件，自动加载第一个模型）
        model_files = self.get_model_files()
        default_model = model_files[0] if model_files else "No models available"

        self.model_var = tk.StringVar(value=default_model)
        self.model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            variable=self.model_var,
            values=model_files,
            command=self.load_selected_model
        )
        self.model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=5)
        if model_files:
            self.load_selected_model(default_model)
    

        # 种子设置
        seed_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        seed_frame.pack(fill=tk.X, pady=15)
        
        self.seed_var = tk.IntVar(value=0)
        self.seed_entry = ctk.CTkEntry(
            seed_frame,
            textvariable=self.seed_var,
            placeholder_text="Input seed"
        )
        self.seed_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.random_seed = ctk.CTkSwitch(
            seed_frame,
            text="Random",
            command=self.toggle_seed_entry
        )
        self.random_seed.pack(side=tk.RIGHT, padx=5)

        # 生成和保存按钮组
        gen_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        gen_frame.pack(fill=tk.X, pady=10)

        self.generate_btn = create_button(
            parent = gen_frame,
            text="Generate Image",
            command=self.generate_image,  
            fill=tk.X,  
            side=tk.LEFT        
        )
        self.save_btn = create_button(
            parent = gen_frame,
            text="Save Image",
            command=self.save_image,   
            fill=tk.X,
            side=tk.RIGHT       
        )

        # 标题
        title_draggan = ctk.CTkLabel(
            self.control_panel,
            text="DragGAN",
            font=("Segoe UI", 20, "bold"),
            anchor="w"
        )
        title_draggan.pack(fill=tk.X, pady=(40,10))

        # 标记点和撤销点按钮组
        drag_frame1 = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        drag_frame1.pack(fill=tk.X, pady=10)

        start_points_btn = create_button(
            parent = drag_frame1,
            text="Start Points",
            command=self.start_points, 
            fill=tk.X,  
            side=tk.LEFT           
        )

        undo_points_btn = create_button(
            parent = drag_frame1,
            text="Undo Last Point",
            command=self.undo_last_point,    
            fill=tk.X,
            side=tk.RIGHT      
        )

        drag_frame2 = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        drag_frame2.pack(fill=tk.X, pady=10)

        # 重置点和lr调整
        reset_points_btn = create_button(
            parent = drag_frame2,
            text="Reset Points",
            command=self.reset_points,    
            fill=tk.X,
            side=tk.LEFT     
        )

        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_entry = ctk.CTkEntry(
            drag_frame2,
            textvariable=self.lr_var,
            placeholder_text="Input lr",
        )
        self.lr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        drag_frame3 = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        drag_frame3.pack(fill=tk.X, pady=10)

        # 开始Drag
        start_drag_btn = create_button(
            parent = drag_frame3,
            text="Start Drag",
            command=self.start_drag,    
            fill=tk.X,
            side=tk.LEFT, 
        )

        # 停止Drag
        stop_drag_btn = create_button(
            parent = drag_frame3,
            text="Stop Drag",
            command=self.stop_drag,  
            fill=tk.X,
            side=tk.LEFT,
            padx=5 
        )

        # 保存生成过程的gif
        save_drag_gif_btn = create_button(
            parent = drag_frame3,
            text="Save Gif",
            command=self.save_drag_gif,
            fill=tk.X,
            side=tk.LEFT,
        )

        # 标记/撤销mask
        mask_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        mask_frame.pack(fill=tk.X, pady=10)

        start_points_btn = create_button(
            parent = mask_frame,
            text="Start Mask",
            command=self.start_mask, 
            fill=tk.X,  
            side=tk.LEFT           
        )

        undo_points_btn = create_button(
            parent = mask_frame,
            text="Reset Mask",
            command=self.reset_mask,    
            fill=tk.X,
            side=tk.RIGHT      
        )

        # 添加画笔工具和大小调节
        brush_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        brush_frame.pack(fill=tk.X, pady=10)

        # 调节画笔大小
        self.brush_size_var = tk.IntVar(value=10)
        brush_size_label = ctk.CTkLabel(brush_frame, text="Brush Size:")
        brush_size_label.pack(side=tk.LEFT, padx=5)
        brush_size_slider = ctk.CTkSlider(
            brush_frame,
            from_=1, to=50,
            variable=self.brush_size_var,
            command=lambda size: self.adjust_brush_size(int(size))
        )
        brush_size_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 选择画笔工具
        tool_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        tool_frame.pack(fill=tk.X, pady=10)

        circle_tool_btn = create_button(
            parent=tool_frame,
            text="Circle Tool",
            command=lambda: self.set_brush_tool("circle"),
            side=tk.LEFT,
            padx=5
        )
        rectangle_tool_btn = create_button(
            parent=tool_frame,
            text="Rectangle Tool",
            command=lambda: self.set_brush_tool("rectangle"),
            side=tk.RIGHT,
            padx=5
        )

    def get_now_image(self):
        """用于获取当前显示的图像（不加mask）"""
        print("get_now_image ", len(self.last_points_image))
        if len(self.last_points_image) == 0:
            return self.current_image_pil
        return self.last_points_image[-1]

    ### 画笔相关
    #全局绑定
    def set_bindings(self):
        """根据当前状态绑定或解绑事件"""
        # 先解绑所有事件
        if self.mask_image is None:
            self.mask_image = Image.new("RGBA", self.current_image_pil.size, (0, 0, 0, 0))
            self.mask_draw = ImageDraw.Draw(self.mask_image)
        
        self.image_label.unbind("<Button-1>")
        self.image_label.unbind("<B1-Motion>")
        self.image_label.unbind("<ButtonRelease-1>")
        self.last_drag_position = None  # 清除上一次拖动位置

        print(self.if_bind_on_image)

        # 根据状态绑定事件
        if self.if_bind_on_image == 0:
            # 点编辑模式
            self.image_label.bind("<Button-1>", self.on_click_points)
        elif self.if_bind_on_image == 1:
            # 拖动编辑模式
            self.image_label.bind("<B1-Motion>", self.on_drag)
            self.image_label.bind("<ButtonRelease-1>", self.drag_on_release)
        elif self.if_bind_on_image == 2:
            # 圆形编辑模式
            self.image_label.bind("<Button-1>", self.on_click_circle)
        elif self.if_bind_on_image == 3:
            # 矩形编辑模式
            self.image_label.bind("<Button-1>", self.on_click_rectangle)
    
    def on_click_circle(self, event):
        """圆形编辑模式的点击事件"""
        x, y = event.x, event.y
        img_x, img_y = int(x * 2), int(y * 2)
        self.start_position = (img_x, img_y)
        print(img_x, img_y)
        self.image_label.bind("<Motion>", self.on_motion_circle)
        self.image_label.bind("<ButtonRelease-1>", self.on_release_circle)

    def on_motion_circle(self, event):
        """圆形编辑模式的鼠标移动事件"""
        x, y = event.x, event.y
        img_x, img_y = int(x * 2), int(y * 2)
        shadow_image = self.get_now_image().convert("RGBA").copy()
        shadow_draw = ImageDraw.Draw(shadow_image)

        radius = int(((img_x - self.start_position[0]) ** 2 + (img_y - self.start_position[1]) ** 2) ** 0.5)
        shadow_draw.ellipse(
            (self.start_position[0] - radius, self.start_position[1] - radius,
            self.start_position[0] + radius, self.start_position[1] + radius),
            outline=(128, 128, 128, 128), width=2
        )

        combined_image = Image.alpha_composite(shadow_image, self.mask_image)

        self.current_image = CTkImage(light_image=combined_image, size=(512, 512))
        self.image_label.configure(image=self.current_image)

    def on_release_circle(self, event):
        """圆形编辑模式的鼠标释放事件"""
        x, y = event.x, event.y
        img_x, img_y = int(x * 2), int(y * 2)
        radius = int(((img_x - self.start_position[0]) ** 2 + (img_y - self.start_position[1]) ** 2) ** 0.5)
        self.mask_draw.ellipse(
            (self.start_position[0] - radius, self.start_position[1] - radius,
            self.start_position[0] + radius, self.start_position[1] + radius),
            fill=(128, 128, 128, 128)
        )
        combined_image = Image.alpha_composite(self.get_now_image().convert("RGBA"), self.mask_image)
        self.current_image = CTkImage(light_image=combined_image, size=(512, 512))
        self.image_label.configure(image=self.current_image)
        self.image_label.unbind("<Motion>")
        self.image_label.unbind("<ButtonRelease-1>")

    def on_click_rectangle(self, event):
        """矩形编辑模式的点击事件"""
        x, y = event.x, event.y
        img_x, img_y = int(x * 2), int(y * 2)
        self.start_position = (img_x, img_y)
        self.image_label.bind("<Motion>", self.on_motion_rectangle)
        self.image_label.bind("<ButtonRelease-1>", self.on_release_rectangle)

    def on_motion_rectangle(self, event):
        """矩形编辑模式的鼠标移动事件"""
        x, y = event.x, event.y
        img_x, img_y = int(x * 2), int(y * 2)

        # 确保坐标顺序正确
        x0, y0 = self.start_position
        x1, y1 = img_x, img_y
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        shadow_image = self.get_now_image().convert("RGBA").copy()
        shadow_draw = ImageDraw.Draw(shadow_image)
        shadow_draw.rectangle(
            (x0, y0, x1, y1),
            outline=(128, 128, 128, 128), width=2
        )
        combined_image = Image.alpha_composite(shadow_image, self.mask_image)

        self.current_image = CTkImage(light_image=combined_image, size=(512, 512))
        self.image_label.configure(image=self.current_image)

    def on_release_rectangle(self, event):
        """矩形编辑模式的鼠标释放事件"""
        x, y = event.x, event.y
        img_x, img_y = int(x * 2), int(y * 2)

        # 确保坐标顺序正确
        x0, y0 = self.start_position
        x1, y1 = img_x, img_y
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        self.mask_draw.rectangle(
            (x0, y0, x1, y1),
            fill=(128, 128, 128, 128)
        )
        combined_image = Image.alpha_composite(self.get_now_image().convert("RGBA"), self.mask_image)
        self.current_image = CTkImage(light_image=combined_image, size=(512, 512))
        self.image_label.configure(image=self.current_image)
        self.image_label.unbind("<Motion>")
        self.image_label.unbind("<ButtonRelease-1>")
   
    def on_drag(self, event):
        """鼠标拖动事件处理函数"""
        if self.current_image_pil is None or self.mask_image is None:
            self.update_status("No image or mask to draw on!", d=3000)
            return

        x, y = event.x, event.y
        if not (0 <= x <= 512 and 0 <= y <= 512):
            return

        # 转换为图像实际坐标，并确保是整数
        img_x, img_y = int(x * 2), int(y * 2)

        # 如果没有上一次位置，初始化为当前点
        if not hasattr(self, 'last_drag_position') or self.last_drag_position is None:
            self.last_drag_position = (img_x, img_y)

        # 绘制平滑的线条
        self.mask_draw.line(
            [self.last_drag_position, (img_x, img_y)],
            fill=(128, 128, 128, 128),  # 浅灰色半透明
            width=self.brush_size
        )

        # 更新上一次位置
        self.last_drag_position = (img_x, img_y)

        # 更新显示的图像
        combined_image = Image.alpha_composite(self.get_now_image().convert("RGBA"), self.mask_image)
        self.current_image = CTkImage(
            light_image=combined_image,
            size=(512, 512)
        )
        self.image_label.configure(image=self.current_image)

    def drag_on_release(self, event):
        """鼠标释放时清除上一次拖动的位置"""
        self.last_drag_position = None
  
    def start_mask(self):
        self.brush_tool = None
        """开始绘制Mask"""
        if not hasattr(self, 'current_image_pil') or self.current_image_pil is None:
            self.update_status("No image to start mask!", d=3000)
            return
        self.if_bind_on_image = 1
        self.update_status("Mask drawing started! Use the mouse to draw.", d=3000)
        self.set_bindings()

        pprint.pprint(self.mask_image)

    def reset_mask(self):
        """重置Mask"""
        if not hasattr(self, 'current_image_pil') or self.current_image_pil is None:
            self.update_status("No image to reset mask!", d=3000)
            return

        # 清除Mask
        self.last_drag_position = None  # 清除上一次拖动位置
        self.start_position = None  # 清除起始位置
        self.image_label.unbind("<Motion>")
        self.image_label.unbind("<B1-Motion>")
        self.image_label.unbind("<ButtonRelease-1>")
        self.current_image = CTkImage(
            light_image=self.get_now_image(),
            size=(512, 512)
        )
        self.image_label.configure(image=self.current_image)

        self.mask_image = Image.new("RGBA", self.current_image_pil.size, (0, 0, 0, 0))
        self.mask_draw = ImageDraw.Draw(self.mask_image)
        self.update_status("Mask has been reset!", d=3000)

    def adjust_brush_size(self, size):
        """调整画笔大小"""
        self.brush_size = size
        self.update_status(f"Brush size set to {size}!", d=2000)

    def set_brush_tool(self, tool):
        """设置画笔工具"""
        if self.brush_tool == tool:
            # 如果再次点击已选中的工具，则取消选中
            self.brush_tool = None
            self.if_bind_on_image = -1  # 重置为不可编辑状态
            self.update_status("Brush tool deselected!", d=2000)
        else:
            # 选中新的工具
            self.brush_tool = tool
            self.if_bind_on_image = 2 if tool == "circle" else 3
            self.update_status(f"Brush tool set to {tool}!", d=2000)

        # 更新绑定
        self.set_bindings()
    ###

    def reset_image_label(self):
        """重置图像显示区域"""
        if not hasattr(self, 'seed_text') or self.seed_text is None:
            return 
        self.seed_text.configure(text="Waiting for start...")
        self.current_image = CTkImage(
            light_image=Image.new("RGB", (512, 512), "#2B2B2B"),
            size=(512, 512)  # 必须指定size参数
        )
        self.image_label.configure(image=self.current_image)

    def create_image_view(self):
        # 显示种子文本
        self.seed_text = ctk.CTkLabel(
            self.image_panel,
            text="Now image Seed: ...",
            font=("Segoe UI", 18, "bold"),
            anchor="center"
        )
        self.seed_text.pack(fill=tk.X, pady=10)

        self.current_image = CTkImage(
            light_image=Image.new("RGB", (512, 512), "#2B2B2B"),
            size=(512, 512)  # 必须指定size参数
        )
        self.image_label = ctk.CTkLabel(
            self.image_panel, 
            image=self.current_image,
            text=""  # 清除默认文本
        )
        self.image_label.pack(expand=False, fill=tk.Y)

    def get_model_files(self):
        """扫描 ./models 目录下的所有合法模型文件"""
        model_dir = "./models"
        valid_extensions = (".pkl", ".pt", ".ckpt")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  # 如果目录不存在，则创建
        return [
            f for f in os.listdir(model_dir)
            if f.endswith(valid_extensions) and os.path.isfile(os.path.join(model_dir, f))
        ]

    def load_selected_model(self, selected_model):
        """加载用户选择的模型"""
        self.reset_image_label()
        model_path = os.path.join("./models", selected_model)
        if os.path.exists(model_path):
            self.update_status("Loading model...")
            self.model.load_ckpt(model_path)
        else:
            self.update_status("Model does not exist!")

    def toggle_seed_entry(self):
        """种子输入控制"""
        if self.random_seed.get():
            self.update_status("Random seed enabled!")
            self.seed_entry.configure(state="disabled") 
            try:
                self.seed_var.set(random.randint(0, self.MAX_seed)) #1e9不可行就换
            except:
                self.seed_var.set(random.randint(0, 65536))
        else:
            self.update_status("Random seed disabled!")
            self.seed_entry.configure(state="normal")

    def on_click_points(self, event):
        """鼠标点击事件处理函数"""

        # 获取点击位置（在显示区域中的坐标）
        x, y = event.x, event.y

        # 计算图像在显示区域中的实际显示尺寸
        display_width, display_height  = 512, 512

        # 判断点击位置是否在图像的实际显示范围内
        print(x, y)
        if not (0 <= x <= display_width and 0 <= y <= display_height):
            self.update_status("Click is outside the image!", d=3000)
            return

        # 转换为图像实际坐标
        img_x, img_y = int(x * 2), int(y * 2)

        # 更新
        self.click_count += 1
        color = "red" if self.click_count % 2 == 1 else "blue"
        self.points.append((img_x, img_y, color))

        # print(img_x, img_y, color)

        #更新image
        if len(self.last_points_image) == 0:
            self.last_points_image.append(self.current_image_pil)
            
        image_with_points = self.last_points_image[-1].copy()

        #开始绘制
        draw = ImageDraw.Draw(image_with_points)
        draw.ellipse(
            (img_x - self.point_radius, 
                img_y - self.point_radius, 
                img_x + self.point_radius, 
                img_y + self.point_radius),
            fill=color
        )

        # 如果有一对点（红点和蓝点），绘制箭头
        if len(self.points) >= 2 and self.points[-2][2] == "red" and self.points[-1][2] == "blue":
            red_point = self.points[-2]
            blue_point = self.points[-1]

            Draw_arrow(draw, red_point[0], red_point[1], 
                        blue_point[0], blue_point[1], 
                        self.point_radius, "red")

        # 更新图像显示
        myimage = image_with_points
        if self.mask_image is not None:
            myimage = Image.alpha_composite(image_with_points.convert("RGBA"), self.mask_image)
        
        self.current_image = CTkImage(
            light_image=myimage,
            size=(512, 512)  # 保持与显示区域一致
        )
        self.image_label.configure(image=self.current_image)
        self.last_points_image.append(image_with_points)

        # 更新状态栏
        self.update_status(f"Point marked at ({img_x}, {img_y}) in {color}!", d=2000)
    
    def start_points(self):
        """点击开始后，用户可以在图片上标记点，奇数次为红点，偶数次为蓝点"""
        if not hasattr(self, 'current_image_pil') or self.current_image_pil is None:
            self.update_status("No image to mark points!", d=3000)
            return

        self.reset_points() 
            
        # 清空标记点列表和点击计数器
        self.points = []
        self.click_count = 0
        # 绑定鼠标点击事件到图像显示区域，如果已经绑定就不需要绑定了
        # print(self.image_label.bindtags())
        if self.if_bind_on_image != 0:
            self.if_bind_on_image = 0
            self.set_bindings()

        self.update_status("Click on the image to mark points!", d=3000)

    def reset_points(self):
        """清除所有标记点"""
        if not hasattr(self, 'current_image_pil') or self.current_image_pil is None:
            self.update_status("No image to reset points!", d=3000)
            return

        # 清空标记点列表和点击计数器
        self.points = []
        self.click_count = 0

        # 恢复图像为原始图像
        myimage = self.current_image_pil.copy()
        if self.mask_image is not None:
            myimage = Image.alpha_composite(myimage.convert("RGBA"), self.mask_image)
        self.current_image = CTkImage(
            light_image=myimage,  # 使用原始图像
            size=(512, 512)  # 保持与显示区域一致
        )
        self.image_label.configure(image=self.current_image)
        self.last_points_image = []

        # 更新状态栏
        self.update_status("All points have been reset!", d=3000)

    def undo_last_point(self):
        """用于撤销上一点的操作,那么实际上会回到上上张图"""
        if self.click_count == 0:
            self.update_status("Please point on the image!")
            return 
        
        if self.mask_image is not None:
            myimage = Image.alpha_composite(self.last_points_image[-2].convert("RGBA"), self.mask_image)
        self.current_image = CTkImage(
            light_image=myimage,  # 使用原始图像
            size=(512, 512)  # 保持与显示区域一致
        )
        self.image_label.configure(image=self.current_image)

        self.click_count -= 1
        self.last_points_image.pop() #把刚刚得到的图去掉
        self.points.pop() #把刚刚得到的点去掉
        
    def dragimage_update(self, image, init_pts):
        """在更新的过程中实现异步更新图片（不然会闪烁）"""
        #不懂这里为什么还要判断？ 但是不判断会报错，不过报错不影响。
        if isinstance(image, torch.Tensor) == True:
            image = f_to_image(image)
        #注意last_points_image的更新
        self.last_points_image = []
   
        #每隔1张图保存一张gif 最后控制在30张左右即可
        self.gif_images.append(image)
        image_with_points = image.copy()

        for i in range(0, len(self.points), 2):
            #注意他是反过来的！！！！
            self.points[i] = (init_pts[i // 2][1] * 2, init_pts[i // 2][0] * 2, self.points[i][2]) #红点改变,
            #绘制红点
            draw = ImageDraw.Draw(image_with_points)
            draw.ellipse(
                (self.points[i][0] - self.point_radius, 
                self.points[i][1] - self.point_radius, 
                self.points[i][0] + self.point_radius, 
                self.points[i][1] + self.point_radius),
                fill="red"
            )
            self.last_points_image.append(image_with_points)
            #绘制蓝点和箭头
            if i + 1 < len(self.points):
                draw.ellipse((
                self.points[i+1][0] - self.point_radius,
                self.points[i+1][1] - self.point_radius,
                self.points[i+1][0] + self.point_radius,
                self.points[i+1][1] + self.point_radius),fill="blue")
                Draw_arrow(draw, self.points[i][0], self.points[i][1],
                            self.points[i+1][0], self.points[i+1][1],
                            self.point_radius, "red")
            self.last_points_image.append(image_with_points)
        # 更新图像显示
        self.current_image = CTkImage(
            light_image=self.last_points_image[-1],
            size=(512, 512)  # 保持与显示区域一致
        )
        self.gif_images_with_points.append(self.last_points_image[-1])
        self.image_label.configure(image=self.current_image)
   
    def start_drag(self):

        # '''
        # 464860682
        # 526404507
        # 972306413
        # '''
        # points1, points2 = get_pair_points("experiments/faces/face1.png",
        #                                    "experiments/faces/face2.png",
        #                                    "experiments/shape_predictor_68_face_landmarks.dat")
        # for i in range(len(points1)):
        #     self.points.append((points1[i][0], points1[i][1], "red"))
        #     self.points.append((points2[i][0], points2[i][1], "blue"))
        #     #更新image
        #     if len(self.last_points_image) == 0:
        #         self.last_points_image.append(self.current_image_pil)
                
        #     image_with_points = self.last_points_image[-1].copy()

        #     #开始绘制
        #     draw = ImageDraw.Draw(image_with_points)
        #     draw.ellipse(
        #         (points1[i][0] - self.point_radius, 
        #             points1[i][1] - self.point_radius, 
        #             points1[i][0] + self.point_radius, 
        #             points1[i][1] + self.point_radius),
        #         fill="red"
        #     )
        #     draw.ellipse(
        #         (points2[i][0] - self.point_radius, 
        #             points2[i][1] - self.point_radius, 
        #             points2[i][0] + self.point_radius, 
        #             points2[i][1] + self.point_radius),
        #         fill="blue"
        #     )

        #     # 如果有一对点（红点和蓝点），绘制箭头
        #     if len(self.points) >= 2 and self.points[-2][2] == "red" and self.points[-1][2] == "blue":
        #         red_point = self.points[-2]
        #         blue_point = self.points[-1]

        #         Draw_arrow(draw, red_point[0], red_point[1], 
        #                     blue_point[0], blue_point[1], 
        #                     self.point_radius, "red")

        #     # 更新图像显示
        #     myimage = image_with_points


        #     print(self.mask_image)
        #     print(image_with_points.convert("RGBA"))
        #     if self.mask_image is not None:
        #         myimage = Image.alpha_composite(image_with_points.convert("RGBA"), self.mask_image)
            
        #     self.current_image = CTkImage(
        #         light_image=myimage,
        #         size=(512, 512)  # 保持与显示区域一致
        #     )
        #     self.image_label.configure(image=self.current_image)
        #     self.last_points_image.append(image_with_points)

        # self.click_count = len(self.points)
        # #更新图像



        """开始拖拽"""
        if not hasattr(self, 'current_image_pil') or self.current_image_pil is None:
            self.update_status("No image to start drag!", d=3000)
            return

        if not hasattr(self, 'click_count') or self.click_count < 2:
            self.update_status("Please mark at least 2 points!", d=3000)
            return
        
        if self.lr_var.get() <= 0.0:
            self.update_status("lr must greater than 0")
            return 
        
        self.if_drag = True
        self.gif_images = [self.current_image_pil]
        self.gif_images_with_points = [self.last_points_image[-1]]

        #获得mask对应的矩阵
        mask_matrix = get_inverted_mask_matrix(self.mask_image)
        
        # 更新图像显示
        self.current_image = CTkImage(
            light_image=self.get_now_image(),
            size=(512, 512)  # 保持与显示区域一致
        )
        self.image_label.configure(image=self.current_image)

        def dragging_thread():
            # 将用户指定的点分成初始点和目标点
            init_pts = []
            tar_pts = []
            for i in range(0, len(self.points), 2):
                #缩放坐标，防止后续出错，从最大值为(1024,1024)缩放到最大值为(512,512)
                # print(len(self.points))
                init_pts.append((self.points[i][0] // 2, self.points[i][1] // 2))
                tar_pts.append((self.points[i+1][0] // 2, self.points[i+1][1] // 2))
            init_pts = np.vstack(init_pts)[:, ::-1].copy()
            tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
            
            #获取mask对应的矩阵

            lr = self.lr_var.get()
            print(lr)
            self.model.prepare_to_drag(init_pts, mask_matrix, lr)
            self.drag_step = 0 #步数
            while (self.if_drag):
                # 迭代一次
                self.update_status("Drag Step {0}".format(self.drag_step), d=1000)
                status, ret = self.model.drag(init_pts, tar_pts)
                if status:
                    init_pts, _, image = ret
                else:
                    self.update_status("Drag Done! Click 'Save Gif' and enjoy it!", d=3000)
                    self.if_drag = False
                    return
                image = f_to_image(image) #得到显示在屏幕上的图像
                # 异步更新图像
                self.root.after(0, lambda: self.dragimage_update(image, init_pts))
                self.drag_step += 1
        threading.Thread(target=dragging_thread, daemon=True).start()

    def stop_drag(self):
        self.update_status("Stop drag ...", d=3000)
        self.if_drag = False

    def save_drag_gif(self):
        self.update_status("Saving gif ...", d=10000)

        #将self.gif_images保存成gif，在目录./gif下
        if os.path.exists("./gif") == False:
            os.makedirs("./gif")

        if len(self.gif_images) == 0:
            return 
        

        if len(self.gif_images) % 35 > len(self.gif_images) % 45:
            d = len(self.gif_images) // 45
        else:
            d = len(self.gif_images) // 35
        #拿第一张，然后每d张拿一张，这样不会太多导致占用内存大
        self.gif_images = [self.gif_images[i] for i in range(0, len(self.gif_images), d)]
        #如果最后一张不是最后一张，就加上最后一张
        if len(self.gif_images) != len(self.gif_images) // d:
            self.gif_images.append(self.gif_images[-1])

        # print(len(self.gif_images)) #应该在30~45之间
        self.gif_images[0].save(f"./gif/seed_gif {self.seed_var.get()}.gif", 
                                save_all=True, append_images=self.gif_images, 
                                duration=100, loop=0)
            
        self.gif_images = []

        if len(self.gif_images_with_points) % 35 > len(self.gif_images_with_points) % 45:
            d = len(self.gif_images_with_points) // 45
        else:
            d = len(self.gif_images_with_points) // 35
        self.gif_images_with_points = [self.gif_images_with_points[i] for i in range(0, len(self.gif_images_with_points), d)]
        if len(self.gif_images_with_points) != len(self.gif_images_with_points) // d:
            self.gif_images_with_points.append(self.gif_images_with_points[-1])
        self.gif_images_with_points[0].save(f"./gif/seed_gif_with_points {self.seed_var.get()}.gif", 
                                save_all=True, append_images=self.gif_images_with_points, 
                                duration=100, loop=0)
            
        self.gif_images_with_points = []

        self.update_status("Gif saved successfully!", d=3000)

    def generate_image(self):
        #将生成按钮和保存按钮设置为不可用
        self.generate_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        """生成图像"""
        if self.random_seed.get():
            try:
                self.seed_var.set(random.randint(0, self.MAX_seed)) #1e9不可行就换
            except:
                self.seed_var.set(random.randint(0, 65536))
        
        #如果已经有点了，就需要全部清除一次
        if hasattr(self, 'points') and len(self.points) > 0:
            self.reset_points()     
        
        #如果有mask，就需要全部清除
        self.mask_image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        self.mask_draw = ImageDraw.Draw(self.mask_image)

        def update_image(image):
            """更新图像显示"""
            self.current_image_pil = image
            print(self.current_image_pil.size)

            #图片产生后就可以生成一张mask了
            # self.mask_image = Image.new("RGBA", self.current_image_pil.size, (0, 0, 0, 0))
            # self.mask_draw = ImageDraw.Draw(self.mask_image)

            self.current_image = CTkImage(
                light_image=image,
                size=(512, 512)  # 保持与显示区域一致
            )
            self.image_label.configure(image=self.current_image)
            self.seed_text.configure(
                text="-------seed {0} image has loaded, click save images to save it-------".format(self.seed_var.get())
            )
            self.update_status("Image generated successfully!", d=500)

        def enable_buttons():
            """在生成图像的时候恢复按钮状态"""
            self.generate_btn.configure(state="enabled")
            self.save_btn.configure(state="enabled")
            
        # 在后台线程中运行耗时任务
        def task():
            try:
                image = f_to_image(self.model.gen_img(self.seed_var.get()))
                self.mask_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
                self.mask_draw = ImageDraw.Draw(self.mask_image)
                self.root.after(0, lambda: update_image(image))
            except Exception:
                self.root.after(0, lambda: self.update_status(f"Error", d=3000))
            finally:
                # 恢复按钮状态
                self.root.after(0, lambda: enable_buttons())
        threading.Thread(target=task, daemon=True).start()
    
    def save_image(self):
        """保存当前图像"""
        if hasattr(self, 'current_image_pil') and self.current_image_pil:

            # 获取模型名和种子值
            model_name = self.model_var.get().replace(" ", "_")  # 替换空格为下划线
            seed_value = self.seed_var.get()

            # 确保保存目录存在
            save_dir = os.path.join("./images" , model_name)

            self.update_status("Saving image...", d=1000)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
           # 如果图片已经存在,就不用生成
            save_path = os.path.join(save_dir, f"seed{seed_value}.png")
            if os.path.exists(save_path):
                self.update_status("Image already exists!", d=3000)
                return

            # 保存图像
            self.current_image_pil.save(save_path)
            self.update_status("Image saved successfully!", d=3000)
        else:
            self.update_status("No image to save!", d=3000)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    root = ThemedTk(theme="arc")  # 使用现代主题
    app = DragganWindow(root)
    root.mainloop()