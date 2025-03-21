import math
import tkinter as tk
from ttkthemes import ThemedTk
import customtkinter as ctk
from PIL import Image, ImageTk
from customtkinter import CTkImage
import sys
from tkinter import font
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
        self.if_bind_on_image = False #只会在按下Strat Points/Mask的时候改变状态
        self.MAX_seed = 1000000000

        #和点有关
        self.point_radius = 12
        self.points = []

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

        def on_button_hover(button, hover_color):
            """鼠标悬停时改变按钮颜色"""
            if button.cget("state") == "disabled":  # 使用 cget 获取按钮的状态
                return
            button.configure(fg_color=hover_color)

        def on_button_leave(button, default_color):
            """鼠标离开时恢复按钮颜色"""
            if button.cget("state") == "disabled":  # 使用 cget 获取按钮的状态
                return
            button.configure(fg_color=default_color)

        # 标题
        title_stylegan = ctk.CTkLabel(
            self.control_panel,
            text="StyleGAN",
            font=("Segoe UI", 23, "bold"),
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

        # 生成按钮
        self.generate_btn = ctk.CTkButton(
            self.control_panel,
            text="Generate Image",
            command=self.generate_image,
            font=("Segoe UI", 18, "bold"),
            height=40,
            corner_radius=10,
            fg_color="#3B8ED0"  # 默认颜色
        )
        self.generate_btn.pack(fill=tk.X, pady=20)
        self.generate_btn.bind("<Enter>", lambda e: on_button_hover(self.generate_btn, "#2C6BA0"))  # 悬停颜色
        self.generate_btn.bind("<Leave>", lambda e: on_button_leave(self.generate_btn, "#3B8ED0"))  # 恢复默认颜色
       
        # 保存按钮
        self.save_btn = ctk.CTkButton(
            self.control_panel,
            text="Save Image",
            command=self.save_image,
            font=("Segoe UI", 18, "bold"),
            height=40,
            corner_radius=10,
            fg_color="#3B8ED0"  # 默认颜色
        )
        self.save_btn.pack(fill=tk.X, pady=10)
        self.save_btn.bind("<Enter>", lambda e: on_button_hover(self.save_btn, "#2C6BA0"))  # 悬停颜色
        self.save_btn.bind("<Leave>", lambda e: on_button_leave(self.save_btn, "#3B8ED0"))  # 恢复默认颜色

        # 标题
        title_draggan = ctk.CTkLabel(
            self.control_panel,
            text="DragGAN",
            font=("Segoe UI", 23, "bold"),
            anchor="w"
        )
        title_draggan.pack(fill=tk.X, pady=(40,10))

        # 标记点和撤销点按钮组
        mark_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        mark_frame.pack(fill=tk.X, pady=10)

        self.start_points_btn = ctk.CTkButton(
            mark_frame,
            text="Start Points",
            command=self.start_points,
            font=("Segoe UI", 18, "bold"),
            height=40,
            corner_radius=8,
            fg_color="#3B8ED0"
        )
        self.start_points_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.start_points_btn.bind("<Enter>", lambda e: on_button_hover(self.start_points_btn, "#2C6BA0"))  # 悬停颜色
        self.start_points_btn.bind("<Leave>", lambda e: on_button_leave(self.start_points_btn, "#3B8ED0"))  # 恢复默认颜色
        
        self.undo_points_btn = ctk.CTkButton(
            mark_frame,
            text="Undo Last Point",
            command=self.undo_last_point,
            font=("Segoe UI", 18, "bold"),
            height=40,
            corner_radius=8,
            fg_color="#3B8ED0"
        )
        self.undo_points_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        self.undo_points_btn.bind("<Enter>", lambda e: on_button_hover(self.undo_points_btn, "#2C6BA0"))  # 悬停颜色
        self.undo_points_btn.bind("<Leave>", lambda e: on_button_leave(self.undo_points_btn, "#3B8ED0"))  # 恢复默认颜色

        # 重置点按钮单独放置
        self.reset_points_btn = ctk.CTkButton(
            self.control_panel,
            text="Reset Points",
            command=self.reset_points,
            font=("Segoe UI", 18, "bold"),
            height=40,
            corner_radius=10,
            fg_color="#3B8ED0"
        )
        self.reset_points_btn.pack(fill=tk.X, pady=10)
        self.reset_points_btn.bind("<Enter>", lambda e: on_button_hover(self.reset_points_btn, "#2C6BA0"))  # 悬停颜色
        self.reset_points_btn.bind("<Leave>", lambda e: on_button_leave(self.reset_points_btn, "#3B8ED0"))  # 恢复默认颜色

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
            self.seed_var.set(0)

    def start_points(self):
        """点击开始后，用户可以在图片上标记点，奇数次为红点，偶数次为蓝点"""
        if not hasattr(self, 'current_image_pil') or self.current_image_pil is None:
            self.update_status("No image to mark points!", d=3000)
            return

        self.reset_points()

        def on_click(event):
            """鼠标点击事件处理函数"""

            # 获取点击位置（在显示区域中的坐标）
            x, y = event.x, event.y

            # img_width, img_height = self.current_image_pil.size
            # label_width = self.image_label.winfo_width()
            # label_height = self.image_label.winfo_height()

            # print(x, y, img_width, img_height, label_width, label_height)

            # 计算图像在显示区域中的缩放比例
            # scale = min(label_width / img_width, label_height / img_height)

            # 计算图像在显示区域中的实际显示尺寸
            display_width, display_height  = 512, 512

            # 判断点击位置是否在图像的实际显示范围内
            print(x, y)
            if not (0 <= x <= display_width and 0 <= y <= display_height):
                self.update_status("Click is outside the image!", d=3000)
                return

            # 转换为图像实际坐标
            img_x = int(x * 2)
            img_y = int(y * 2)

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

                arrow_size = 20
                dx = blue_point[0] - red_point[0]
                dy = blue_point[1] - red_point[1]
                angle = math.atan2(dy, dx)

                #绘制直线
                draw.line(
                    [(red_point[0], red_point[1]), 
                     (blue_point[0] - self.point_radius * math.cos(angle), 
                      blue_point[1] - self.point_radius * math.sin(angle))],
                    fill="red",
                    width=3
                )

                # 调整箭头头部的位置，使其在蓝色小球外部
                arrow_base_x = blue_point[0] - self.point_radius * math.cos(angle)  # self.point_radius 是蓝色小球的半径
                arrow_base_y = blue_point[1] - self.point_radius * math.sin(angle)

                # 箭头头部的两个点
                arrow_x1 = arrow_base_x - arrow_size * math.cos(angle - math.pi / 6)
                arrow_y1 = arrow_base_y - arrow_size * math.sin(angle - math.pi / 6)
                arrow_x2 = arrow_base_x - arrow_size * math.cos(angle + math.pi / 6)
                arrow_y2 = arrow_base_y - arrow_size * math.sin(angle + math.pi / 6)
                
                # 绘制箭头头部
                draw.polygon(
                    [(arrow_base_x, arrow_base_y), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)],
                    fill="red"
                )

            # 更新图像显示
            self.current_image = CTkImage(
                light_image=image_with_points,
                size=(512, 512)  # 保持与显示区域一致
            )
            self.image_label.configure(image=self.current_image)
            self.last_points_image.append(image_with_points)

            # 更新状态栏
            self.update_status(f"Point marked at ({img_x}, {img_y}) in {color}!", d=2000)
            
        # 清空标记点列表和点击计数器
        self.points = []
        self.click_count = 0
        # 绑定鼠标点击事件到图像显示区域，如果已经绑定就不需要绑定了
        print(self.image_label.bindtags())
        if self.if_bind_on_image == False:
            self.image_label.bind("<Button-1>" ,on_click)
            self.if_bind_on_image = True
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
        self.current_image = CTkImage(
            light_image=self.current_image_pil,  # 使用原始图像
            size=(512, 512)  # 保持与显示区域一致
        )
        self.image_label.configure(image=self.current_image)
        self.last_points_image = []

        # 更新状态栏
        self.update_status("All points have been reset!", d=3000)

    def undo_last_point(self):
        """用于撤销上一点的操作,那么实际上会回到上上张图"""
        if len(self.last_points_image) < 2:
            self.update_status("Please point on the image!")
            return 
        self.current_image = CTkImage(
            light_image=self.last_points_image[-2],  # 使用原始图像
            size=(512, 512)  # 保持与显示区域一致
        )
        self.last_points_image.pop() #把刚刚得到的图去掉
        self.image_label.configure(image=self.current_image)

    def generate_image(self):
        #将生成按钮和保存按钮设置为不可用
        self.generate_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        """生成图像"""
        if self.random_seed.get():
            self.seed_var.set(random.randint(0, 65535)) 

        # 更新文本
        self.seed_text.configure(text="Loading seed {0} image...".format(self.seed_var.get()))        
        self.update_status("Generating image...", d=1000)

        def update_image(image):
            """更新图像显示"""
            self.current_image_pil = image
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
                # 耗时操作
                image = self.model.gen_img(self.seed_var.get())
                image = Image.fromarray(image[0].cpu().numpy(), 'RGB')

                # 更新 UI 必须在主线程中完成
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