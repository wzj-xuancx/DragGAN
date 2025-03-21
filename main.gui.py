import os
import tkinter as tk
import threading

from ttkthemes import ThemedTk
from draggan_gui import DragganWindow 
from backend import UI_Backend

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("WZJ_DragGAN")

        self.model = UI_Backend(device='cuda')

        # 设定加载状态
        self.is_loaded = False

        # 加载ui
        self.create_main_ui()

        # 加载模型
        self.load_model_in_background()

        self.after_id = None  # 用于存储当前的 after 调用 ID

    def update_status(self, message, fg="black", d=1500):
        """更新状态栏消息"""
        
        # 取消之前的 after 调用
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)  

        self.status_label.config(
            text=message,
            fg=fg
        )

        self.after_id = self.root.after(d, self.reset_status) #等待d毫秒后重置状态栏

    def reset_status(self):
        """重置状态栏为默认消息"""
        self.status_label.set("Initializing ModernGANApp...")
        self.after_id = None  # 重置 after_id

    def create_main_ui(self):
        """创建大组件"""
        # 标题
        self.title_label = tk.Label(
            self.root,
            text="WZJ_DragGAN",
            font=("Segoe UI", 30, "bold"),
            fg="black"
        )
        self.title_label.pack(pady=50)

        # 状态信息
        self.status_label = tk.Label(
            self.root,
            text="Loading, please wait...",
            font=("Segoe UI", 16),
            fg="gray"
        )
        self.status_label.pack(pady=20)

        # 进入按钮
        self.enter_button = tk.Button(
            self.root,
            text="Enter",
            font=("Segoe UI", 18, "bold"),
            bg="#3B8ED0",
            fg="white",
            command=self.enter_main_app
        )
        self.enter_button.pack(pady=20)

        #说明按钮
        self.info_button = tk.Button(
            self.root,
            text="About",
            font=("Segoe UI", 18, "bold"),
            bg="#3B8ED0",
            fg="white",
            command=self.show_info
        )
        self.info_button.pack(pady=20)

        # 关闭界面
        close_button = tk.Button(
            self.root,
            text="Close",
            font=("Segoe UI", 18, "bold"),
            bg="#3B8ED0",
            fg="white",
            command=self.root.destroy
        )
        close_button.pack(pady=20)

    def load_model_in_background(self):
        """加载模型"""
        def load_model():
            # 初始化模型，避免等会太卡
            try:
                self.status_label.config(text="Initializing WZJ_DragGAN...")

                model_dir = "./models"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)  # 如果目录不存在，则创建

                self.model.load_ckpt("./models/generator_baby-stylegan2-config-f.pkl")
                self.model.gen_img(0)

                # Mark as loaded
                self.is_loaded = True
                self.status_label.config(text="Model loaded successfully", fg="green")
            except Exception as e:
                self.status_label.config(text=f"Failed to load model: {e}", fg="red")

        threading.Thread(target=load_model, daemon=True).start()

    def show_info(self):
        """打开新窗口显示说明"""
        info = """
WZJ_DragGAN

This is a GUI application based on StyleGAN2, which can generate images of cartoon characters.

Developed by WZJ, this tool allows users to interactively manipulate generated images.

How to use:
1. Wait for the model to finish loading. The status will display "Model loaded successfully".
2. Click the "Enter" button to open the main application interface.
3. In the main interface, you can:
    - Mark points on the image by clicking.
    - Drag points to manipulate the generated image.
    - Reset points if needed.
4. Use the "About" button to view this information again.
        """
        # Create a new window for the information
        info_window = tk.Toplevel(self.root)
        info_window.title("About WZJ_DragGAN")
        info_window.geometry("800x600")

        # Add a label to display the information
        info_label = tk.Label(
            info_window,
            text=info,
            font=("Segoe UI", 14),
            wraplength=600,
            justify="left"
        )
        info_label.pack(pady=20, padx=20)
        
    def enter_main_app(self):
        """进入主界面的前置"""
        if not self.is_loaded:
            self.update_status("Waiting model load...", "red")
            return

        self.root.destroy()  

        new_root = ThemedTk(theme="arc")  # 使用现代主题
        app = DragganWindow(new_root)
        new_root.mainloop()  
        
if __name__ == "__main__":
    #用一个进程创建主页面，先隐藏，后面直接打开即可，加快速度。
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()