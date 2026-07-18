import tkinter as tk
from tkinter import messagebox
import torch
import platform

class SomsedEngineApp:
    def __init__(self, window_root):
        self.window_root = window_root
        self.window_root.title("Somsed")
        self.window_root.geometry("1000x700")
        self.raw_drawn_points = []
        self.configure_hardware_accelerator()
        self.initialize_ui()

    def configure_hardware_accelerator(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA acceleration.")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS acceleration.")
        else:
            self.device = torch.device("cpu")
            is_macos_architecture = platform.system() == "Darwin"
            if not is_macos_architecture:
                print("\n" + "=" * 80)
                print("GPU acceleration is unavailable. The application will continue to run on the CPU. Performance may be significantly slower.")
                print("Installing the appropriate GPU drivers is recommended.")
                print("https://developer.nvidia.com/cuda/toolkit")
                print("=" * 80 + "\n")
            else:
                print("MPS acceleration is unavailable. The application will continue running on the CPU. Performance may be significantly slower.")
    def initialize_ui(self):

        self.sidebar_control_panel = tk.Frame(
            self.window_root,
            width=200,
            bg="lightgray",
            relief=tk.RAISED,
            bd=2
        )  
        self.sidebar_control_panel.pack(side=tk.LEFT, fill=tk.Y)

        self.main_content_frame = tk.Frame(
            self.window_root,
            bg="white"
        )
        self.main_content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.hardware_label = tk.Label(
            self.sidebar_control_panel,
            text=f"Hardware Accelerator: {self.device.type.upper()}",
                bg="lightgray",
                font=("Arial", 12, "bold")
        )
        self.hardware_label.pack(pady=15, padx=10)

        self.precision_title_label = tk.Label(
            self.sidebar_control_panel,
            text="Epoch",
            bg="#1e1e24",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.precision_title_label.pack(pady=(20, 0), padx=10)

        self.precision_slider = tk.Scale(
            self.sidebar_control_panel,
            from_=200,
            to=3000,
            orient=tk.HORIZONTAL,
            resolution=100,
            command=self.on_precision_change,
            bg="#1e1e24",
            fg="white",
            highlightthickness=0,
            troughcolor="#3e3e42",
        )
        self.precision_slider.pack(pady=5, padx=10, fill=tk.X)

        self.estimated_time_label = tk.Label(
            self.sidebar_control_panel,
            text="Estimated Time: N/A",
            font=("Arial", 10, "italic"),
            bg="#1e1e24",
            fg="#a4b0be"
        )
        self.estimated_time_label.pack(pady=5, padx=10, anchor=tk.W)

        self.optimize_button = tk.Button(
            self.sidebar_control_panel,
            text="Optimize",
            command=self.execute_curve_optimization,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.optimize_button.pack(pady=30, padx=10, fill=tk.X)

        self.clear_button = tk.Button(
            self.sidebar_control_panel,
            text="Clear",
            command=self.clear_canvas,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.clear_button.pack(pady=10, padx=10, fill=tk.X)

        self.canvas = tk.Canvas(
            self.main_content_frame,
            width=800,
            height=500,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.log_console = tk.Text(
            self.main_content_frame,
            height=10,
            bg="#1e1e24",
            fg="white",
            font=("Courier New", 10),
            bd=0,
            padx=10,
            pady=10,
        )
        self.log_console.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_console.config(state=tk.DISABLED)

        self.canvas.bind("<Button-1>", self.on_continuous_draw)
        self.canvas.bind("<B1-Motion>", self.on_continuous_coordinate_draw)
        self.on_performance_change(self.precision_slider.get())
        
