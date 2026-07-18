import tkinter as tk
from tkinter import messagebox
import torch
import platform
import time

class SomsedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Somsed")
        self.root.geometry("1000x700")
        self.points = []
        self.init_hardware()
        self.benchmark_device()
        self.init_ui()

    def init_hardware(self):
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
    def benchmark_device(self):
        x = torch.randn(100, 15, device=self.device)
        w = torch.randn(15, 1, device=self.device)
        for _ in range(5):
            torch.mm(x, w)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            torch.mm(x, w)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        self.time_coefficient = (t1 - t0) / 100
    def init_ui(self):

        self.sidebar = tk.Frame(
            self.root,
            width=200,
            bg="lightgray",
            relief=tk.RAISED,
            bd=2
        )  
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.main_content_frame = tk.Frame(
            self.root,
            bg="white"
        )
        self.main_content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.hardware_label = tk.Label(
            self.sidebar,
            text=f"Hardware Accelerator: {self.device.type.upper()}",
                bg="lightgray",
                font=("Arial", 12, "bold")
        )
        self.hardware_label.pack(pady=15, padx=10)

        self.precision_title_label = tk.Label(
            self.sidebar,
            text="Epoch",
            bg="#1e1e24",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.precision_title_label.pack(pady=(20, 0), padx=10)

        self.precision_slider = tk.Scale(
            self.sidebar,
            from_=200,
            to=3000,
            orient=tk.HORIZONTAL,
            resolution=100,
            command=self.update_time,
            bg="#1e1e24",
            fg="white",
            highlightthickness=0,
            troughcolor="#3e3e42",
        )
        self.precision_slider.pack(pady=5, padx=10, fill=tk.X)

        self.estimated_time_label = tk.Label(
            self.sidebar,
            text="Estimated Time: N/A",
            font=("Arial", 10, "italic"),
            bg="#1e1e24",
            fg="#a4b0be"
        )
        self.estimated_time_label.pack(pady=5, padx=10, anchor=tk.W)

        self.optimize_button = tk.Button(
            self.sidebar,
            text="Optimize",
            command=self.optimize_curve,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.optimize_button.pack(pady=30, padx=10, fill=tk.X)

        self.clear_button = tk.Button(
            self.sidebar,
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

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.update_time(self.precision_slider.get())

    def log(self, message):
        self.log_console.config(state=tk.NORMAL)
        self.log_console.insert(tk.END, message + "\n")
        self.log_console.see(tk.END)
        self.log_console.config(state=tk.DISABLED)

    def update_time(self, value):
        epochs = int(value)
        calculated_time = epochs * self.time_coefficient
        self.estimated_time_label.config(text=f"Estimated Time: {calculated_time:.2f} seconds")
        self.log(f"Epochs set to: {epochs}. Estimated time: {calculated_time:.2f} seconds.")
    
    def start_draw(self, event):
        self.points.append((event.x, event.y))
        self.log(f"Point drawn at: ({event.x}, {event.y})")
    
    def draw(self, event):
        x, y = event.x, event.y
        prev_x, prev_y = self.points[-1]
        self.canvas.create_line(prev_x, prev_y, x, y, fill="black", width=3, capstyle=tk.ROUND, smooth=True)
        self.points.append((x, y))
    
    def optimize_curve(self):
        self.log("Starting curve optimization...")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.points.clear()
        self.log_console.config(state=tk.NORMAL)
        self.log_console.delete(1.0, tk.END)
        self.log_console.config(state=tk.DISABLED)
        print("Canvas cleared")
        self.log("Canvas cleared.")
if __name__ == "__main__":
    root = tk.Tk()
    app = SomsedApp(root)
    root.mainloop()
