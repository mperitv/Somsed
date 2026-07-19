import tkinter as tk
import customtkinter as ctk
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
        self.axis_range = 10
        self.init_hardware()
        self.benchmark_device()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
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

        self.bottom_frame = ctk.CTkFrame(
            self.root,
            height=150
        )

        self.bottom_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew"
        )
        self.bottom_frame.grid_propagate(False)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=3)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(
            self.bottom_frame,
            width=200,
        )  
        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=5,
            pady=5
        )
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_columnconfigure(1, weight=2)

        self.hardware_label = ctk.CTkLabel(
            self.sidebar,
            text=f"Hardware Accelerator: {self.device.type.upper()}",
                font=("Arial", 12, "bold")
        )
        self.hardware_label.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(8,4)
        )

        self.precision_slider = ctk.CTkSlider(
            self.sidebar,
            from_=200,
            to=3000,
            command=self.update_time,
        )
        self.precision_slider.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=10
        )

        self.precision_title_label = ctk.CTkLabel(
            self.sidebar,
            text="Epoch: 200",
            font=("Arial", 10, "bold")
        )
        self.precision_title_label.grid(
            row=2,
            column=0,
            sticky="w",
            padx=10
        )

        self.estimated_time_label = ctk.CTkLabel(
            self.sidebar,
            text="Estimated Time: N/A",
            font=("Arial", 10, "italic"),
        )
        self.estimated_time_label.grid(
            row=3,
            column=1,
            sticky="w",
            padx=10,
            pady=(0,8)
        )

        self.optimize_button = ctk.CTkButton(
            self.sidebar,
            text="Optimize",
            command=self.optimize_curve,
            font=("Arial", 12, "bold"),
        )
        self.optimize_button.grid(
            row=1,
            column=0,
            padx=(10,5),
            pady=8,
            sticky="ew"
        )

        self.clear_button = ctk.CTkButton(
            self.sidebar,
            text="Clear",
            command=self.clear_canvas,
            font=("Arial", 12, "bold"),
        )
        self.clear_button.grid(
            row=1,
            column=1,
            padx=(5,10),
            pady=8,
            sticky="ew"
        )

        self.canvas = tk.Canvas(
            self.root,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=5,
            pady=5
        )

        self.root.after(100, self.draw_grid)

        self.log_console = ctk.CTkTextbox(
            self.bottom_frame,
            height=140,
            font=("Consolas", 11)
        )
        self.log_console.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=5,
            pady=5,
        )
        self.log_console.configure(state="disabled")

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.update_time(self.precision_slider.get())

    def draw_grid(self):
        self.root.update_idletasks()
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        spacing = 25

        center_x = width // 2
        center_y = height // 2

        for x in range(center_x, width, spacing):
            self.canvas.create_line(x, 0, x, height, fill="#d9d9d9")

        for x in range(center_x, 0 , -spacing):
            self.canvas.create_line(x, 0, x, height, fill="#d9d9d9")

        for y in range(center_y, height, spacing):
            self.canvas.create_line(0, y, width, y, fill="#d9d9d9")

        for y in range(center_y, 0, -spacing):
            self.canvas.create_line(0, y, width, y, fill="#d9d9d9")
        
        self.canvas.create_line(
            width / 2, 0,
            width / 2, height,
            fill="black",
            width=2
        )

        self.canvas.create_line(
            0, height / 2,
            width, height / 2,
            fill="black",
            width=2
        )

    def canvas_to_math(self, x, y):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        scale_x = (2 * self.axis_range) / width
        scale_y = (2 * self.axis_range) / height

        math_x = (x - width / 2) * scale_x
        math_y = (height / 2 - y) * scale_y

        return math_x, math_y

    def log(self, message):
        self.log_console.configure(state="normal")
        self.log_console.insert(tk.END, message + "\n")
        self.log_console.see(tk.END)
        self.log_console.configure(state="disabled")

    def update_time(self, value):
        epochs = round(value / 100) * 100
        self.precision_slider.set(epochs)
        self.precision_title_label.configure(
            text=f"Epoch: {epochs}"
        )
        calculated_time = epochs * self.time_coefficient
        self.estimated_time_label.configure(
            text=f"Estimated Time: {calculated_time:.2f} s"
        )

    def start_draw(self, event):
        mx, my = self.canvas_to_math(event.x, event.y)

        self.points.append((event.x, event.y))
        self.log(f"Point drawn at: ({mx:.1f}, {my:.1f})")
    
    def draw(self, event):
        x, y = event.x, event.y
        prev_x, prev_y = self.points[-1]
        self.canvas.create_line(prev_x, prev_y, x, y, fill="black", width=3, capstyle=tk.ROUND, smooth=True)
        self.points.append((x, y))
    
    def optimize_curve(self):
        self.log("Starting curve optimization...")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.points.clear()
        self.log_console.configure(state="normal")
        self.log_console.delete(1.0, tk.END)
        self.log_console.configure(state="disabled")
        print("Canvas cleared")
        self.log("Canvas cleared.")
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = SomsedApp(root)
    root.mainloop()
