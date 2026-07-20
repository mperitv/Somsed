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
        self.pixel_points = []
        self.math_points = []
        self.axis_range = 10
        self.min_distance = 3
        self.stabilization = 0.1
        self.last_filtered_point = None
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

        pixel_spacing = width / (2 * self.axis_range)

        center_x = width // 2
        center_y = height // 2

        for x in range(int(center_x), width, int(pixel_spacing)):
            self.canvas.create_line(x, 0, x, height, fill="#d9d9d9")
        
        for i in range(-self.axis_range, self.axis_range + 1):
            x = center_x + i * pixel_spacing

            self.canvas.create_text(
                x,
                center_y + 15,
                text=str(i),
                font=("Arial, 16"),
                fill="black"
            )

        for x in range(int(center_x), 0 , -int(pixel_spacing)):
            self.canvas.create_line(x, 0, x, height, fill="#d9d9d9")

        for i in range(-self.axis_range, self.axis_range + 1):
            x = center_x + i * pixel_spacing

            self.canvas.create_text(
                x,
                center_y + 15,
                text=str(i),
                font=("Arial, 16"),
                fill="black"
            )

        for y in range(int(center_y), height, int(pixel_spacing)):
            self.canvas.create_line(0, y, width, y, fill="#d9d9d9")

        for i in range(-self.axis_range, self.axis_range + 1):
            y = center_y - i * pixel_spacing

            if i != 0:
                self.canvas.create_text(
                    center_x - 15,
                    y,
                    text=str(i),
                    font=("Arial", 16),
                    fill="black"
                )

        for y in range(int(center_y), 0, -int(pixel_spacing)):
            self.canvas.create_line(0, y, width, y, fill="#d9d9d9")
        
        for i in range(-self.axis_range, self.axis_range + 1):
            y = center_y - i * pixel_spacing

            if i != 0:
                self.canvas.create_text(
                    center_x - 15,
                    y,
                    text=str(i),
                    font=("Arial", 16),
                    fill="black"
                )
        
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

        x_scale = (2 * self.axis_range) / width
        y_scale = (2 * self.axis_range) / width

        math_x = (x - width / 2) * x_scale
        math_y = (height / 2 - y) * y_scale

        return math_x, math_y
    
    def filter_point(self, x, y):
        if self.last_filtered_point is None:
            self.last_filtered_point = (x, y)
            return x, y
        
        last_x, last_y = self.last_filtered_point

        filtered_x = (
            self.stabilization * x + (1 - self.stabilization) * last_x
        )

        filtered_y = (
            self.stabilization * y + (1 - self.stabilization) * last_y
        )

        self.last_filtered_point = (filtered_x, filtered_y)
        return filtered_x, filtered_y

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
        self.last_filtered_point = None
        fx, fy = self.filter_point(event.x, event.y)
        mx, my = self.canvas_to_math(fx, fy)
        self.pixel_points.append((fx, fy))
        self.math_points.append((mx, my))
        self.log(f"Point drawn at: ({mx:.1f}, {my:.1f})")
    
    def draw(self, event):
        fx, fy = self.filter_point(event.x, event.y)
        mx, my = self.canvas_to_math(fx, fy)
        prev_x, prev_y = self.pixel_points[-1]
        distance = ((fx - prev_x) ** 2 + (fy - prev_y) ** 2) ** 0.5
        if distance < self.min_distance:
            return
        self.pixel_points.append((fx, fy))
        self.math_points.append((mx, my))
        self.canvas.create_line(
            prev_x, prev_y,
            fx, fy,
            fill="black",
            width=3,
            capstyle=tk.ROUND,
            smooth=True
        )

    def smooth_points(self, window=21):
        if len(self.math_points) < window:
            return self.math_points
        
        smoothed = []

        for i in range(len(self.math_points)):
            start = max(0, i - window // 2)
            end = min(len(self.math_points), i + window // 2 + 1)

            avg_x = sum(p[0] for p in self.math_points[start:end]) / (end - start)
            avg_y = sum(p[1] for p in self.math_points[start:end]) / (end - start)

            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def perpendicular_distance(self, point, start, end):
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end

        if start == end:
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

        return numerator / denominator
    
    def douglas_peucker(self, points, epsilon):
        if len(points) < 3:
            return points
        
        max_distance = 0
        index = 0

        for i in range(1, len(points) - 1):
            distance = self.perpendicular_distance(
                points[i],
                points[0],
                points[-1]
            )

            if distance > max_distance:
                max_distance = distance
                index = i

        if max_distance > epsilon:
            left = self.douglas_peucker(points[:index + 1], epsilon)
            right = self.douglas_peucker(points[index:], epsilon)

            return left[:-1] + right
        
        return [points[0], points[-1]]

    def math_to_canvas(self, x, y):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        x_scale = width / (2 * self.axis_range)
        y_scale = width / (2 * self.axis_range)

        canvas_x = width / 2 + x * x_scale
        canvas_y = height / 2 - y * y_scale

        return canvas_x, canvas_y
    
    def is_function(self, tolerance=0.1):

        if len(self.math_points) < 2:
            return False

        bins = {}

        for x, y in self.math_points:
            key = round(x / tolerance)

            if key not in bins:
                bins[key] = y
            else:
                if abs(bins[key] - y) > tolerance:
                    return False
                
        return True

    def optimize_curve(self):

        if self.is_function():
            self.log("Function")
        else:
            self.log("Non-function")

        smoothed = self.smooth_points()
        simplified = self.douglas_peucker(smoothed, epsilon=0.1)

        self.log(f"Original: {len(self.math_points)} points")
        self.log(f"Smoothed: {len(smoothed)} points")
        self.log(f"Simplified: {len(simplified)} points")

        for i in range(1, len(smoothed)):
            x1, y1 = self.math_to_canvas(*smoothed[i - 1])
            x2, y2 = self.math_to_canvas(*smoothed[i])

            self.canvas.create_line(
                x1, y1,
                x2, y2,
                fill="red",
                width=3,
                smooth=True
            )
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.pixel_points.clear()
        self.math_points.clear()
        self.log_console.configure(state="normal")
        self.log_console.delete(1.0, tk.END)
        self.log_console.configure(state="disabled")
        print("Canvas cleared")
        self.log("Canvas cleared.")
        self.last_filtered_point = None
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = SomsedApp(root)
    root.mainloop()
