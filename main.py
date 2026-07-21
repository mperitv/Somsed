import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import torch
import platform
import time
import math
import numpy as np


class SomsedApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Somsed")
        self.root.geometry("1000x700")
        self.functions = {
            "F1": {
                "pixel_points": [],
                "math_points": [],
                "equation": "Not optimized"
            }
        }
        self.current_function = "F1"
        self.axis_range = 10
        self.min_distance = 3
        self.stabilization = 0.1
        self.function_counter = 1
        self.last_filtered_point = None
        self.init_hardware()
        self.benchmark_device()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)

        self.root.grid_columnconfigure(0, weight=0, minsize=400)
        self.root.grid_columnconfigure(1, weight=1)
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
        x = torch.randn(2000, 2000, device=self.device)
        w = torch.randn(2000, 2000, device=self.device)
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

        self.function_frame = ctk.CTkFrame(
            self.root,
            width=400
        )

        self.function_list = ctk.CTkScrollableFrame(
            self.function_frame,
            width=370,
            height=400
        )

        self.function_list.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=(120,5)
        )

        self.function_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(5,0),
            pady=5
        )
        self.function_frame.grid_propagate(False)

        self.canvas = tk.Canvas(
            self.root,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.grid(
            row=0,
            column=1,
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
        self.refresh_function_list()

    def refresh_function_list(self):
        self.function_frame.grid_columnconfigure(0, weight=1)
        self.function_frame.grid_columnconfigure(1, weight=1)

        for widget in self.function_list.winfo_children():
            widget.destroy()
        
        title = ctk.CTkLabel(
            self.function_frame,
            text="Functions",
            font=("Arial", 16, "bold")
        )
        title.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=(10,15),
            sticky="n"
        )

        add_button = ctk.CTkButton(
            self.function_frame,
            text="+ Add Function",
            command=self.add_function,
            height=40,
            fg_color="green",
            hover_color="darkgreen"
        )

        add_button.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(8,4),
            pady=(0, 10)
        )

        delete_button = ctk.CTkButton(
            self.function_frame,
            text="- Delete Function",
            command=self.delete_function,
            height=40,
            fg_color="firebrick",
            hover_color="darkred"
        )

        delete_button.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(4,8),
            pady=(0, 10)
        )

        row = 2
        for name in self.functions:
            equation = self.functions[name]["equation"]
            text = f"{name}: {equation}"
            if name == self.current_function:
                button = ctk.CTkButton(
                    self.function_list,
                    text=text,
                    command=lambda n=name: self.switch_function(n),
                    height=55,
                    width=260,
                    anchor="w",
                    fg_color="#00897B",
                    hover_color="#00695C"
                )
            else:
                button = ctk.CTkButton(
                    self.function_list,
                    text=text,
                    command=lambda n=name: self.switch_function(n),
                    height=55,
                    width=360,
                    anchor="w",
                    fg_color=("gray75", "gray25"),
                    hover_color=("gray65", "gray35")
            )
            button.pack(
                fill="x",
                padx=8,
                pady=5
            )
            row += 1



    def add_function(self):
        self.function_counter += 1
        name = f"F{self.function_counter}"
        self.functions[name] = {
            "pixel_points": [],
            "math_points": [],
            "equation": "Not optimized"
        }
        self.current_function = name
        self.refresh_function_list()
        self.redraw_canvas()
        self.log(f"{name} created.")

    def delete_function(self):
        if len(self.functions) == 1:
            messagebox.showwarning(
                "Warning",
                "At least one function must exist."
            )
            return
        
        function_names = list(self.functions.keys())
        current_index = function_names.index(
            self.current_function
        )
        
        del self.functions[self.current_function]
        remaining_functions = list(self.functions.keys())
        if current_index > 0:
            self.current_function = remaining_functions[current_index - 1]
        else:
            self.current_function = remaining_functions[0]
        self.refresh_function_list()
        self.redraw_canvas()
        self.log("Function deleted.")

    def switch_function(self, name):
        self.current_function = name
        self.refresh_function_list()
        self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()

        pixels = self.current_pixels()

        for i in range(1, len(pixels)):
            x1, y1 = pixels[i-1]
            x2, y2 = pixels[i]

            self.canvas.create_line(
                x1, y1,
                x2, y2,
                width=3,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )

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
    
    def current_pixels(self):
        return self.functions[self.current_function]["pixel_points"]
    
    def current_math(self):
        return self.functions[self.current_function]["math_points"]
    
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
        self.current_pixels().append((fx, fy))
        self.current_math().append((mx, my))
        self.log(f"Point drawn at: ({mx:.1f}, {my:.1f})")
    
    def draw(self, event):
        fx, fy = self.filter_point(event.x, event.y)
        mx, my = self.canvas_to_math(fx, fy)
        prev_x, prev_y = self.current_pixels()[-1]
        distance = ((fx - prev_x) ** 2 + (fy - prev_y) ** 2) ** 0.5
        if distance < self.min_distance:
            return
        self.current_pixels().append((fx, fy))
        self.current_math().append((mx, my))
        self.redraw_canvas()

    def smooth_points(self, window=21):
        if len(self.current_math()) < window:
            return self.current_math()
        
        smoothed = []

        for i in range(len(self.current_math())):
            start = max(0, i - window // 2)
            end = min(len(self.current_math()), i + window // 2 + 1)

            avg_x = sum(p[0] for p in self.current_math()[start:end]) / (end - start)
            avg_y = sum(p[1] for p in self.current_math()[start:end]) / (end - start)

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
    
    def calculate_angles(self, points):
        angles = []

        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]

            angle = math.atan2(
                y2 - y1,
                x2 - x1
            )

            angles.append(angle)

        return angles

    def math_to_canvas(self, x, y):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        x_scale = width / (2 * self.axis_range)
        y_scale = width / (2 * self.axis_range)

        canvas_x = width / 2 + x * x_scale
        canvas_y = height / 2 - y * y_scale

        return canvas_x, canvas_y
    
    def is_function(self, tolerance=0.1):

        if len(self.current_math()) < 2:
            return False

        bins = {}

        for x, y in self.current_math():
            key = round(x / tolerance)

            if key not in bins:
                bins[key] = y
            else:
                if abs(bins[key] - y) > tolerance:
                    return False
                
        return True
    
    def calculate_error(self, coefficients, x, y):
        predictions = np.polyval(
            coefficients,
            x
        )

        error = np.mean(
            (predictions - y) ** 2
        )

        return error

    def optimize_curve(self):

        if len(self.current_math()) < 3:
            self.log("Not enough points")
            return

        points = self.current_math()

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        try:
            degree = 3

            coefficients = np.polyfit(
                x,
                y,
                degree
            )

            loss = self.calculate_error(
                coefficients,
                x,
                y
            )

            self.log(f"Current Loss: {loss:.6f}")

            equation = "y = "

            for i, c in enumerate(coefficients):

                power = degree - i

                if abs(c) < 0.001:
                    continue

                if power == 0:
                    equation += f"{c:.3f}"

                elif power == 1:
                    equation += f"{c:.3f}x + "

                else:
                    equation += f"{c:.3f}x^{power} + "

            equation = equation.replace("+ -", "- ")

            self.functions[self.current_function]["equation"] = equation
            self.refresh_function_list()

            test_x = np.linspace(
                min(x),
                max(x),
                200
            )

            test_y = np.polyval(
                coefficients,
                test_x
            )

            for i in range(1, len(test_x)):

                x1, y1 = self.math_to_canvas(
                    test_x[i-1],
                    test_y[i-1]
                )

                x2, y2 = self.math_to_canvas(
                    test_x[i],
                    test_y[i]
                )

                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="red",
                    width=3
                )

        except Exception as e:
            self.log(f"Optimization error: {e}")
    
    def clear_canvas(self):
        self.current_pixels().clear()
        self.current_math().clear()
        self.last_filtered_point = None
        self.redraw_canvas()
        self.log(f"{self.current_function} cleared.")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = SomsedApp(root)
    root.mainloop()
