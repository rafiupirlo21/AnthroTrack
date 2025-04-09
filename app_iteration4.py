#!/usr/bin/env python3
"""
app.py

A Windows app for the ENCM 509 Anthropometric Measurement Project with three tabs:
  1. About – Introduces Our Team and Our App.
  2. Main – Provides data visualization and analysis operations (1 to 7):
         - Visualize Skeleton
         - Calculate Height
         - Calculate Girth
         - Estimate Weight from Height & Girth using regression (multiple models)
         - Data Evaluation (RMSE)
         - Generate Synthetic Data
         - Export Data
  3. Dashboard – Displays outputs from operations

Users may choose whether to run operations on a single 20-row block (a specific image frame)
or on all available frames for a given Person ID.

All styling uses shades of pink.

Code Running Instructions:
  1. Install dependencies:
         pip install numpy pandas matplotlib scikit-learn scipy
  2. Place this file along with helper_functions.py in the same folder.
  3. Run:
         python app.py

Author: [Your Name]
Date: [Current Date]
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser
import pandas as pd
import numpy as np
import helper_functions as helpers  # Import our helper functions

# Define pink-themed colors
BG_COLOR = "#FFC0CB"  # Light pink background
TAB_BG = "#FFB6C1"  # Slightly darker pink for tabs
BUTTON_BG = "#FF69B4"  # Hot pink for buttons


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ENCM 509 Anthropometric Measurement App")
        self.geometry("900x750")
        self.configure(bg=BG_COLOR)
        self.df = None  # Loaded DataFrame

        # Variables for dropdown selections
        self.person_id_var = tk.StringVar()
        self.image_flag_var = tk.StringVar()
        self.frame_mode_var = tk.StringVar(value="Single Frame")  # New: frame mode option
        self.regression_model_var = tk.StringVar(value="Linear")  # For weight estimation

        self.create_widgets()

    def create_widgets(self):
        # Set up Notebook (tabs)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=TAB_BG, foreground="black", padding=[10, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", BUTTON_BG)],
                  foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(self, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.about_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.main_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.dashboard_tab = tk.Frame(self.notebook, bg=BG_COLOR)

        self.notebook.add(self.about_tab, text="About")
        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.dashboard_tab, text="Dashboard")

        # Populate tabs
        self.create_about_tab()
        self.create_main_tab()
        self.create_dashboard_tab()

    def create_about_tab(self):
        # "Our Team" Section
        team_frame = tk.LabelFrame(self.about_tab, text="Our Team", bg=BG_COLOR, font=("Helvetica", 14, "bold"))
        team_frame.pack(fill=tk.X, padx=20, pady=10)

        # Team member 1
        me_frame = tk.Frame(team_frame, bg=BG_COLOR)
        me_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(me_frame, text="Md Rafiu Hossain", bg=BG_COLOR, font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        tk.Label(me_frame, text="Short intro about Md Rafiu Hossain...", bg=BG_COLOR).pack(anchor=tk.W)
        tk.Button(me_frame, text="LinkedIn", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://www.linkedin.com/in/rafiuprofile")).pack(anchor=tk.W, pady=2)

        # Team member 2
        khadiza_frame = tk.Frame(team_frame, bg=BG_COLOR)
        khadiza_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(khadiza_frame, text="Khadiza Ahsan", bg=BG_COLOR, font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        tk.Label(khadiza_frame, text="Short intro about Khadiza Ahsan...", bg=BG_COLOR).pack(anchor=tk.W)
        tk.Button(khadiza_frame, text="LinkedIn", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://www.linkedin.com/in/khadizaprofile")).pack(anchor=tk.W,
                                                                                                      pady=2)

        # "Our App" Section
        app_frame = tk.LabelFrame(self.about_tab, text="Our App", bg=BG_COLOR, font=("Helvetica", 14, "bold"))
        app_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(app_frame,
                 text="This project uses depth imaging for anthropometric measurements. Our app analyzes data from a depth camera, visualizes 3D skeletal data, and estimates biometric parameters.",
                 bg=BG_COLOR, wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=5)
        tk.Button(app_frame, text="View GitHub Code", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://github.com/yourgithubrepo")).pack(anchor=tk.W, padx=10,
                                                                                             pady=5)

    def create_main_tab(self):
        # Top frame for selecting Person ID, ImageFlag, Frame Mode, and Regression Model
        selection_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        selection_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(selection_frame, text="Select Person ID:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=0, column=0,
                                                                                                      padx=5, pady=5,
                                                                                                      sticky=tk.W)
        self.id_combobox = ttk.Combobox(selection_frame, textvariable=self.person_id_var, state="readonly", width=15)
        self.id_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.id_combobox.bind("<<ComboboxSelected>>", self.update_image_flags)

        tk.Label(selection_frame, text="Select ImageFlag:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=0, column=2,
                                                                                                      padx=5, pady=5,
                                                                                                      sticky=tk.W)
        self.image_flag_combobox = ttk.Combobox(selection_frame, textvariable=self.image_flag_var, state="readonly",
                                                width=15)
        self.image_flag_combobox.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(selection_frame, text="Frame Mode:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=1, column=0, padx=5,
                                                                                                pady=5, sticky=tk.W)
        self.frame_mode_combobox = ttk.Combobox(selection_frame, textvariable=self.frame_mode_var, state="readonly",
                                                width=15)
        self.frame_mode_combobox["values"] = ["Single Frame", "All Frames"]
        self.frame_mode_combobox.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(selection_frame, text="Select Regression Model:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=1,
                                                                                                             column=2,
                                                                                                             padx=5,
                                                                                                             pady=5,
                                                                                                             sticky=tk.W)
        self.regression_model_combobox = ttk.Combobox(selection_frame, textvariable=self.regression_model_var,
                                                      state="readonly", width=15)
        self.regression_model_combobox["values"] = ["Linear", "Polynomial", "Ridge", "All"]
        self.regression_model_combobox.grid(row=1, column=3, padx=5, pady=5)

        # Left frame for operation buttons
        options_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        tk.Label(options_frame, text="Data Visualization & Analysis", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(
            pady=10)
        tk.Button(options_frame, text="1. Visualize Skeleton", width=25, bg=BUTTON_BG, fg="white",
                  command=self.visualize_skeleton_op).pack(pady=5)
        tk.Button(options_frame, text="2. Calculate Height", width=25, bg=BUTTON_BG, fg="white",
                  command=self.calculate_height_op).pack(pady=5)
        tk.Button(options_frame, text="3. Calculate Girth", width=25, bg=BUTTON_BG, fg="white",
                  command=self.calculate_girth_op).pack(pady=5)
        tk.Button(options_frame, text="4. Estimate Weight", width=25, bg=BUTTON_BG, fg="white",
                  command=self.estimate_weight_op).pack(pady=5)
        tk.Button(options_frame, text="5. Data Evaluation (RMSE)", width=25, bg=BUTTON_BG, fg="white",
                  command=self.evaluate_data_op).pack(pady=5)
        tk.Button(options_frame, text="6. Generate Synthetic Data", width=25, bg=BUTTON_BG, fg="white",
                  command=self.generate_synthetic_op).pack(pady=5)
        tk.Button(options_frame, text="7. Export Data", width=25, bg=BUTTON_BG, fg="white",
                  command=self.export_data_op).pack(pady=5)

        # Right frame for preview/export area
        preview_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(preview_frame, text="Preview/Export Area", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=10)
        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, bg="white", fg="black")
        self.preview_text.pack(fill=tk.BOTH, expand=True)

    def create_dashboard_tab(self):
        dashboard_frame = tk.Frame(self.dashboard_tab, bg=BG_COLOR)
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(dashboard_frame, text="Dashboard", bg=BG_COLOR, font=("Helvetica", 16, "bold")).pack(pady=10)
        self.dashboard_text = tk.Text(dashboard_frame, wrap=tk.WORD, bg="white", fg="black")
        self.dashboard_text.pack(fill=tk.BOTH, expand=True)
        tk.Button(dashboard_frame, text="Refresh Dashboard", bg=BUTTON_BG, fg="white",
                  command=self.refresh_dashboard).pack(pady=10)

    # ------------------ Helper Methods ---------------------
    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = helpers.load_skeletal_data(file_path)
                self.df["ImageFlag"] = self.df["ImageFlag"].ffill()  # Ensure forward fill
                messagebox.showinfo("File Loaded", "CSV file loaded successfully.")
                self.preview_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                self.dashboard_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                unique_ids = sorted(self.df["ID"].unique().tolist())
                self.id_combobox["values"] = unique_ids
                if unique_ids:
                    self.person_id_var.set(str(unique_ids[0]))
                    self.update_image_flags()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")

    def update_image_flags(self, event=None):
        if self.df is None:
            return
        selected_id = self.person_id_var.get()
        subset = self.df[self.df["ID"].astype(str).str.strip() == str(selected_id)]
        unique_flags = sorted(subset["ImageFlag"].astype(str).str.strip().unique().tolist())
        # Do not append "All" here; we will allow the user to choose "All" explicitly in the frame mode dropdown.
        self.image_flag_combobox["values"] = unique_flags
        if unique_flags:
            self.image_flag_var.set(unique_flags[0])

    def get_selected_frame(self):
        """
        Filter the loaded DataFrame based on the selected Person ID and ImageFlag,
        and return the 20 rows corresponding to the image frame.
        This function is used when the user chooses "Single Frame" mode.
        """
        person_id = self.person_id_var.get()
        image_flag = self.image_flag_var.get()
        if self.df is None or not person_id or not image_flag:
            raise ValueError("Both Person ID and ImageFlag must be selected.")
        queried_frame = helpers.get_image_frame_by_id(self.df, person_id, image_flag)
        return queried_frame

    def get_all_frames(self):
        """
        Retrieve all frames (blocks of 20 consecutive rows) for the selected Person ID.
        This ignores the ImageFlag value.
        """
        person_id = self.person_id_var.get()
        if self.df is None or not person_id:
            raise ValueError("Person ID must be selected.")
        return helpers.get_all_frames_by_id(self.df, person_id)

    # ------------------ Operation Functions ---------------------
    def visualize_skeleton_op(self):
        try:
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frame = self.get_selected_frame()
                coord_data = frame.iloc[:, 2:]
                helpers.visualize_skeleton(coord_data, title="3D Skeleton Plot with Depth Visualization", section=1)
            else:
                frames = helpers.get_all_frames_by_id(self.df, self.person_id_var.get())
                for i, frame in enumerate(frames):
                    coord_data = frame.iloc[:, 2:]
                    helpers.visualize_skeleton(coord_data, title="3D Skeleton Plot", section=i + 1)
                    # Optionally, pause between frames:
                    plt.pause(1)
            self.dashboard_text.insert(tk.END, "Skeleton visualization displayed.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_height_op(self):
        try:
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frame = self.get_selected_frame()
                parts = helpers.assign_skeletal_parts(frame)
                height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
                height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'],
                                                     convert_to_inches=True)
                result = f"Calculated Height for Person {self.person_id_var.get()}, Image {self.image_flag_var.get()}:\n{height_m:.2f} m\n{height_in:.2f} in\n"
            else:
                frames = self.get_all_frames()
                heights = []
                for frame in frames:
                    parts = helpers.assign_skeletal_parts(frame)
                    height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
                    heights.append(height_m)
                avg_height = np.mean(heights)
                result = f"Average Height for Person {self.person_id_var.get()} (across {len(heights)} frames):\n{avg_height:.2f} m\n"
            messagebox.showinfo("Height Calculation", result)
            self.dashboard_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_girth_op(self):
        try:
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frame = self.get_selected_frame()
                parts = helpers.assign_skeletal_parts(frame)
                girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
                result = f"Calculated Girth (above waist) for Person {self.person_id_var.get()}, Image {self.image_flag_var.get()}: {girth:.2f} m\n"
            else:
                frames = self.get_all_frames()
                girths = []
                for frame in frames:
                    parts = helpers.assign_skeletal_parts(frame)
                    girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
                    girths.append(girth)
                avg_girth = np.mean(girths)
                result = f"Average Girth for Person {self.person_id_var.get()} (across {len(girths)} frames): {avg_girth:.2f} m\n"
            messagebox.showinfo("Girth Calculation", result)
            self.dashboard_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def estimate_weight_op(self):
        try:
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frames = [self.get_selected_frame()]
            else:
                frames = self.get_all_frames()

            # For each frame, compute weight estimate (we’ll average the estimates)
            weight_estimates = []
            regression_results = ""
            for frame in frames:
                parts = helpers.assign_skeletal_parts(frame)
                height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'],
                                                     convert_to_inches=True)
                girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
                # Reference table for BMI class 22 (example values)
                ref_heights = np.array([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76])
                ref_weights = np.array(
                    [105, 109, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 162, 166, 171, 176, 180])
                interpolated_weight = helpers.interpolate_weight(height_in, ref_heights, ref_weights)

                # Prepare training data for regression: use a constant girth value (from the frame) for all ref heights.
                X_train = np.column_stack((ref_heights, np.full_like(ref_heights, girth)))
                y_train = ref_weights

                reg_model_choice = self.regression_model_var.get().lower()  # "linear", "polynomial", "ridge", or "all"
                regression_models = {}
                if reg_model_choice == "all":
                    for model_type in ["linear", "polynomial", "ridge"]:
                        degree = 2 if model_type == "polynomial" else 1
                        model, coeffs = helpers.fit_weight_regression(X_train, y_train, regression_type=model_type,
                                                                      degree=degree)
                        regression_models[model_type] = (model, coeffs)
                    # For each model type, display coefficients (for demonstration, we use the last frame's coefficients).
                    for m_type, (_, coeffs) in regression_models.items():
                        regression_results += f"{m_type.capitalize()} Coefficients: {coeffs}\n"
                    # For simplicity, take the weight from the linear model as the frame estimate
                    weight_est = helpers.fit_weight_regression(X_train, y_train, regression_type="linear")[0].predict(
                        np.array([[height_in, girth]]))[0]
                else:
                    degree = 2 if reg_model_choice == "polynomial" else 1
                    model, coeffs = helpers.fit_weight_regression(X_train, y_train, regression_type=reg_model_choice,
                                                                  degree=degree)
                    regression_results = f"{reg_model_choice.capitalize()} Coefficients: {coeffs}\n"
                    weight_est = model.predict(np.array([[height_in, girth]]))[0]

                # Combine interpolation and regression: For demonstration, average the two estimates.
                combined_estimate = (interpolated_weight + weight_est) / 2.0
                weight_estimates.append(combined_estimate)

            avg_weight = np.mean(weight_estimates)
            result = (f"Estimated Weight for Person {self.person_id_var.get()}\n"
                      f"Average Estimated Weight (across {len(weight_estimates)} frames): {avg_weight:.2f} lbs\n"
                      + regression_results)
            messagebox.showinfo("Weight Estimation", result)
            self.dashboard_text.insert(tk.END, result)

            # Plot regression results for the single frame case (if only one frame, or for the first frame)
            if frame_mode == "Single Frame":
                reg_model_choice = self.regression_model_var.get().lower()
                degree = 2 if reg_model_choice == "polynomial" else 1
                model, _ = helpers.fit_weight_regression(X_train, y_train, regression_type=reg_model_choice,
                                                         degree=degree)
                helpers.plot_regression_results(X_train, y_train, model,
                                                title=f"{reg_model_choice.capitalize()} Regression Results")
            else:
                # For "All Frames", overlay all regression surfaces from the last processed frame as an example.
                if reg_model_choice == "all":
                    # Create a 3D plot and overlay each model's regression surface.
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='b', label='Reference Data', s=50)
                    height_grid = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 20)
                    girth_grid = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 20)
                    H, G = np.meshgrid(height_grid, girth_grid)
                    grid_points = np.column_stack((H.ravel(), G.ravel()))
                    colors = {'linear': 'r', 'polynomial': 'g', 'ridge': 'm'}
                    for m_type, (model, _) in regression_models.items():
                        W = model.predict(grid_points).reshape(H.shape)
                        ax.plot_surface(H, G, W, color=colors[m_type], alpha=0.4)
                    ax.set_xlabel('Height')
                    ax.set_ylabel('Girth')
                    ax.set_zlabel('Weight')
                    ax.set_title("Regression Surfaces for All Models")
                    plt.show()
                else:
                    model, _ = helpers.fit_weight_regression(X_train, y_train, regression_type=reg_model_choice,
                                                             degree=degree)
                    helpers.plot_regression_results(X_train, y_train, model,
                                                    title=f"{reg_model_choice.capitalize()} Regression Results")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def evaluate_data_op(self):
        try:
            if self.df is None:
                raise ValueError("Please load a CSV file first.")
            selected_id = self.person_id_var.get()
            frames = helpers.get_all_frames_by_id(self.df, selected_id)
            if not frames:
                raise ValueError("No valid image frames found for this person.")
            heights = []
            for i, frame in enumerate(frames):
                try:
                    parts = helpers.assign_skeletal_parts(frame)
                    height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
                    heights.append(height_m)
                except Exception as e:
                    print(f"Skipping frame {i}: {e}")
            if not heights:
                raise ValueError("No valid frames found for evaluation.")
            rmse_value = helpers.calculate_rmse(np.array(heights))
            result = f"Data Evaluation (RMSE) for Person {selected_id}:\nRMSE of Heights: {rmse_value:.4f} m\n"
            messagebox.showinfo("Data Evaluation", result)
            self.dashboard_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_synthetic_op(self):
        try:
            frame = self.get_selected_frame()
            parts = helpers.assign_skeletal_parts(frame)
            girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
            ref_heights = np.array([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76])
            ref_weights = np.array(
                [105, 109, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 162, 166, 171, 176, 180])
            X_train = np.column_stack((ref_heights, np.full_like(ref_heights, girth)))
            y_train = ref_weights
            model, coeffs = helpers.fit_weight_regression(X_train, y_train, regression_type='linear')
            height_range = (ref_heights.min(), ref_heights.max())
            girth_range = (girth * 0.9, girth * 1.1)
            X_synth, y_synth = helpers.synthesize_data(model, num_samples=100, height_range=height_range,
                                                       girth_range=girth_range, noise_std=2.0)
            helpers.plot_synthetic_vs_real(X_train, y_train, X_synth, y_synth)
            helpers.plot_weight_distribution(y_synth)
            result = "Synthetic data generated and plots displayed.\n"
            messagebox.showinfo("Synthetic Data", result)
            self.dashboard_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_data_op(self):
        result = "Export data functionality not implemented yet.\n"
        messagebox.showinfo("Export Data", result)
        self.dashboard_text.insert(tk.END, result)

    def refresh_dashboard(self):
        self.dashboard_text.delete("1.0", tk.END)
        self.dashboard_text.insert(tk.END, "Dashboard refreshed with latest outputs.\n")

    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open CSV", command=self.load_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)


if __name__ == '__main__':
    app = MainApp()
    app.create_menu()
    app.mainloop()


