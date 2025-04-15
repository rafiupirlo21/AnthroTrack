#!/usr/bin/env python3
"""
app.py

A Windows app for the ENCM 509 Anthropometric Measurement Project with two tabs:
  1. About – Introduces Our Team and Our App.
  2. Main – Provides data visualization and analysis operations (1 to 7) including:
         - Visualize Skeleton
         - Calculate Height
         - Calculate Girth
         - Estimate Weight from Height & Girth using regression (multiple models)
         - Data Evaluation (RMSE)
         - Generate Synthetic Data
         - Export Data (all outputs and graphs into a PDF)

Users may choose whether to run operations on a single 20-row block (a specific image frame)
or on all available frames for a given Person ID.


Code Running Instructions:
  1. Install dependencies:
         pip install -r requirements.txt
  2. Place this file along with helper_functions.py in the same folder.
  3. Run:
         python AnthroTrackApp.py

Authors: Md Rafiu Hossain, Khadiza Ahsan
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import helper_functions as helpers  # Import our helper functions
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Define pink-themed colors
BG_COLOR = "#FFC0CB"      # Light pink background
TAB_BG = "#FFB6C1"        # Slightly darker pink for tabs
BUTTON_BG = "#FF69B4"     # Hot pink for buttons

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AnthroTrack")
        self.iconbitmap("app_icon.ico")
        self.geometry("900x750")
        self.configure(bg=BG_COLOR)
        self.df = None  # Loaded DataFrame
        self.plot_files = []

        # Variables for dropdown selections
        self.person_id_var = tk.StringVar()
        self.image_flag_var = tk.StringVar()
        self.frame_mode_var = tk.StringVar(value="Single Frame")  # "Single Frame" or "All Frames"
        self.regression_model_var = tk.StringVar(value="Linear")   # Regression model choice

        self.create_widgets()

    def create_widgets(self):
        # Set up Notebook with two tabs: About and Main (Dashboard is merged in Main)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=TAB_BG, foreground="black", padding=[10, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", BUTTON_BG)],
                  foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(self, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs: About and Main
        self.about_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.main_tab = tk.Frame(self.notebook, bg=BG_COLOR)

        self.notebook.add(self.about_tab, text="About")
        self.notebook.add(self.main_tab, text="Main")

        # Populate tabs
        self.create_about_tab()
        self.create_main_tab()

    def create_about_tab(self):

        # "Our App" Section
        app_frame = tk.LabelFrame(self.about_tab, text="Our App", bg=BG_COLOR, font=("Helvetica", 16, "bold"))
        app_frame.pack(fill=tk.X, padx=30, pady=10)
        tk.Label(app_frame,
                 text="ENCM 509 Anthropometric Measurement App: This project uses depth imaging for anthropometric measurements. Our app analyzes data from a depth camera, visualizes 3D skeletal data, and estimates biometric parameters.",
                 bg=BG_COLOR, wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=5)
        tk.Button(app_frame, text="View GitHub Code", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://github.com/rafiupirlo21/AnthroTrack")).pack(anchor=tk.W, padx=10, pady=5)

        # "Our Team" Section
        team_frame = tk.LabelFrame(self.about_tab, text="Our Team", bg=BG_COLOR, font=("Helvetica", 16, "bold"))
        team_frame.pack(fill=tk.X, padx=20, pady=10)

        # Team member 1
        me_frame = tk.Frame(team_frame, bg=BG_COLOR)
        me_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(me_frame, text="Md Rafiu Hossain", bg=BG_COLOR, font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        tk.Label(me_frame, text="Electrical Engineering Student", bg=BG_COLOR).pack(anchor=tk.W)
        tk.Button(me_frame, text="LinkedIn", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://www.linkedin.com/in/rafiuhossain/")).pack(anchor=tk.W, pady=2)

        # Team member 2
        khadiza_frame = tk.Frame(team_frame, bg=BG_COLOR)
        khadiza_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(khadiza_frame, text="Khadiza Ahsan", bg=BG_COLOR, font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        tk.Label(khadiza_frame, text="Biomedical Engineering Student", bg=BG_COLOR).pack(anchor=tk.W)
        tk.Button(khadiza_frame, text="LinkedIn", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://www.linkedin.com/in/khadiza-ahsan/")).pack(anchor=tk.W, pady=2)

    def create_main_tab(self):
        # Top frame for selection controls
        selection_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        selection_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(selection_frame, text="Select Person ID:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.id_combobox = ttk.Combobox(selection_frame, textvariable=self.person_id_var, state="readonly", width=15)
        self.id_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.id_combobox.bind("<<ComboboxSelected>>", self.update_image_flags)

        tk.Label(selection_frame, text="Select ImageFlag:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.image_flag_combobox = ttk.Combobox(selection_frame, textvariable=self.image_flag_var, state="readonly", width=15)
        self.image_flag_combobox.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(selection_frame, text="Frame Mode:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.frame_mode_combobox = ttk.Combobox(selection_frame, textvariable=self.frame_mode_var, state="readonly", width=15)
        self.frame_mode_combobox["values"] = ["Single Frame", "All Frames"]
        self.frame_mode_combobox.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(selection_frame, text="Select Regression Model:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.regression_model_combobox = ttk.Combobox(selection_frame, textvariable=self.regression_model_var, state="readonly", width=15)
        self.regression_model_combobox["values"] = ["Linear", "Polynomial", "Bayesian", "All"]
        self.regression_model_combobox.grid(row=1, column=3, padx=5, pady=5)

        # Left frame for operation buttons
        options_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        tk.Label(options_frame, text="Data Visualization & Analysis", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=10)
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

        # Right frame renamed as Dashboard: displays all outputs
        dashboard_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        dashboard_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(dashboard_frame, text="Dashboard", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=10)
        self.preview_text = tk.Text(dashboard_frame, wrap=tk.WORD, bg="white", fg="black")
        self.preview_text.pack(fill=tk.BOTH, expand=True)

    # ------------------ Helper Methods ---------------------
    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = helpers.load_skeletal_data(file_path)
                self.df["ImageFlag"] = self.df["ImageFlag"].ffill()  # Ensure forward fill
                self.preview_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
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
                    plt.pause(1)
            self.preview_text.insert(tk.END, "Skeleton visualization displayed.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_height_op(self):
        try:
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frame = self.get_selected_frame()
                parts = helpers.assign_skeletal_parts(frame)
                height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
                height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'], convert_to_inches=True)
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
            self.preview_text.insert(tk.END, result)
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
            self.preview_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def estimate_weight_op(self):
        try:
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frame = self.get_selected_frame()
            else:
                frames = self.get_all_frames()
                frame = frames[0]

            parts = helpers.assign_skeletal_parts(frame)
            height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'],
                                                 convert_to_inches=True)
            girth = float(helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder']))

            # Reference data for BMI class 22.
            ref_heights = np.array([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76])
            ref_weights = np.array(
                [105, 109, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 162, 166, 171, 176, 180])
            interpolated_weight = helpers.interpolate_weight(height_in, ref_heights, ref_weights)

            # Collect girth variation from all frames.
            frames_all = helpers.get_all_frames_by_id(self.df, self.person_id_var.get())
            girths = [helpers.calculate_girth(helpers.assign_skeletal_parts(f)['left_shoulder'],
                                              helpers.assign_skeletal_parts(f)['right_shoulder'])
                      for f in frames_all]
            mean_girth = np.mean(girths) if girths else girth
            std_girth = np.std(girths) if girths else 0.01 * girth
            std_girth = std_girth or 0.01 * mean_girth

            # Build training data: simulate variation in girth.
            girth_variation = np.random.uniform(mean_girth - std_girth, mean_girth + std_girth, size=ref_heights.shape)
            X_train = np.column_stack((ref_heights, girth_variation))
            y_train = ref_weights

            reg_model_choice = self.regression_model_var.get().lower()  # e.g., 'linear', 'polynomial', 'bayesian', or 'all'
            model_types = ["linear", "polynomial", "bayesian"] if reg_model_choice == "all" else [reg_model_choice]
            regression_models = {
                m: helpers.fit_weight_regression(X_train, y_train, regression_type=m,
                                                 degree=(2 if m == "polynomial" else 1))
                for m in model_types
            }

            # Build results table.
            if reg_model_choice == "all":
                regression_results = "Regression Coefficients Table:\n"
                regression_results += f"{'Model':<12}{'Intercept':>12}{'Height Coef':>16}{'Girth Coef':>16}\n"
                regression_results += "-" * 56 + "\n"
                for m, (_, coeffs) in regression_models.items():
                    intercept = coeffs[0]
                    height_coef = coeffs[1] if len(coeffs) > 1 else 0.0
                    girth_coef = coeffs[2] if len(coeffs) > 2 else 0.0
                    regression_results += f"{m.capitalize():<12}{intercept:12.4f}{height_coef:16.4f}{girth_coef:16.4f}\n"
            else:
                coeffs = regression_models[reg_model_choice][1]
                regression_results = f"{reg_model_choice.capitalize()} Coefficients: {coeffs}\n"

            result = (f"Estimated Weight for Person {self.person_id_var.get()}, Image {self.image_flag_var.get()}:\n"
                      f"Interpolated Weight: {interpolated_weight:.2f} lbs\n" + regression_results)
            self.preview_text.insert(tk.END, result)

            # Plot regression results with dynamic title based on frame mode.
            if reg_model_choice == "all":
                filename = helpers.plot_regression_results(
                    X_train, y_train, model=None,
                    multiple_models=regression_models,
                    title="Regression Surfaces",
                    person_id=self.person_id_var.get(),
                    frame_mode=self.frame_mode_var.get(),
                    image_flag=self.image_flag_var.get()
                )
            else:
                model_to_plot = list(regression_models.values())[0][0]
                filename = helpers.plot_regression_results(
                    X_train, y_train, model=model_to_plot,
                    title=f"{reg_model_choice.capitalize()} Regression Results",
                    model_label=reg_model_choice,
                    person_id=self.person_id_var.get(),
                    frame_mode=self.frame_mode_var.get(),
                    image_flag=self.image_flag_var.get()
                )
            if filename:
                self.plot_files.append(filename)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def evaluate_data_op(self):
        try:
            if self.df is None:
                raise ValueError("Please load a CSV file first.")
            selected_id = self.person_id_var.get()
            frame_mode = self.frame_mode_var.get()
            if frame_mode == "Single Frame":
                frame = self.get_selected_frame()
                parts = helpers.assign_skeletal_parts(frame)
                height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
                result = (f"Single Frame Height for Person {selected_id} :\n"
                          f"Height: {height_m:.2f} m\n"
                          "Note: RMSE cannot be computed on a single frame.")
            else:
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
                    raise ValueError("No valid image frames found for evaluation.")
                rmse_value = helpers.calculate_rmse(np.array(heights))
                result = (f"Data Evaluation (RMSE) for Person {selected_id} across all frames:\n"
                          f"RMSE of Heights: {rmse_value:.4f} m\n"
                          f"Frame Heights: {heights}\n")
            self.preview_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_synthetic_op(self):
        try:
            # Use a representative frame for measurements.
            frame = self.get_selected_frame()  # for single frame; adjust if needed for all frames
            parts = helpers.assign_skeletal_parts(frame)
            girth = float(helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder']))
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
            # Use Person ID and Frame Mode to build a flag string.
            person_id = self.person_id_var.get()
            # For single frame, use the selected image flag; for all frames, use "All".
            image_flag = self.image_flag_var.get() if self.frame_mode_var.get() == "Single Frame" else "All"
            # Save synthetic plot and distribution plot with dynamic names.
            synth_filename = helpers.plot_synthetic_vs_real(X_train, y_train, X_synth, y_synth, person_id, image_flag)
            dist_filename = helpers.plot_weight_distribution(y_synth, person_id, image_flag)
            # Save these filenames for later export.
            self.plot_files.append(synth_filename)
            self.plot_files.append(dist_filename)

            result = "Synthetic data generated and plots saved.\n"
            messagebox.showinfo("Synthetic Data", result)
            self.preview_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_data_op(self):
        try:
            pdf_file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if not pdf_file:
                return
            c = canvas.Canvas(pdf_file, pagesize=letter)
            page_width, page_height = letter

            # Write dashboard text from the preview_text widget onto the first PDF page
            dashboard_text = self.preview_text.get("1.0", tk.END)
            c.setFont("Helvetica", 10)
            text_obj = c.beginText(40, page_height - 40)
            for line in dashboard_text.splitlines():
                text_obj.textLine(line)
            c.drawText(text_obj)
            c.showPage()

            # List of plot image files to include (ensure your plotting functions save images to these files)
            plot_files = ["regression_plot.png", "synthetic_plot.png", "distribution_plot.png"]
            for pf in plot_files:
                try:
                    import os
                    if os.path.exists(pf):
                        img = ImageReader(pf)
                        c.drawImage(img, 40, 200, width=page_width - 80, preserveAspectRatio=True, mask='auto')
                        c.showPage()
                except Exception as img_e:
                    print(f"Could not add image {pf}: {img_e}")

            c.save()
            export_result = f"Data exported to {pdf_file}\n"
            self.preview_text.insert(tk.END, export_result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def refresh_dashboard(self):
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, "Dashboard refreshed with latest outputs.\n")

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
