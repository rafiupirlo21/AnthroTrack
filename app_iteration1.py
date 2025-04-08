#!/usr/bin/env python3
"""
app.py

A Windows app for the ENCM 509 Anthropometric Measurement Project.
This app has three tabs:
  1. About – Introduces our team (Md Rafiu Hossain, Khadiza Ahsan) with short intros and LinkedIn links,
             and our app with a GitHub link.
  2. Main – Offers data visualization options (operations 1 to 6), a button to load the dataset, and an export option.
  3. Dashboard – Displays all outputs from operations 1 to 6 on one page.

The app uses shades of pink for a unified look.

Code Running Instructions:
  1. Ensure Python 3 is installed.
  2. Install required packages:
         pip install numpy pandas matplotlib scikit-learn scipy
  3. Save this file as "app.py" and ensure "helpers.py" is in the same folder.
  4. Run the app with:
         python app.py

Author: Md Rafiu Hossain, Khadiza Ahsan
Date: [Current Date]
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser
import helper_functions as helpers  # Import our helper functions

# Define pink-themed colors
BG_COLOR = "#FFC0CB"  # Light pink background
TAB_BG = "#FFB6C1"  # Slightly darker pink for tabs
BUTTON_BG = "#FF69B4"  # Hot pink for buttons


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ENCM 509 Anthropometric Measurement App")
        self.geometry("900x700")
        self.configure(bg=BG_COLOR)
        self.df = None  # This will store our loaded DataFrame

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
        # Left frame for visualization options and loading dataset
        options_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        tk.Label(options_frame, text="Data Visualization Options", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(
            pady=10)

        # Add a button to load the dataset
        tk.Button(options_frame, text="Load Dataset", width=25, bg=BUTTON_BG, fg="white", command=self.load_csv).pack(
            pady=5)

        tk.Button(options_frame, text="1. Visualize Skeleton", width=25, bg=BUTTON_BG, fg="white",
                  command=self.visualize_skeleton_op).pack(pady=5)
        tk.Button(options_frame, text="2. Calculate Height", width=25, bg=BUTTON_BG, fg="white",
                  command=self.calculate_height_op).pack(pady=5)
        tk.Button(options_frame, text="3. Calculate Girth", width=25, bg=BUTTON_BG, fg="white",
                  command=self.calculate_girth_op).pack(pady=5)
        tk.Button(options_frame, text="4. Estimate Weight", width=25, bg=BUTTON_BG, fg="white",
                  command=self.estimate_weight_op).pack(pady=5)
        tk.Button(options_frame, text="5. Generate Synthetic Data", width=25, bg=BUTTON_BG, fg="white",
                  command=self.generate_synthetic_op).pack(pady=5)
        tk.Button(options_frame, text="6. Export Data", width=25, bg=BUTTON_BG, fg="white",
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

    # ------------------ Operation Functions ---------------------
    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Use helper function to load data
                self.df = helpers.load_skeletal_data(file_path)
                messagebox.showinfo("File Loaded", "CSV file loaded successfully.")
                self.preview_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                self.dashboard_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")

    def visualize_skeleton_op(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return
        # For demonstration, select the first 20 rows corresponding to one image.
        frame = self.df.iloc[0:20]
        # Pass only coordinate columns (skip metadata)
        coord_data = frame.iloc[:, 2:]
        helpers.visualize_skeleton(coord_data, title="3D Skeleton Plot with Depth Visualization", section=1)
        self.dashboard_text.insert(tk.END, "Skeleton visualization displayed.\n")

    def calculate_height_op(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return
        # Use the first 20 rows for one image
        frame = self.df.iloc[0:20]
        coord_data = frame.iloc[:, 2:]
        parts = helpers.assign_skeletal_parts(coord_data)
        height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
        height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'],
                                             convert_to_inches=True)
        result = f"Calculated Height:\n{height_m:.2f} m\n{height_in:.2f} in\n"
        messagebox.showinfo("Height Calculation", result)
        self.dashboard_text.insert(tk.END, result)

    def calculate_girth_op(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return
        frame = self.df.iloc[0:20]
        coord_data = frame.iloc[:, 2:]
        parts = helpers.assign_skeletal_parts(coord_data)
        girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
        result = f"Calculated Girth (above waist): {girth:.2f} m\n"
        messagebox.showinfo("Girth Calculation", result)
        self.dashboard_text.insert(tk.END, result)

    def estimate_weight_op(self):
        # Placeholder: Implement weight estimation using interpolation/regression from helpers.
        result = "Weight estimation functionality not implemented yet.\n"
        messagebox.showinfo("Weight Estimation", result)
        self.dashboard_text.insert(tk.END, result)

    def generate_synthetic_op(self):
        # Placeholder: Implement synthetic data generation using helpers.synthesize_data.
        result = "Synthetic data generation functionality not implemented yet.\n"
        messagebox.showinfo("Synthetic Data", result)
        self.dashboard_text.insert(tk.END, result)

    def export_data_op(self):
        # Placeholder: Implement data export functionality.
        result = "Export data functionality not implemented yet.\n"
        messagebox.showinfo("Export Data", result)
        self.dashboard_text.insert(tk.END, result)

    def refresh_dashboard(self):
        self.dashboard_text.delete("1.0", tk.END)
        self.dashboard_text.insert(tk.END, "Dashboard refreshed with latest outputs.\n")

    # ------------------ Menu Functions ---------------------
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




