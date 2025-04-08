#!/usr/bin/env python3
"""
app.py

A Windows app for the ENCM 509 Anthropometric Measurement Project.

This application allows you to:
  1. Load a CSV file containing skeletal data.
     The CSV file is expected to have:
       - Column 0: New Image Flag (e.g., "000-14", "001-14", etc.)
       - Column 1: Person ID (e.g., 1, 2, etc.)
       - Columns 2 onward: Coordinate data for each row (each row contains a set of x, y, z coordinates).
         Each image is represented by multiple rows (e.g., 20 rows) sharing the same Person ID and Image Flag.
  2. Group the data by Person ID and then by Image Flag.
  3. Populate a drop-down menu for Person IDs and, based on that, a drop-down menu for the corresponding Image Flags.
  4. Execute operations (Visualize Skeleton, Calculate Height, Calculate Girth, etc.) on the selected image.

The UI uses shades of pink.

Code Running Instructions:
  1. Ensure you have Python 3 installed.
  2. Install required packages: pandas, matplotlib (Tkinter is included on Windows).
  3. Save this file as app.py and ensure helper_functions.py is in the same folder.
  4. Run: python app.py

Author: [Your Name]
Date: [Current Date]
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
import pandas as pd
import helper_functions as helpers  # This module must contain:
                                   # - load_skeletal_data(csv_file)
                                   # - group_data_by_person_and_image(df)
                                   # - visualize_skeleton(coord_data, title, section)
                                   # - assign_skeletal_parts(coord_data)
                                   # - calculate_height(head, left_ankle, right_ankle, convert_to_inches=False)
                                   # - calculate_girth(left_shoulder, right_shoulder)
                                   
# Define pink-themed colors
BG_COLOR = "#FFC0CB"      # Light pink background
TAB_BG = "#FFB6C1"        # Slightly darker pink for tabs
BUTTON_BG = "#FF69B4"     # Hot pink for buttons

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ENCM 509 Anthropometric Measurement App")
        self.geometry("900x700")
        self.configure(bg=BG_COLOR)
        self.df = None              # Original DataFrame
        self.grouped_data = None      # Grouped data: dict[PersonID][ImageFlag] = coordinate data (numpy array)
        self.create_widgets()
    
    def create_widgets(self):
        # Create Notebook (tabs)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=TAB_BG, foreground="black", padding=[10, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", BUTTON_BG)],
                  foreground=[("selected", "white")])
        
        self.notebook = ttk.Notebook(self, style="TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create three tabs: About, Main, Dashboard
        self.about_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.main_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.dashboard_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        
        self.notebook.add(self.about_tab, text="About")
        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        
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
                  command=lambda: webbrowser.open("https://www.linkedin.com/in/khadizaprofile")).pack(anchor=tk.W, pady=2)
        
        # "Our App" Section
        app_frame = tk.LabelFrame(self.about_tab, text="Our App", bg=BG_COLOR, font=("Helvetica", 14, "bold"))
        app_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(app_frame, text="This project uses depth imaging for anthropometric measurements. Our app analyzes data from a depth camera, visualizes 3D skeletal data, and estimates biometric parameters.", 
                 bg=BG_COLOR, wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=5)
        tk.Button(app_frame, text="View GitHub Code", bg=BUTTON_BG, fg="white", 
                  command=lambda: webbrowser.open("https://github.com/yourgithubrepo")).pack(anchor=tk.W, padx=10, pady=5)
    
    def create_main_tab(self):
        # Create a top frame for controls (drop-down menus)
        top_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        # Button to load CSV file
        tk.Button(top_frame, text="Load CSV", bg=BUTTON_BG, fg="white", command=self.load_csv).grid(row=0, column=0, padx=5, pady=5)
        
        # Drop-down for Person ID
        tk.Label(top_frame, text="Select Person ID:", bg=BG_COLOR, font=("Helvetica", 12, "bold")).grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.person_combo = ttk.Combobox(top_frame, state="readonly", width=10)
        self.person_combo.grid(row=0, column=2, padx=5, pady=5)
        self.person_combo.bind("<<ComboboxSelected>>", self.update_image_flags)
        
        # Drop-down for Image Flag (for selected Person)
        tk.Label(top_frame, text="Select Image Flag:", bg=BG_COLOR, font=("Helvetica", 12, "bold")).grid(row=0, column=3, padx=5, pady=5, sticky="e")
        self.image_combo = ttk.Combobox(top_frame, state="readonly", width=15)
        self.image_combo.grid(row=0, column=4, padx=5, pady=5)
        
        # Create a frame for operation buttons (1-6)
        op_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        op_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        tk.Button(op_frame, text="Visualize Skeleton", width=25, bg=BUTTON_BG, fg="white", command=self.visualize_skeleton_op).pack(pady=5)
        tk.Button(op_frame, text="Calculate Height", width=25, bg=BUTTON_BG, fg="white", command=self.calculate_height_op).pack(pady=5)
        tk.Button(op_frame, text="Calculate Girth", width=25, bg=BUTTON_BG, fg="white", command=self.calculate_girth_op).pack(pady=5)
        tk.Button(op_frame, text="Estimate Weight", width=25, bg=BUTTON_BG, fg="white", command=self.estimate_weight_op).pack(pady=5)
        tk.Button(op_frame, text="Generate Synthetic Data", width=25, bg=BUTTON_BG, fg="white", command=self.generate_synthetic_op).pack(pady=5)
        tk.Button(op_frame, text="Export Data", width=25, bg=BUTTON_BG, fg="white", command=self.export_data_op).pack(pady=5)
        
        # Create a preview frame for output messages
        preview_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        preview_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=20, pady=10)
        tk.Label(preview_frame, text="Preview/Output Area", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=5)
        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, bg="white", fg="black")
        self.preview_text.pack(fill=tk.BOTH, expand=True)
    
    def create_dashboard_tab(self):
        dashboard_frame = tk.Frame(self.dashboard_tab, bg=BG_COLOR)
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(dashboard_frame, text="Dashboard", bg=BG_COLOR, font=("Helvetica", 16, "bold")).pack(pady=10)
        self.dashboard_text = tk.Text(dashboard_frame, wrap=tk.WORD, bg="white", fg="black")
        self.dashboard_text.pack(fill=tk.BOTH, expand=True)
        tk.Button(dashboard_frame, text="Refresh Dashboard", bg=BUTTON_BG, fg="white", command=self.refresh_dashboard).pack(pady=10)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                messagebox.showinfo("File Loaded", "CSV file loaded successfully.")
                self.preview_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                self.dashboard_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                # Group the data by Person ID and Image Flag using our helper function.
                self.grouped_data = helpers.group_data_by_person_and_image(self.df)
                # Populate Person ID drop-down with keys from grouped_data.
                person_ids = sorted(list(self.grouped_data.keys()))
                self.person_combo["values"] = person_ids
                if person_ids:
                    self.person_combo.current(0)
                    self.update_image_flags()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
    
    def update_image_flags(self, event=None):
        # Update the Image Flag drop-down based on the selected Person ID.
        selected_person = self.person_combo.get()
        if not selected_person:
            return
        try:
            person_id = int(selected_person)
        except ValueError:
            person_id = selected_person
        if self.grouped_data and person_id in self.grouped_data:
            image_flags = sorted(list(self.grouped_data[person_id].keys()))
            self.image_combo["values"] = image_flags
            if image_flags:
                self.image_combo.current(0)
        else:
            self.image_combo["values"] = []
            self.image_combo.set("")
    
    def visualize_skeleton_op(self):
        # Retrieve the coordinate data for the selected Person ID and Image Flag.
        selected_person = self.person_combo.get()
        selected_flag = self.image_combo.get()
        if not selected_person or not selected_flag:
            messagebox.showerror("Error", "Please select both a Person ID and an Image Flag.")
            return
        try:
            person_id = int(selected_person)
        except ValueError:
            person_id = selected_person
        try:
            coord_data = self.grouped_data[person_id][selected_flag]
        except KeyError:
            messagebox.showerror("Error", "Selected Image Flag not found for the chosen Person.")
            return
        # Visualize the skeleton (coordinate data is assumed to be a numpy array of shape (n, 3))
        helpers.visualize_skeleton(coord_data, title=f"Skeleton Visualization for Person {person_id} - {selected_flag}")
        self.dashboard_text.insert(tk.END, f"Visualized skeleton for Person {person_id}, Image Flag: {selected_flag}\n")
    
    def calculate_height_op(self):
        # Calculate height for the selected image.
        selected_person = self.person_combo.get()
        selected_flag = self.image_combo.get()
        if not selected_person or not selected_flag:
            messagebox.showerror("Error", "Please select both a Person ID and an Image Flag.")
            return
        try:
            person_id = int(selected_person)
        except ValueError:
            person_id = selected_person
        try:
            coord_data = self.grouped_data[person_id][selected_flag]
        except KeyError:
            messagebox.showerror("Error", "Selected Image Flag not found for the chosen Person.")
            return
        # Use the helper function assign_skeletal_parts to tag key joints.
        parts = helpers.assign_skeletal_parts(coord_data)
        height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
        height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'], convert_to_inches=True)
        result = f"Person {person_id}, Image Flag {selected_flag}:\nHeight = {height_m:.2f} m ({height_in:.2f} in)\n"
        messagebox.showinfo("Height Calculation", result)
        self.dashboard_text.insert(tk.END, result)
    
    def calculate_girth_op(self):
        # Calculate girth for the selected image.
        selected_person = self.person_combo.get()
        selected_flag = self.image_combo.get()
        if not selected_person or not selected_flag:
            messagebox.showerror("Error", "Please select both a Person ID and an Image Flag.")
            return
        try:
            person_id = int(selected_person)
        except ValueError:
            person_id = selected_person
        try:
            coord_data = self.grouped_data[person_id][selected_flag]
        except KeyError:
            messagebox.showerror("Error", "Selected Image Flag not found for the chosen Person.")
            return
        parts = helpers.assign_skeletal_parts(coord_data)
        girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
        result = f"Person {person_id}, Image Flag {selected_flag}:\nGirth (above waist) = {girth:.2f} m\n"
        messagebox.showinfo("Girth Calculation", result)
        self.dashboard_text.insert(tk.END, result)
    
    def estimate_weight_op(self):
        # Placeholder for weight estimation
        selected_person = self.person_combo.get()
        selected_flag = self.image_combo.get()
        result = f"Weight estimation not implemented for Person {selected_person}, Image Flag {selected_flag}.\n"
        messagebox.showinfo("Weight Estimation", result)
        self.dashboard_text.insert(tk.END, result)
    
    def generate_synthetic_op(self):
        # Placeholder for synthetic data generation
        selected_person = self.person_combo.get()
        selected_flag = self.image_combo.get()
        result = f"Synthetic data generation not implemented for Person {selected_person}, Image Flag {selected_flag}.\n"
        messagebox.showinfo("Synthetic Data", result)
        self.dashboard_text.insert(tk.END, result)
    
    def export_data_op(self):
        # Placeholder for data export
        selected_person = self.person_combo.get()
        selected_flag = self.image_combo.get()
        result = f"Data export not implemented for Person {selected_person}, Image Flag {selected_flag}.\n"
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
