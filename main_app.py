"""
main_app.py

This application provides a GUI for the ENCM509 Depth Imaging Project.
Users can load a skeletal coordinates CSV file and choose from several operations:
  1. Visualize 3D Skeleton
  2. Calculate Height (meters and inches)
  3. Calculate Girth (above the waist)
  4. Estimate Weight via Regression (with interpolation)
  5. Data Evaluation (RMSE calculation)
  6. Generate Synthetic Data and Display Plots

Running Instructions:
  1. Ensure Python 3 is installed.
  2. Install required packages:
       pip install numpy pandas matplotlib scikit-learn scipy
  3. Place this file (gui_app.py) together with the "helpers.py" module.
  4. Run the application:
       python gui_app.py

Author: Md Rafiu Hossain, Khadiza Ahsan
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import helper_functions as helpers

class ProjectGUI:
    def __init__(self, master):
        self.master = master
        master.title("ENCM509 Depth Imaging Project GUI")
        
        # Initialize dataset container
        self.df = None

        # Label to display loaded file
        self.label_file = tk.Label(master, text="No dataset loaded", fg="blue")
        self.label_file.pack(pady=5)

        # Button to load dataset
        self.btn_load = tk.Button(master, text="Load Dataset", command=self.load_dataset)
        self.btn_load.pack(pady=5)
        
        # Instructions for selecting an operation
        self.lbl_instruction = tk.Label(master, text="Select an operation (1-6):")
        self.lbl_instruction.pack(pady=5)
        
        # Variable to store selected option
        self.selected_option = tk.IntVar(value=1)
        
        # Radio buttons for six operations
        options = [
            ("1. Visualize 3D Skeleton", 1),
            ("2. Calculate Height", 2),
            ("3. Calculate Girth", 3),
            ("4. Estimate Weight", 4),
            ("5. Data Evaluation (RMSE)", 5),
            ("6. Synthetic Data Generation", 6)
        ]
        
        for text, value in options:
            tk.Radiobutton(master, text=text, variable=self.selected_option, value=value).pack(anchor=tk.W)
        
        # Button to run the selected operation
        self.btn_run = tk.Button(master, text="Run Operation", command=self.run_operation)
        self.btn_run.pack(pady=10)
        
        # Quit button
        self.btn_quit = tk.Button(master, text="Quit", command=master.quit)
        self.btn_quit.pack(pady=5)
    
    def load_dataset(self):
        """Open a file dialog to load the CSV dataset."""
        file_path = filedialog.askopenfilename(
            title="Select Skeletal Coordinates CSV",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if file_path:
            try:
                self.df = helpers.load_skeletal_data(file_path)
                self.label_file.config(text=f"Loaded: {file_path}")
                messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
    
    def run_operation(self):
        """Run the selected operation using helper functions."""
        option = self.selected_option.get()
        
        # Check if dataset is loaded for operations that require it
        if self.df is None and option in [1, 2, 3]:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        if option == 1:
            # Visualize 3D Skeleton: use the first 20 rows (one frame)
            frame = self.df.iloc[0:20]
            helpers.visualize_skeleton(frame)
        
        elif option == 2:
            # Calculate Height from Depth Data
            frame = self.df.iloc[0:20]
            parts = helpers.assign_skeletal_parts(frame)
            height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
            height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'], convert_to_inches=True)
            messagebox.showinfo("Height Calculation", f"Calculated Height:\n{height_m:.2f} m\n{height_in:.2f} in")
        
        elif option == 3:
            # Calculate Girth from Depth Data
            frame = self.df.iloc[0:20]
            parts = helpers.assign_skeletal_parts(frame)
            girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
            messagebox.showinfo("Girth Calculation", f"Calculated Girth (above waist): {girth:.2f} m")
        
        elif option == 4:
            # Estimate Weight from Girth and Height using Regression
            # For demonstration, we use synthetic sample data.
            ref_heights = np.linspace(58, 76, 19)  # Reference heights (inches)
            ref_weights = np.array([105, 109, 112, 116, 120, 124, 128, 132, 136, 140,
                                    144, 149, 153, 157, 162, 166, 171, 176, 180])
            # Generate sample data
            Height_samples = np.random.uniform(60, 70, 30)  # in inches
            Girth_samples = np.random.uniform(0.3, 0.4, 30)
            y_samples = np.interp(Height_samples, ref_heights, ref_weights)
            X = np.column_stack((Height_samples, Girth_samples))
            model, coeffs = helpers.fit_weight_regression(X, y_samples, regression_type='linear')
            messagebox.showinfo("Weight Estimation", 
                                f"Regression Coefficients:\nIntercept: {coeffs[0]:.2f}\n"
                                f"Slope (Height): {coeffs[1]:.2f}\nSlope (Girth): {coeffs[2]:.2f}")
            helpers.plot_regression_results(X, y_samples, model, title="Weight Regression (Linear)")
        
        elif option == 5:
            # Data Evaluation: Calculate RMSE (simulate with sample height measurements)
            # For demonstration, we use synthetic height measurements.
            Height_samples = np.random.uniform(60, 70, 30)
            rmse_val = helpers.calculate_rmse(Height_samples)
            messagebox.showinfo("Data Evaluation (RMSE)", f"Calculated RMSE for height measurements: {rmse_val:.2f}")
        
        elif option == 6:
            # Synthetic Data Generation and Plotting
            # Use the sample regression model from option 4 as a demonstration.
            ref_heights = np.linspace(58, 76, 19)
            ref_weights = np.array([105, 109, 112, 116, 120, 124, 128, 132, 136, 140,
                                    144, 149, 153, 157, 162, 166, 171, 176, 180])
            Height_samples = np.random.uniform(60, 70, 30)
            Girth_samples = np.random.uniform(0.3, 0.4, 30)
            y_samples = np.interp(Height_samples, ref_heights, ref_weights)
            X = np.column_stack((Height_samples, Girth_samples))
            model, coeffs = helpers.fit_weight_regression(X, y_samples, regression_type='linear')
            # Generate synthetic data based on the regression model.
            X_synth, y_synth = helpers.synthesize_data(model, num_samples=100,
                                                       height_range=(Height_samples.min(), Height_samples.max()),
                                                       girth_range=(Girth_samples.min(), Girth_samples.max()),
                                                       noise_std=2.0)
            helpers.plot_synthetic_vs_real(X, y_samples, X_synth, y_synth)
            helpers.plot_weight_distribution(y_synth)
        
        else:
            messagebox.showerror("Error", "Invalid operation selected!")

if __name__ == '__main__':
    root = tk.Tk()
    app = ProjectGUI(root)
    root.mainloop()
