import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser
import pandas as pd
import helper_functions as helpers  # Import our helper functions

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
        self.df = None  # This will store our loaded DataFrame

        # Variables for dropdown selections
        self.person_id_var = tk.StringVar()
        self.image_flag_var = tk.StringVar()

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
                  command=lambda: webbrowser.open("https://www.linkedin.com/in/khadizaprofile")).pack(anchor=tk.W, pady=2)

        # "Our App" Section
        app_frame = tk.LabelFrame(self.about_tab, text="Our App", bg=BG_COLOR, font=("Helvetica", 14, "bold"))
        app_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(app_frame,
                 text="This project uses depth imaging for anthropometric measurements. Our app analyzes data from a depth camera, visualizes 3D skeletal data, and estimates biometric parameters.",
                 bg=BG_COLOR, wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=5)
        tk.Button(app_frame, text="View GitHub Code", bg=BUTTON_BG, fg="white",
                  command=lambda: webbrowser.open("https://github.com/yourgithubrepo")).pack(anchor=tk.W, padx=10, pady=5)

    def create_main_tab(self):
        # Top frame for selecting person ID and image flag
        selection_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        selection_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(selection_frame, text="Select Person ID:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.id_combobox = ttk.Combobox(selection_frame, textvariable=self.person_id_var, state="readonly", width=15)
        self.id_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.id_combobox.bind("<<ComboboxSelected>>", self.update_image_flags)

        tk.Label(selection_frame, text="Select ImageFlag:", bg=BG_COLOR, font=("Helvetica", 12)).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.image_flag_combobox = ttk.Combobox(selection_frame, textvariable=self.image_flag_var, state="readonly", width=15)
        self.image_flag_combobox.grid(row=0, column=3, padx=5, pady=5)

        # Left frame for operation buttons
        options_frame = tk.Frame(self.main_tab, bg=BG_COLOR)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        tk.Label(options_frame, text="Data Visualization Options", bg=BG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=10)

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
                self.df["ImageFlag"] = self.df["ImageFlag"].ffill()
                # Cast ID and ImageFlag columns to string to ensure dropdown matching
                self.df["ID"] = self.df["ID"].astype(str)
                self.df["ImageFlag"] = self.df["ImageFlag"].astype(str)

                messagebox.showinfo("File Loaded", "CSV file loaded successfully.")
                self.preview_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                self.dashboard_text.insert(tk.END, f"Loaded CSV: {file_path}\n")
                # Populate person ID dropdown based on loaded data
                unique_ids = sorted(self.df["ID"].unique().tolist())
                self.id_combobox["values"] = unique_ids
                if unique_ids:
                    self.person_id_var.set(unique_ids[0])
                    self.update_image_flags()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
    
    def update_image_flags(self, event=None):
        """Update the image flag dropdown based on the selected person ID."""
        if self.df is None:
            return
        selected_id = self.person_id_var.get()
        subset = self.df[self.df["ID"] == selected_id]
        unique_flags = sorted(subset["ImageFlag"].unique().tolist())
        self.image_flag_combobox["values"] = unique_flags
        if unique_flags:
            self.image_flag_var.set(unique_flags[0])
    def get_selected_frame(self):
        """
        Filter the loaded DataFrame based on the selected person ID and image flag,
        and return the 20 rows corresponding to the image frame.
        """
        person_id = self.person_id_var.get()
        image_flag = self.image_flag_var.get()
        if self.df is None or not person_id or not image_flag:
            raise ValueError("Both Person ID and ImageFlag must be selected.")
        # Use helper function to filter data (this function should be in your helper module)
        # It will extract rows matching both the person and image, and ensure there are 20 rows.
        queried_frame = helpers.get_image_frame_by_id(self.df, person_id, image_flag)
        print(queried_frame)
        return queried_frame

    def visualize_skeleton_op(self):
        try:
            frame = self.get_selected_frame()
            print(frame)
            # Extract only coordinate columns (skip the first two metadata columns)
            coord_data = frame.iloc[:, 2:]
            print(coord_data)
            helpers.visualize_skeleton(coord_data, title="3D Skeleton Plot with Depth Visualization", section=1)
            self.dashboard_text.insert(tk.END, "Skeleton visualization displayed.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_height_op(self):
        try:
            frame = self.get_selected_frame()
            parts = helpers.assign_skeletal_parts(frame)
            height_m = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'])
            height_in = helpers.calculate_height(parts['head'], parts['left_ankle'], parts['right_ankle'], convert_to_inches=True)
            result = f"Calculated Height for Person {self.person_id_var.get()}, Image {self.image_flag_var.get()}:\n{height_m:.2f} m\n{height_in:.2f} in\n"
            messagebox.showinfo("Height Calculation", result)
            self.dashboard_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_girth_op(self):
        try:
            frame = self.get_selected_frame()
            parts = helpers.assign_skeletal_parts(frame)
            girth = helpers.calculate_girth(parts['left_shoulder'], parts['right_shoulder'])
            result = f"Calculated Girth (above waist) for Person {self.person_id_var.get()}, Image {self.image_flag_var.get()}: {girth:.2f} m\n"
            messagebox.showinfo("Girth Calculation", result)
            self.dashboard_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def estimate_weight_op(self):
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

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python app_iteration2.py <csv_file>")
        return

    csv_file = sys.argv[1]
    print(f"Received CSV file path: {csv_file}")
    # TODO: call your function to load or process the CSV here


if __name__ == '__main__':
    main()
