"""
helpers.py

This module contains helper functions for the ENCM509 Lab Project on Depth Imaging for 
Anthropometric Measurements. It includes functionality for:
  1. Loading and processing skeletal data.
  2. Visualizing a 3D skeleton.
  3. Calculating height and girth from depth data.
  4. Interpolating weight from reference data and fitting regression models.
  5. Evaluating data consistency using RMSE.
  6. Generating synthetic data based on a regression model.
  7. Plotting regression results and comparing synthetic vs. real data.

Code Running Instructions:
  1. Ensure you have Python 3 installed.
  2. Install required packages via pip:
       pip install numpy pandas matplotlib scikit-learn scipy
  3. Save this file as "helpers.py".
  4. You can run this file directly for testing:
       python helpers.py
     or import the functions into your main script as needed.
     
Author: Md Rafiu Hossain, Khadiza Ahsan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import norm

# Data Loading and Skeletal Assignment

def load_skeletal_data(csv_file):
    """
    Load the skeletal coordinates from a CSV file into a pandas DataFrame.
    
    The CSV file should include columns for x, y, z coordinates and optionally other fields.
    The joint ordering is assumed to follow the MATLAB code's convention.
    
    Parameters:
        csv_file (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the skeletal data.
    """
    df = pd.read_csv(csv_file)
    return df

# Skeletal Assignment

def assign_skeletal_parts(frame_data):
    """
    Given data for one frame (e.g., 20 joints), assign key skeletal parts based on predetermined indices.
    
    Parameters:
        frame_data (pd.DataFrame or np.ndarray): Data for one frame.
    
    Returns:
        dict: Keys are 'head', 'right_shoulder', 'left_shoulder', 'right_ankle', 'left_ankle'
              with their (x, y, z) coordinates as numpy arrays.
    """
    # Convert frame_data to a numpy array of type float
    try:
        data = np.array(frame_data, dtype=float)
    except Exception as e:
        raise ValueError("Could not convert frame_data to a float array: " + str(e))
    
    if np.isnan(data).any():
        raise ValueError("The input data contains NaN values. Please clean your CSV file.")
    
    parts = {
        'head': data[3, 0:3],            # Head at index 3 (MATLAB index 4 -> Python index 3)
        'right_shoulder': data[4, 0:3],    # Right Shoulder at index 4
        'left_shoulder': data[8, 0:3],     # Left Shoulder at index 8
        'right_ankle': data[14, 0:3],      # Right Ankle at index 14
        'left_ankle': data[18, 0:3]        # Left Ankle at index 18
    }
    return parts

def visualize_skeleton(frame_data, connections=None):
    """
    Visualize a 3D skeleton based on joint data and their connections.
    
    Parameters:
        frame_data (pd.DataFrame or np.ndarray): Skeletal data for one frame.
            Each row is a joint with columns [x, y, z].
        connections (list of tuples): List of pairs of indices (0-indexed) defining connections
            between joints. If None, a default connection set is used.
    """
    # Force conversion to a numpy array of type float and check for NaNs.
    try:
        data = np.array(frame_data, dtype=float)
    except Exception as e:
        raise ValueError("Could not convert frame_data to a float array: " + str(e))
    
    if np.isnan(data).any():
        raise ValueError("The input data contains NaN values. Please clean your CSV file.")
    
    # Default connections if none provided (based on MATLAB example)
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3),         # Spine and head
            (2, 4), (4, 5), (5, 6), (6, 7),   # Right arm
            (2, 8), (8, 9), (9, 10), (10, 11),# Left arm
            (0, 12), (12, 13), (13, 14), (14, 15),  # Right leg
            (0, 16), (16, 17), (17, 18), (18, 19)   # Left leg
        ]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    # Create a scatter plot with color indicating depth (z-coordinate)
    sc = ax.scatter(x, y, z, c=z, cmap='jet', s=50)
    plt.colorbar(sc, label='Depth (Z)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Skeleton Visualization')
    
    # Connect joints based on the connections list
    for (i, j) in connections:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-', linewidth=2)
    
    plt.show()

# Anthropometric Calculations

def calculate_height(head, left_ankle, right_ankle, convert_to_inches=False):
    """
    Calculate the height as the Euclidean distance between the head and the midpoint 
    of the left and right ankles.
    
    Parameters:
        head (np.array): (x, y, z) coordinate of the head.
        left_ankle (np.array): (x, y, z) coordinate of the left ankle.
        right_ankle (np.array): (x, y, z) coordinate of the right ankle.
        convert_to_inches (bool): If True, convert the height from meters to inches.
    
    Returns:
        float: Calculated height.
    """
    # Calculate the ankle midpoint
    ankle_midpoint = (left_ankle + right_ankle) / 2.0
    # Euclidean distance from head to ankle midpoint
    height = np.linalg.norm(head - ankle_midpoint)
    
    if convert_to_inches:
        height *= 39.3701  # Conversion factor from meters to inches
    
    return height

def calculate_girth(left_shoulder, right_shoulder):
    """
    Calculate the girth above the waist as the Euclidean distance between the left and right shoulders.
    
    Parameters:
        left_shoulder (np.array): (x, y, z) coordinate of the left shoulder.
        right_shoulder (np.array): (x, y, z) coordinate of the right shoulder.
    
    Returns:
        float: Calculated girth.
    """
    return np.linalg.norm(left_shoulder - right_shoulder)

# Weight Estimation via Interpolation and Regression

def interpolate_weight(height, ref_heights, ref_weights):
    """
    Interpolate or extrapolate the weight for a given height using a reference table.
    
    Parameters:
        height (float or np.array): Height(s) for which weight is estimated.
        ref_heights (np.array): Array of reference heights.
        ref_weights (np.array): Array of corresponding reference weights.
    
    Returns:
        float or np.array: Interpolated weight value(s).
    """
    return np.interp(height, ref_heights, ref_weights)

def fit_weight_regression(X, y, regression_type='linear', degree=1):
    """
    Fit a regression model to predict weight from height and girth measurements.
    
    Parameters:
        X (np.array): Design matrix with columns representing features (e.g., height and girth).
        y (np.array): Target variable (weight).
        regression_type (str): 'linear' for linear regression or 'polynomial' for polynomial regression.
        degree (int): Degree for polynomial regression (ignored for linear regression).
    
    Returns:
        model: The trained regression model.
        coeffs (np.array): Regression coefficients (first element is intercept).
    """
    if regression_type == 'linear':
        model = LinearRegression().fit(X, y)
        intercept = model.intercept_
        coef = model.coef_
        coeffs = np.concatenate(([intercept], coef))
    elif regression_type == 'polynomial':
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        linear_reg = model.named_steps['linearregression']
        intercept = linear_reg.intercept_
        coef = linear_reg.coef_
        coeffs = np.concatenate(([intercept], coef))
    else:
        raise ValueError("Unsupported regression type. Choose 'linear' or 'polynomial'.")
    
    return model, coeffs

# Data Evaluation

def calculate_rmse(measurements):
    """
    Calculate the Root Mean Square Error (RMSE) of measurements relative to their mean.
    
    This can be used to assess consistency (e.g., height measurements for the same person).
    
    Parameters:
        measurements (np.array): Array of measurement values.
    
    Returns:
        float: RMSE value.
    """
    mean_val = np.mean(measurements)
    rmse = np.sqrt(np.mean((measurements - mean_val)**2))
    return rmse

# Synthetic Data Generation and Plotting

def synthesize_data(model, num_samples, height_range, girth_range, noise_std=1.0):
    """
    Generate synthetic data based on the fitted regression model.
    
    Parameters:
        model: Regression model with a predict() method.
        num_samples (int): Number of synthetic samples to generate.
        height_range (tuple): (min_height, max_height) for synthetic height values.
        girth_range (tuple): (min_girth, max_girth) for synthetic girth values.
        noise_std (float): Standard deviation of Gaussian noise to add.
    
    Returns:
        X_synthetic (np.array): Synthetic design matrix (height, girth).
        weight_synthetic (np.array): Predicted weight values with added noise.
    """
    synthetic_heights = np.random.uniform(height_range[0], height_range[1], num_samples)
    synthetic_girths = np.random.uniform(girth_range[0], girth_range[1], num_samples)
    X_synthetic = np.column_stack((synthetic_heights, synthetic_girths))
    weight_pred = model.predict(X_synthetic)
    noise = np.random.randn(num_samples) * noise_std
    weight_synthetic = weight_pred + noise
    return X_synthetic, weight_synthetic

def plot_regression_results(X, y, model, title='Regression Results'):
    """
    Plot original data points and the regression plane.
    
    Parameters:
        X (np.array): Design matrix with height and girth.
        y (np.array): Actual weight values.
        model: Regression model with a predict() method.
        title (str): Title for the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the original data as a scatter plot
    ax.scatter(X[:, 0], X[:, 1], y, c='b', label='Original Data', s=50)
    
    # Create a grid for the regression plane
    height_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    girth_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    H, G = np.meshgrid(height_grid, girth_grid)
    grid_points = np.column_stack((H.ravel(), G.ravel()))
    W = model.predict(grid_points).reshape(H.shape)
    
    # Plot the regression surface
    ax.plot_surface(H, G, W, color='r', alpha=0.5)
    ax.set_xlabel('Height')
    ax.set_ylabel('Girth')
    ax.set_zlabel('Weight')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_synthetic_vs_real(X_real, y_real, X_synth, y_synth):
    """
    Plot a 3D scatter plot comparing real and synthetic data.
    
    Parameters:
        X_real (np.array): Real design matrix with height and girth.
        y_real (np.array): Real weight values.
        X_synth (np.array): Synthetic design matrix.
        y_synth (np.array): Synthetic weight values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_real[:, 0], X_real[:, 1], y_real, c='b', label='Real Data', s=50)
    ax.scatter(X_synth[:, 0], X_synth[:, 1], y_synth, c='r', label='Synthetic Data', s=50)
    ax.set_xlabel('Height')
    ax.set_ylabel('Girth')
    ax.set_zlabel('Weight')
    ax.set_title('Real vs Synthetic Data')
    ax.legend()
    plt.show()

def plot_weight_distribution(weights):
    """
    Plot the histogram and overlay a normal distribution curve for weight data.
    
    Parameters:
        weights (np.array): Array of weight values.
    """
    plt.hist(weights, bins=20, density=True, alpha=0.6, color='g')
    mu, sigma = np.mean(weights), np.std(weights)
    x_vals = np.linspace(weights.min(), weights.max(), 100)
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma), 'k-', linewidth=2)
    plt.xlabel('Weight')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Weights')
    plt.show()

