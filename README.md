# AnthroTrack 🧠📏  
** Anthropometric Measurement for Rural Bangladesh**  
University of Calgary – ENCM 509 Final Project

## 📌 Project Overview

AnthroTrack is a lightweight desktop app that uses depth imaging and AI to estimate **height**, **girth**, and **weight** from 3D skeletal joint data. It is built to serve rural or low-resource environments where accurate anthropometric assessments are often inaccessible. Using Intel RealSense-like depth data, the app reconstructs a person’s skeleton, performs biometric calculations, and evaluates data consistency using RMSE and regression modeling. 

This project helps standardize contactless biometric screening, and empowers frontline healthcare workers with real-time, portable, and scalable tools for public health assessments.

---

## ⚙️ How to Run It
1. Clone the repository or download the source files
2. Open a terminal in the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Make sure both AnthroTrackApp.py and helper_functions.py are in the same folder
5. python AnthroTrackApp.py
