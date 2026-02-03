# 🔥 Calories Burnt Prediction 


A Machine Learning project that predicts the number of calories burnt during exercise based on user's physical characteristics and workout intensity.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---
# Calories Burnt Prediction 🔥

This app predicts calories burned based on your activity data.

## 🚀 Live Demo
 👉 [Click Here to Try](https://calories-burnt-prediction-1.streamlit.app/)

---

## 📊 Project Overview

This project uses **Random Forest Regressor** to predict calories burnt during exercise with **99.8% accuracy (R² = 0.998)**. The model considers:

- **Physical characteristics**: Gender, Age, Height, Weight
- **Exercise metrics**: Duration, Heart Rate, Body Temperature

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Calories-Burnt-Prediction.git
cd Calories-Burnt-Prediction
```

### 2️⃣ Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
# Run the Streamlit web app
streamlit run app.py

# Or train the model from scratch
python calories_prediction_beginner.py
```

## 📁 Project Structure

```
Calories-Burnt-Prediction/
│
├── Dataset/
│   ├── exercise.csv          # Exercise data (User info + workout metrics)
│   └── calories.csv          # Target data (Calories burnt)
│
├── calories_prediction_beginner.py  # 📘 Main training script (BEGINNER FRIENDLY!)
├── app.py                           # 🌐 Streamlit web application
├── requirements.txt                 # 📦 Python dependencies
│
├── calories_model.pkl        # 🤖 Trained model + scaler (USE THIS!)
├── scaler.pkl               # 📊 StandardScaler (for preprocessing)
├── model_only.pkl           # 🤖 Model only (without scaler)
│
├── data_exploration.png     # 📈 EDA visualizations
├── correlation_heatmap.png  # 🔗 Feature correlations
├── feature_importance.png   # 🎯 Feature importance chart
├── model_predictions.png    # 📊 Actual vs Predicted plots
│
└── README.md                # 📖 This file
```

## 📊 Dataset Information

| Feature | Description | Type |
|---------|-------------|------|
| Gender | Male/Female | Categorical |
| Age | Age in years | Numeric |
| Height | Height in cm | Numeric |
| Weight | Weight in kg | Numeric |
| Duration | Exercise duration (minutes) | Numeric |
| Heart_Rate | Heart rate (bpm) | Numeric |
| Body_Temp | Body temperature (°C) | Numeric |
| Calories | **Target** - Calories burnt | Numeric |

**Dataset Size**: 15,000 samples

## 🤖 Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.9661 | 11.47 | 8.86 |
| **Random Forest** | **0.9980** | **2.76** | **1.68** |
| Gradient Boosting | 0.9966 | 3.58 | 2.52 |

🏆 **Best Model: Random Forest Regressor**

## 💻 How to Use the Model

### Method 1: Using the Web App
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Method 2: Python Script
```python
import pickle
import numpy as np

# Load the model and scaler
with open('calories_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# Prepare your data [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]
# Gender: 1=Male, 0=Female
input_data = np.array([[1, 25, 175, 70, 30, 120, 40.0]])

# Scale the input data (IMPORTANT!)
input_scaled = scaler.transform(input_data)

# Make prediction
calories = model.predict(input_scaled)
print(f"Predicted Calories: {calories[0]:.2f} kcal")
```

## 📈 Data Visualizations

### Feature Importance
The most important features for predicting calories burnt are:
1. **Duration** - Most important
2. **Heart Rate** - Second most important
3. **Body Temperature** - Third most important

### Correlation Analysis
- Duration, Heart Rate, and Body Temperature have **strong positive correlation** with Calories
- Gender has moderate impact on calorie prediction

## 🛠️ Technical Details

### Machine Learning Pipeline
1. **Data Loading**: Merge exercise.csv and calories.csv
2. **Preprocessing**: 
   - Remove User_ID (not useful for prediction)
   - Encode Gender (Male=1, Female=0)
   - Handle outliers using 3-sigma rule
3. **Feature Scaling**: StandardScaler for normalization
4. **Model Training**: Random Forest with 100 trees
5. **Model Saving**: Pickle format with scaler included

### Key Libraries
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning
- **matplotlib/seaborn** - Visualization
- **streamlit** - Web app

## 📝 Files Description

| File | Description |
|------|-------------|
| `calories_prediction_beginner.py` | Complete ML workflow with detailed comments (START HERE!) |
| `app.py` | Streamlit web application for predictions |
| `calories_model.pkl` | Trained model + scaler - use this for predictions |
| `requirements.txt` | All Python package dependencies |

## 🔧 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "File not found" error
Make sure you're in the correct directory and the Dataset folder exists with both CSV files.

### Model prediction seems wrong
Make sure you're **scaling the input data** before prediction! The model was trained on scaled data.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
<div align="center">

## 👨‍💻 Created By

### Karthik Vana

#scan the QR code below:

<img src="karthik-vana.png" alt="Scan for Live Demo" width="200">

**Data Science Enthusiast | Machine Learning Engineer | AI Engineer**

*Building practical ML solutions for real-world problems*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/karthik-vana)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/karthik-vana/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](karthikvana236@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white)](https://portfolio-v-smoky.vercel.app/)

## 💼 Open to Data Science & ML opportunities

**Made with ❤️ and Python**

⭐ *Star this repository if you found it helpful!* ⭐

*Last Updated: December 2025*
<div></div>
