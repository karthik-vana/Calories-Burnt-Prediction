"""
=====================================================
🔥 CALORIES BURNT PREDICTION - BEGINNER FRIENDLY CODE
=====================================================

This script demonstrates a complete Machine Learning workflow 
for predicting calories burnt during exercise.

Author: AI Assistant
Date: February 2026

WHAT YOU WILL LEARN:
1. How to load and explore data (EDA)
2. How to preprocess data for ML
3. How to train and compare multiple models
4. How to save the best model using pickle
5. How to make predictions with saved model

=====================================================
"""

# =====================================================
# STEP 1: IMPORT LIBRARIES
# =====================================================
# First, we import all the tools (libraries) we need

print("="*60)
print("STEP 1: IMPORTING LIBRARIES")
print("="*60)

# Data manipulation
import pandas as pd          # For working with tables (dataframes)
import numpy as np           # For mathematical operations

# Data visualization
import matplotlib.pyplot as plt   # For creating plots
import seaborn as sns            # For beautiful statistical plots

# Machine Learning tools
from sklearn.model_selection import train_test_split   # Split data into train/test
from sklearn.preprocessing import StandardScaler       # Scale/normalize data
from sklearn.linear_model import LinearRegression      # Linear Regression model
from sklearn.ensemble import RandomForestRegressor     # Random Forest model
from sklearn.ensemble import GradientBoostingRegressor # Gradient Boosting model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# For saving/loading our trained model
import pickle

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print()


# =====================================================
# STEP 2: LOAD THE DATA
# =====================================================
# We have two CSV files that we need to combine

print("="*60)
print("STEP 2: LOADING THE DATA")
print("="*60)

# Load the exercise data (contains user info and exercise metrics)
print("📂 Loading exercise.csv...")
exercise_data = pd.read_csv('Dataset/exercise.csv')

# Load the calories data (contains calories burnt)
print("📂 Loading calories.csv...")
calories_data = pd.read_csv('Dataset/calories.csv')

# Show what columns each file has
print("\n📋 Exercise Data Columns:")
print(exercise_data.columns.tolist())

print("\n📋 Calories Data Columns:")
print(calories_data.columns.tolist())

print("✅ Data loaded successfully!")
print()


# =====================================================
# STEP 3: EXPLORE THE RAW DATA
# =====================================================
# Let's see what our data looks like

print("="*60)
print("STEP 3: EXPLORING RAW DATA")
print("="*60)

print("\n🔍 First 5 rows of EXERCISE data:")
print(exercise_data.head())

print("\n🔍 First 5 rows of CALORIES data:")
print(calories_data.head())

print("\n📊 Shape of Exercise data:", exercise_data.shape)
print("📊 Shape of Calories data:", calories_data.shape)
print()


# =====================================================
# STEP 4: MERGE THE DATASETS
# =====================================================
# Combine both datasets using the common column 'User_ID'

print("="*60)
print("STEP 4: MERGING DATASETS")
print("="*60)

# Merge the two datasets on 'User_ID'
# This combines the exercise info with calories burnt
df = pd.merge(exercise_data, calories_data, on='User_ID')

print("✅ Datasets merged successfully!")
print(f"📊 Combined dataset shape: {df.shape}")
print(f"   - {df.shape[0]} rows (samples)")
print(f"   - {df.shape[1]} columns (features)")

print("\n🔍 First 5 rows of MERGED data:")
print(df.head())
print()


# =====================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
# Understand our data better before building models

print("="*60)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# 5.1 Check data types
print("\n📋 Data Types and Info:")
print(df.info())

# 5.2 Check for missing values
print("\n🔍 Missing Values:")
missing = df.isnull().sum()
print(missing)
print(f"\n✅ No missing values!" if missing.sum() == 0 else f"⚠️ Found {missing.sum()} missing values!")

# 5.3 Statistical summary
print("\n📊 Statistical Summary:")
print(df.describe())

# 5.4 Check unique values for Gender
print("\n👥 Unique Gender values:")
print(df['Gender'].value_counts())

# 5.5 Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n🔄 Duplicate rows: {duplicates}")


# =====================================================
# STEP 6: DATA VISUALIZATION
# =====================================================
# Create plots to understand data patterns

print("\n" + "="*60)
print("STEP 6: DATA VISUALIZATION")
print("="*60)

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('📊 Calories Burnt - Data Exploration', fontsize=16, fontweight='bold')

# Plot 1: Distribution of Calories
axes[0, 0].hist(df['Calories'], bins=30, color='coral', edgecolor='black')
axes[0, 0].set_title('Distribution of Calories Burnt')
axes[0, 0].set_xlabel('Calories')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Calories vs Duration
axes[0, 1].scatter(df['Duration'], df['Calories'], alpha=0.5, c='teal')
axes[0, 1].set_title('Calories vs Duration')
axes[0, 1].set_xlabel('Duration (mins)')
axes[0, 1].set_ylabel('Calories')

# Plot 3: Calories vs Heart Rate
axes[0, 2].scatter(df['Heart_Rate'], df['Calories'], alpha=0.5, c='purple')
axes[0, 2].set_title('Calories vs Heart Rate')
axes[0, 2].set_xlabel('Heart Rate (bpm)')
axes[0, 2].set_ylabel('Calories')

# Plot 4: Calories by Gender
df.boxplot(column='Calories', by='Gender', ax=axes[1, 0])
axes[1, 0].set_title('Calories by Gender')
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Calories')

# Plot 5: Age distribution
axes[1, 1].hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
axes[1, 1].set_title('Distribution of Age')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Frequency')

# Plot 6: Body Temp vs Calories
axes[1, 2].scatter(df['Body_Temp'], df['Calories'], alpha=0.5, c='orange')
axes[1, 2].set_title('Calories vs Body Temperature')
axes[1, 2].set_xlabel('Body Temperature (°C)')
axes[1, 2].set_ylabel('Calories')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: data_exploration.png")

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('🔗 Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: correlation_heatmap.png")
plt.close('all')

print()


# =====================================================
# STEP 7: DATA PREPROCESSING
# =====================================================
# Prepare data for machine learning

print("="*60)
print("STEP 7: DATA PREPROCESSING")
print("="*60)

# 7.1 Remove unnecessary columns
print("\n🗑️ Removing 'User_ID' column (not useful for prediction)...")
df = df.drop('User_ID', axis=1)

# 7.2 Convert Gender to numeric (Encoding)
# Male = 1, Female = 0
print("🔄 Converting Gender to numeric (Male=1, Female=0)...")
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

print("\n📊 Data after preprocessing:")
print(df.head())

# 7.3 Check for any outliers
print("\n🔍 Checking for extreme outliers using IQR method...")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Outliers per column:")
print(outliers)

# 7.4 Remove any rows with 3-sigma outliers in Calories
mean_cal = df['Calories'].mean()
std_cal = df['Calories'].std()
df = df[(df['Calories'] > mean_cal - 3*std_cal) & (df['Calories'] < mean_cal + 3*std_cal)]
print(f"\n📊 Data shape after removing extreme outliers: {df.shape}")

print("\n✅ Data preprocessing complete!")
print()


# =====================================================
# STEP 8: SPLIT DATA INTO FEATURES AND TARGET
# =====================================================
# Separate what we want to predict (target) from input features

print("="*60)
print("STEP 8: SPLITTING FEATURES AND TARGET")
print("="*60)

# Features (X) - What we use to predict
# We use: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
X = df.drop('Calories', axis=1)

# Target (y) - What we want to predict
# Calories burnt
y = df['Calories']

print("📋 Feature columns (X):", X.columns.tolist())
print(f"📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

print("\n🔍 Sample features:")
print(X.head())
print()


# =====================================================
# STEP 9: SPLIT INTO TRAINING AND TESTING SETS
# =====================================================
# Train on 80% of data, test on 20%

print("="*60)
print("STEP 9: TRAIN-TEST SPLIT")
print("="*60)

# Split the data
# - test_size=0.2 means 20% for testing, 80% for training
# - random_state=42 ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"📊 Training set size: {X_train.shape[0]} samples")
print(f"📊 Testing set size: {X_test.shape[0]} samples")
print(f"📊 Split ratio: {X_train.shape[0]/len(X)*100:.1f}% train / {X_test.shape[0]/len(X)*100:.1f}% test")
print()


# =====================================================
# STEP 10: FEATURE SCALING (STANDARDIZATION)
# =====================================================
# Scale features to have mean=0 and std=1
# This helps many ML algorithms work better

print("="*60)
print("STEP 10: FEATURE SCALING")
print("="*60)

# Create the scaler
scaler = StandardScaler()

# Fit the scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Only transform (not fit) the test data
# This prevents data leakage!
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")
print("\n📊 Before scaling - First sample:")
print(f"   {X_train.iloc[0].values}")
print("📊 After scaling - First sample:")
print(f"   {X_train_scaled[0]}")
print()


# =====================================================
# STEP 11: TRAIN MULTIPLE MODELS
# =====================================================
# We'll train 3 different models and compare them

print("="*60)
print("STEP 11: TRAINING MACHINE LEARNING MODELS")
print("="*60)

# Dictionary to store our models
models = {}
results = {}

# MODEL 1: Linear Regression
# Simple model - assumes linear relationship
print("\n🤖 Training Model 1: Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr_model
print("   ✅ Trained!")

# MODEL 2: Random Forest Regressor
# Ensemble of decision trees - very powerful
print("\n🤖 Training Model 2: Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of each tree
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model
print("   ✅ Trained!")

# MODEL 3: Gradient Boosting Regressor
# Another powerful ensemble method
print("\n🤖 Training Model 3: Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,      # Number of boosting stages
    max_depth=5,           # Maximum depth of trees
    learning_rate=0.1,     # How much each tree contributes
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb_model
print("   ✅ Trained!")

print("\n✅ All models trained successfully!")
print()


# =====================================================
# STEP 12: EVALUATE MODELS
# =====================================================
# Test each model and see which performs best

print("="*60)
print("STEP 12: MODEL EVALUATION")
print("="*60)

print("\n📊 Performance Metrics:")
print("-" * 70)
print(f"{'Model':<25} {'R² Score':<15} {'RMSE':<15} {'MAE':<15}")
print("-" * 70)

best_model = None
best_score = -float('inf')

for name, model in models.items():
    # Make predictions on test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[name] = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }
    
    # Print results
    print(f"{name:<25} {r2:.6f}      {rmse:.4f}        {mae:.4f}")
    
    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model = name

print("-" * 70)
print(f"\n🏆 BEST MODEL: {best_model} with R² = {best_score:.6f}")
print()

# Explanation of metrics
print("📖 UNDERSTANDING THE METRICS:")
print("-" * 50)
print("R² Score (0-1): Higher is better. 1.0 = perfect prediction")
print("RMSE: Root Mean Squared Error - lower is better")
print("MAE: Mean Absolute Error - lower is better")
print()


# =====================================================
# STEP 13: VISUALIZE MODEL PREDICTIONS
# =====================================================
# Create plots comparing actual vs predicted values

print("="*60)
print("STEP 13: VISUALIZING PREDICTIONS")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('📊 Actual vs Predicted Calories', fontsize=14, fontweight='bold')

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    
    axes[idx].scatter(y_test, y_pred, alpha=0.5, c='teal')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    axes[idx].set_xlabel('Actual Calories')
    axes[idx].set_ylabel('Predicted Calories')
    axes[idx].set_title(f'{name}\nR² = {results[name]["R2"]:.4f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: model_predictions.png")
plt.close()

# Feature importance for Random Forest
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models['Random Forest'].feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
plt.xlabel('Importance')
plt.title('🎯 Feature Importance (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: feature_importance.png")
plt.close()
print()


# =====================================================
# STEP 14: SAVE THE BEST MODEL AND SCALER
# =====================================================
# Save the model using pickle so we can use it later

print("="*60)
print("STEP 14: SAVING THE BEST MODEL")
print("="*60)

# We save the Random Forest model (best performer)
# We also save the scaler because we need it for new predictions

# Get the best model object
final_model = models[best_model]

# Save both model and scaler together in one file
print(f"\n💾 Saving {best_model} model and scaler...")

# Method 1: Save as tuple (model, scaler) - recommended
with open('calories_model.pkl', 'wb') as f:
    pickle.dump((final_model, scaler), f)
print("   ✅ Saved: calories_model.pkl (model + scaler)")

# Method 2: Save scaler separately (alternative)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved: scaler.pkl (scaler only)")

# Method 3: Save model separately (alternative)
with open('model_only.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("   ✅ Saved: model_only.pkl (model only)")

print("\n✅ All files saved successfully!")
print()


# =====================================================
# STEP 15: LOAD MODEL AND MAKE PREDICTIONS
# =====================================================
# Demonstrate how to use the saved model

print("="*60)
print("STEP 15: LOAD MODEL AND MAKE PREDICTIONS")
print("="*60)

# Load the model and scaler
print("\n📂 Loading saved model and scaler...")
with open('calories_model.pkl', 'rb') as f:
    loaded_model, loaded_scaler = pickle.load(f)
print("   ✅ Loaded successfully!")

# Create a sample prediction
print("\n🧪 Making a sample prediction:")
print("-" * 50)

# Sample person data
sample_data = {
    'Gender': 1,          # 1 = Male, 0 = Female
    'Age': 25,            # Years
    'Height': 175,        # cm
    'Weight': 70,         # kg
    'Duration': 30,       # minutes of exercise
    'Heart_Rate': 120,    # bpm
    'Body_Temp': 40.0     # °C
}

print("📋 Input Data:")
for key, value in sample_data.items():
    print(f"   {key}: {value}")

# Convert to proper format
sample_array = np.array([[
    sample_data['Gender'],
    sample_data['Age'],
    sample_data['Height'],
    sample_data['Weight'],
    sample_data['Duration'],
    sample_data['Heart_Rate'],
    sample_data['Body_Temp']
]])

# Scale the data (IMPORTANT!)
sample_scaled = loaded_scaler.transform(sample_array)

# Make prediction
calories_predicted = loaded_model.predict(sample_scaled)

print(f"\n🔥 PREDICTED CALORIES BURNT: {calories_predicted[0]:.2f} kcal")
print()


# =====================================================
# STEP 16: INTERACTIVE PREDICTION DEMO
# =====================================================
def predict_calories(gender, age, height, weight, duration, heart_rate, body_temp):
    """
    Function to predict calories burnt.
    
    Parameters:
    -----------
    gender : int (1 for Male, 0 for Female)
    age : int (years)
    height : float (cm)
    weight : float (kg)
    duration : float (minutes)
    heart_rate : float (bpm)
    body_temp : float (°C)
    
    Returns:
    --------
    float : Predicted calories burnt
    """
    # Prepare input
    input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
    
    # Scale the input
    input_scaled = loaded_scaler.transform(input_data)
    
    # Predict
    prediction = loaded_model.predict(input_scaled)
    
    return prediction[0]


# Test the function
print("="*60)
print("STEP 16: TESTING PREDICTION FUNCTION")
print("="*60)

# Test case 1: Young male, intense workout
cal1 = predict_calories(1, 22, 180, 75, 60, 145, 40.5)
print(f"🏃 22yo Male, 60min intense: {cal1:.2f} kcal")

# Test case 2: Middle-aged female, moderate workout
cal2 = predict_calories(0, 40, 165, 60, 30, 110, 39.5)
print(f"🏃 40yo Female, 30min moderate: {cal2:.2f} kcal")

# Test case 3: Senior, light workout
cal3 = predict_calories(1, 65, 170, 80, 15, 90, 38.5)
print(f"🏃 65yo Male, 15min light: {cal3:.2f} kcal")

print()


# =====================================================
# FINAL SUMMARY
# =====================================================
print("="*60)
print("🎉 FINAL SUMMARY")
print("="*60)

print("""
✅ WHAT WE ACCOMPLISHED:
------------------------
1. Loaded and merged exercise & calories datasets
2. Performed Exploratory Data Analysis (EDA)
3. Preprocessed data (encoding, scaling)
4. Trained 3 different ML models
5. Evaluated and compared model performance
6. Saved the best model using pickle
7. Created a prediction function for new data

📁 FILES CREATED:
-----------------
• calories_model.pkl    - Best model + scaler (use this!)
• scaler.pkl           - Scaler only
• model_only.pkl       - Model only
• data_exploration.png - EDA visualizations
• correlation_heatmap.png - Feature correlations
• model_predictions.png - Actual vs Predicted plots
• feature_importance.png - Feature importance chart

🚀 HOW TO USE THE MODEL:
------------------------
1. Load the model:
   with open('calories_model.pkl', 'rb') as f:
       model, scaler = pickle.load(f)

2. Prepare your data:
   data = [[gender, age, height, weight, duration, heart_rate, body_temp]]

3. Scale and predict:
   scaled_data = scaler.transform(data)
   calories = model.predict(scaled_data)

🌐 RUN THE WEB APP:
-------------------
   streamlit run app.py
""")

print("="*60)
print("🎯 SCRIPT COMPLETED SUCCESSFULLY!")
print("="*60)
