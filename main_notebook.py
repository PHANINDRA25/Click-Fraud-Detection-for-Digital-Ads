# Databricks notebook source
# 📝 main_pipeline.ipynb

print("🏁 Starting Full ML Pipeline: Click Ad Fraud Detection")

# Step 1: Preprocess Training Data
print("🔹 Step 1: Preprocessing training data...")
%run ./notebooks/data_preprocessing.py

# Step 2: Exploratory Data Analysis
print("🔹 Step 2: Running EDA...")
%run ./notebooks/EDA.py

# Step 3: Train Model with LightGBM + MLflow Tracking
print("🔹 Step 3: Training model...")
%run ./notebooks/train_lightgbm.py

# Step 4: Preprocess Hold-out Test Data
print("🔹 Step 4: Preprocessing test data...")
%run ./notebooks/test_preprocessing.py

# Step 5: Evaluate Best Model on Test Set
print("🔹 Step 5: Evaluating model on hold-out test set...")
%run ./notebooks/evaluate.py

# Step 6: Run Batch Inference on New Data (optional)
print("🔹 Step 6: Running batch prediction...")
%run ./batch_inference.py

print("✅ Pipeline completed successfully!")
