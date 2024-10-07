# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:40:00 2024

@author: gabv1
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Feature Selection
# Numerical features used for the model
selected_numerical_features = ['i_eva', 'i_tad', 'i_tas', 'i_fr']

# Categorical features used for the model
selected_categorical_features = ['tipo_anestesia', 'sexo']

# Combine selected features
selected_features = selected_numerical_features + selected_categorical_features

# Step 2: Data Preparation
# Extract the selected features (X) and the target variable (y) from the dataset
X = all_data[selected_features]
y = all_data['dolor']  # Target variable: 'dolor' (pain)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Data Preprocessing
# Numerical data preprocessing (median imputation and standardization)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Categorical data preprocessing (most frequent imputation and one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Apply one-hot encoding to categorical variables
])

# Combine both numerical and categorical preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selected_numerical_features),
        ('cat', categorical_transformer, selected_categorical_features)
    ])

# Step 4: Logistic Regression Model
# Initialize Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)

# Create a pipeline that preprocesses the data and applies the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Step 5: Model Training
# Train the model on the training data
clf.fit(X_train, y_train)

# Step 6: Model Evaluation
# Predict pain (y_pred) and predict probabilities (y_prob) for the test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the classification report (precision, recall, F1-score)
class_report = classification_report(y_test, y_pred, target_names=['No Dolor', 'Dolor'])
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", class_report)

# %% Step 7: AUROC Curve

# Calculate the AUROC score
auroc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUROC = {auroc:.4f})', color='salmon', lw=2)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random (AUROC = 0.5000)')

# Customize plot labels and title
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)

# Display the legend and plot
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()