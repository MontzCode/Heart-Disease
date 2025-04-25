# src/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer # Keep for reference

# Simple Imputation + Scaling for all features
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Add other pipeline definitions here if needed