import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import joblib


def preprocess_for_xgboost(df):
    
    '''cols_to_impute = ['Glucose', 'SkinThickness', 'Insulin']
    df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)

    # Apply KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    cols_to_impute_2 = ['Age', 'SkinThickness', 'BMI','BloodPressure']
    df[cols_to_impute_2] = df[cols_to_impute_2].replace(0, np.nan)


    df[cols_to_impute_2] = imputer.fit_transform(df[cols_to_impute_2])'''

    cols_to_impute = ['Glucose', 'SkinThickness', 'Insulin', 'Age', 'BMI', 'BloodPressure']

# Replace 0s with NaNs in all selected columns
    df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)

# Impute missing values using mean (use 'median' if needed)
    imputer = SimpleImputer(strategy='mean')
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])


    return df


def preprocess_for_logreg(df):
    df["Glucose"] = df["Glucose"].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    df[["Glucose"]] = imputer.fit_transform(df[["Glucose"]])

    cols_to_impute = ['BloodPressure', 'BMI', 'Insulin', 'SkinThickness']
    df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)

# Apply KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    # feature engineering 
    new_features = ['BMI', 'BloodPressure', 'Glucose', 'Insulin']

    for i, (f1, f2) in enumerate(combinations(new_features, 2), 1):
        df[f'f{i}'] = df[f1] * df[f2]
    
    df['glucose_bmi'] = df['Glucose'] * df['Age']
    df['glucose_age'] = df['Glucose'] * df['Age']
    df['insulin_bmi_ratio'] = df['Insulin'] / (df['BMI'] + 0.01)

# Create health ratios
    df['glucose_insulin_ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    df['pregnancies_per_age'] = df['Pregnancies'] / (df['Age'] + 0.01)

# Risk composite scores
    df['metabolic_risk'] = (df['Glucose']/100) + (df['BMI']/30) + (df['Age']/50)

    feature_columns = [col for col in df.columns if col != 'Outcome']

    # clip outliers
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    X = df.drop(columns=['Outcome'])  # all columns except Outcome
    y = df['Outcome']  # keep Outcome separately

# Apply RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame (optional but helpful)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Recombine with Outcome
    df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)


    return df


def preprocess_for_NN(df):

    # missing value handling 
    df["Glucose"] = df["Glucose"].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[["Glucose"]] = imputer.fit_transform(df[["Glucose"]])
    
    cols_to_impute = ['BloodPressure', 'BMI', 'Insulin', 'SkinThickness']
    df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)
    imputer = KNNImputer(n_neighbors=5)
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    df['glucose_bmi'] = df['Glucose'] * df['Age']
    df['glucose_age'] = df['Glucose'] * df['Age']
    df['insulin_bmi_ratio'] = df['Insulin'] / (df['BMI'] + 0.01)

# Create health ratios
    df['glucose_insulin_ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    df['pregnancies_per_age'] = df['Pregnancies'] / (df['Age'] + 0.01)

# Risk composite scores
    df['metabolic_risk'] = (df['Glucose']/100) + (df['BMI']/30) + (df['Age']/50)
    '''
    df['age_bmi'] = df['BMI'] * df['Age']
    df['pregnancies_age'] = df['Pregnancies'] * df['Age']
    df['skin_bmi'] = df['SkinThickness'] / (df['BMI']+0.01)
    df['pressure_age'] = df['BloodPressure'] * df['Age']'''

    
    #outlier clipping
    feature_columns = [col for col in df.columns if col != 'Outcome']
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)


       
    #scaling
    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    
    return df