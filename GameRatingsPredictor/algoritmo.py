import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# Define the path to your dataset CSV file (Corrected to the name you provided)
DATASET_PATH = "Video_Games.csv" 
# The target variable we want to predict (now in lowercase)
TARGET_COLUMN = 'rating'
# The file to save the best model to
MODEL_OUTPUT_PATH = "game_rating_classifier.joblib"

def load_and_clean_data(path):
    """
    Loads the dataset and performs initial cleaning and feature engineering.
    """
    print(f"Loading and cleaning data from {path}...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{path}' was not found.")
        print("Please make sure the CSV file is in the same directory as your script or provide the full path.")
        return None
    
    # --- ROBUSTNESS FIX: Standardize all column names to lowercase ---
    df.columns = df.columns.str.lower()
    
    # We will print the actual column names here for definitive debugging if needed
    print("Columns found in CSV (standardized to lowercase):")
    print(df.columns.tolist())

    # Columns to drop (now in lowercase)
    # Note: Added 'unnamed: 16' as it's often a an empty column in this dataset
    columns_to_drop = ['name', 'platform', 'year_of_release', 'genre', 'publisher', 'developer', 'global_sales']
    # Drop only the columns that actually exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns_to_drop)


    # --- Data Cleaning ---
    # Drop rows where the target variable is missing or 'RP'
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] != 'rp'] # 'rp' is now lowercase

    # Convert 'user_score' to numeric (now in lowercase)
    df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

    # Impute remaining missing values with the median
    df.fillna(df.median(), inplace=True)
    
    print(f"Data loaded. Shape after cleaning: {df.shape}")
    print(f"Target classes: {df[TARGET_COLUMN].unique()}")
    return df

def train_models(df):
    """
    Splits data, defines models and pipelines, and runs training with MLflow.
    """
    if df is None:
        print("DataFrame is not loaded. Aborting training.")
        return

    print("Starting model training and evaluation...")
    # Define features (X) and target (y)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {
            'classifier__C': [0.1, 1, 10]
        }),
        'KNeighborsClassifier': (KNeighborsClassifier(), {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance']
        }),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None]
        }),
        'XGBClassifier': (XGBClassifier(eval_metric='mlogloss', random_state=42), {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5, 7]
        })
    }

    best_f1_score = -1
    best_model = None

    for name, (model, params) in models.items():
        print(f"\n--- Training {name} ---")
        
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_estimator = grid_search.best_estimator_
            
            y_pred = best_estimator.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(best_estimator, f"model_{name}")
            
            print(f"Results for {name}:")
            print(f"  Best Params: {grid_search.best_params_}")
            print(f"  F1 Score (Weighted): {f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = best_estimator
                print(f"*** New best model found: {name} ***")

    if best_model:
        print(f"\nSaving best overall model to {MODEL_OUTPUT_PATH}")
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
    else:
        print("No model was successfully trained.")

def main():
    """
    Main function to run the complete pipeline.
    """
    df = load_and_clean_data(DATASET_PATH)
    train_models(df)
    print("\nProcess finished. Run 'mlflow ui' to see experiment results.")


if __name__ == '__main__':
    main()