# src/classifier.py
import pandas as pd
import joblib
import numpy as np

def load_model(model_path: str = "models\relevancy_classifier_v1.joblib"):
    """Load the relevancy classifier model."""
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


def predict_relevancy(df: pd.DataFrame, model) -> pd.DataFrame:
    """Run the classifier and add relevancy predictions + confidence."""
    df = df.copy()

    # Ensure required columns exist
    if "product_description" not in df.columns:
        raise KeyError("'product_description' column missing in DataFrame")

    # Prepare numeric columns if they exist
    for col in ["unit_price", "quantity", "match_score"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "")
                .str.replace(",", "")
                .replace("", np.nan)
                .astype(float)
            )

    # Predict
    # Use 'Model' if it exists (from extractor), otherwise fall back
    model_col_to_use = "Model" if "Model" in df.columns else "Normalized_Model_Internal"
    
    # Ensure all required columns for prediction exist
    required_cols = ["product_description"]
    cols_to_use = [c for c in ["unit_price", "quantity", "match_score"] if c in df.columns]
    
    # Check for missing columns
    missing_data_cols = [c for c in required_cols + cols_to_use if c not in df.columns]
    if missing_data_cols:
         raise KeyError(f"Missing required columns for prediction: {missing_data_cols}")
         
    X = df[required_cols + cols_to_use]
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Add to DataFrame
    # --- RENAMED COLUMN ---
    df["Relevancy"] = np.where(preds == 1, "Relevant", "Irrelevant")
    # ----------------------
    df["relevancy_confidence"] = probs.round(4) # Keep this new column

    print(f"Predicted relevancy for {len(df)} rows")
    return df# src/classifier.py
import pandas as pd
import joblib
import numpy as np

def load_model(model_path: str = "models\relevancy_classifier_v1.joblib"):
    """Load the relevancy classifier model."""
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


def predict_relevancy(df: pd.DataFrame, model) -> pd.DataFrame:
    """Run the classifier and add relevancy predictions + confidence."""
    df = df.copy()

    # Ensure required columns exist
    if "product_description" not in df.columns:
        raise KeyError("'product_description' column missing in DataFrame")

    # --- START FIX: Impute NaN values before prediction ---
    
    # 1. Fill NaN in the text column with an empty string
    df["product_description"] = df["product_description"].fillna("")
    
    # 2. Fill NaN in numeric columns with 0
    # We also robustly convert to numeric here, in case preprocess missed any
    cols_to_use = []
    for col in ["unit_price", "quantity", "match_score"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str)
                           .str.replace("$", "")
                           .str.replace(",", ""),
                    errors='coerce' # Convert any parsing errors to NaN
                )
                .fillna(0) # *** This is the key fix: fill all NaN with 0 ***
            )
            cols_to_use.append(col)
            
    print(f"Imputed NaN values in 'product_description' (with '') and {cols_to_use} (with 0).")
    # --- END FIX ---


    # Predict
    # Use 'Model' if it exists (from extractor), otherwise fall back
    model_col_to_use = "Model" if "Model" in df.columns else "Normalized_Model_Internal"
    
    # Ensure all required columns for prediction exist
    required_cols = ["product_description"]
    # 'cols_to_use' is already defined and imputed above
    
    # Check for missing columns
    missing_data_cols = [c for c in required_cols if c not in df.columns]
    if missing_data_cols:
         raise KeyError(f"Missing required columns for prediction: {missing_data_cols}")
         
    X = df[required_cols + cols_to_use]
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Add to DataFrame
    # --- RENAMED COLUMN ---
    df["Relevancy"] = np.where(preds == 1, "Relevant", "Irrelevant")
    # ----------------------
    df["relevancy_confidence"] = probs.round(4) # Keep this new column

    print(f"Predicted relevancy for {len(df)} rows")
    return df