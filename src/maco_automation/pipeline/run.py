import os
import sys
import pandas as pd
from datetime import datetime

# --- Path Setup ---
# Ensure the package root is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# --- Local Imports ---
from pipeline.preprocess import preprocess_csv
from pipeline.normalize import normalize_dataset
from pipeline.unit_converter import standardize_units
from pipeline.extractor import extract_features, load_product_model
from pipeline.classifier import load_model, predict_relevancy
from pipeline.llm_refiner import refine_batch  # [NEW] Import LLM Refiner
from pipeline.validator import validate_dataset

# --- Configuration ---
# Use absolute paths relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..", "..") # Adjust as needed based on where you run this

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "relevancy_classifier_v1.joblib")
PRODUCT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "product_pipeline.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(input_path: str, output_dir: str = OUTPUT_DIR, product_group: str = "ROTARY_UNION"):
    """
    Run full MACO data automation pipeline on given CSV file.
    Args:
        product_group: 'ROTARY_UNION', 'LIFT', etc. Controls specific extraction logic.
    """

    print(f"\n🚀 Starting MACO Pipeline for file: {input_path} | Product Group: {product_group}")

    try:
        # ---------------------------------------------------------------
        # [0] PREPROCESSING
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[0] Reading and Preprocessing Input CSV")
        print("=" * 70)
        df, sample = preprocess_csv(input_path)
        print(f"✅ Preprocessing complete — {len(df):,} rows, {len(df.columns)} columns")
        
        # ---------------------------------------------------------------
        # [2] NORMALIZATION
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[2] Normalizing Company / Model Data")
        print("=" * 70)
        df = normalize_dataset(df)
        print(f"✅ Normalization done — Columns: {len(df.columns)}")

        # ---------------------------------------------------------------
        # [2.5] UNIT & QUANTITY STANDARDIZATION
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[2.5] Standardizing Units & Quantities (KGS/PCS -> NOS)")
        print("=" * 70)
        df = standardize_units(df)
        print("✅ Unit conversion complete")

        # ---------------------------------------------------------------
        # [3] FEATURE EXTRACTION
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"[3] Extracting Product Features ({product_group} Logic)")
        print("=" * 70)

        # Load ML model for Product fallback (if available)
        print("\n[3.1] Loading ML Product Classifier (if available)")
        try:
            load_product_model(PRODUCT_MODEL_PATH)
        except Exception as e:
            print(f"⚠️ Could not load product model (skipping ML fallback): {e}")

        # Pass product_group to extractor
        df = extract_features(df, product_group=product_group)
        print("✅ Extraction done")

        # ---------------------------------------------------------------
        # [4] RELEVANCY CLASSIFICATION
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[4] Running Relevancy Classifier")
        print("=" * 70)
        
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            df = predict_relevancy(df, model)
            print("✅ Classification complete")
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}. Skipping Relevancy Classification.")
            df['Relevancy'] = 'Relevant' # Safest default fallback

        # ---------------------------------------------------------------
        # [4.5] LLM REFINEMENT
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[4.5] Running LLM Data Refinement & Applying Fallbacks")
        print("=" * 70)
        df = refine_batch(df, product_group)
        print("✅ LLM Refinement & Fallbacks complete")

        # ---------------------------------------------------------------
        # [5] VALIDATION
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[5] Running Post-Classification Validation")
        print("=" * 70)
        
        # Ensure output dirs exist
        audit_dir = os.path.join(output_dir, "audit")
        os.makedirs(audit_dir, exist_ok=True)
        
        df, audit_df = validate_dataset(df)
        print(f"✅ Validation complete — {len(audit_df)} issues logged")

        # ---------------------------------------------------------------
        # [6] FORMATTING & EXPORT RESULTS
        # ---------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[6] Formatting and Exporting Final Results")
        print("=" * 70)

        # --- ALIGN COLUMNS (Input First, Processed After) ---
        # Known processed columns to push to the right
        processed_cols = [
            "Seller Group", "Buyer Group", "Manufacturer", "Model", "Product", 
            "category", "Type", "Application", "Spare / Unit / Others", 
            "Battery/Diesel", "Height (ft)", "YOM", "Relevancy", "Match_Score",
            "Normalized_Model_Internal", "Unit_Normalized", "Extracted_Condition"
        ]
        
        all_cols = list(df.columns)
        # Identify original input columns (anything not in the processed list)
        input_cols = [c for c in all_cols if c not in processed_cols]
        
        # Reorder dataframe: Input Columns -> Processed Columns
        final_col_order = input_cols + [c for c in processed_cols if c in all_cols]
        df = df[final_col_order]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Processed File
        processed_path = os.path.join(output_dir, f"processed_{timestamp}.csv")
        df.to_csv(processed_path, index=False, encoding="utf-8-sig")
        
        # Save Audit Reports
        audit_path = os.path.join(audit_dir, f"audit_report_{timestamp}.csv")
        audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
        
        print(f"💾 Saved final processed file to: {processed_path}")
        print(f"💾 Saved audit report to: {audit_path}")

        print("\n🎯 Pipeline completed successfully!")
        
        # [UPDATED]: Return all 4 values so main.py can save the audit report to GridFS
        return df, audit_df, processed_path, audit_path

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise e