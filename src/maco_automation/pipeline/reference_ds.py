# src/maco_automation/pipeline/reference_ds.py
import re
import pandas as pd

def classify_ds_logic(description):
    """
    Derives Product, Category, and Type for DS (Desulphurizing Reagents).
    """
    if not description:
        description = ""
    
    desc_upper = str(description).upper()

    # --- 1. PRODUCT ---
    product = "DS"

    # --- 2. CATEGORY ---
    category = "Unknown"
    if "CALCIUM CARBIDE" in desc_upper:
        category = "CALCIUM CARBIDE"
    elif "CALCIUM" in desc_upper or "CA " in desc_upper or "CA:" in desc_upper:
        category = "CA"
    elif "MAGNESIUM" in desc_upper or "MG" in desc_upper:
        category = "MG"

    # --- 3. TYPE (Size Extraction) ---
    # Patterns: "0.1-2MM", "30-80 MM", "25-50MM"
    # Captures ranges like X-Y MM
    type_ = "NA"
    
    # Regex for range (e.g., 0.1-2 MM)
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*MM", desc_upper)
    
    if range_match:
        # Standardize format to "X-Y MM"
        start, end = range_match.group(1), range_match.group(2)
        type_ = f"{start}-{end} MM"
    
    return product, category, type_

def enrich_ds_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestration function called by extractor.py
    """
    def process_row(row):
        # Get description (prioritize normalized, fallback to raw)
        desc = row.get("normalized_description") 
        if not desc or str(desc) == 'nan':
            desc = row.get("product_description", "")

        # Run Logic
        product, category, type_ = classify_ds_logic(desc)

        # DS doesn't seem to use the standard 'Model' column in your reference file,
        # but we return it to fit the schema if needed.
        return pd.Series([product, category, type_])

    # Apply Logic
    cols = ['Product', 'Category', 'Type']
    df[cols] = df.apply(process_row, axis=1)
    
    return df