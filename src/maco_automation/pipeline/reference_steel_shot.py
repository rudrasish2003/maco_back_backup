# src/maco_automation/pipeline/reference_steel_shot.py
import re
import pandas as pd

def classify_steel_shot_logic(model, description):
    """
    Derives Category, Type, and Product based on the extracted Model and Description.
    """
    # Handle cases where model might be None
    if not model:
        model = ""
    if not description:
        description = ""

    model_upper = str(model).upper().replace(" ", "").replace("-", "")
    desc_upper = str(description).upper()

    # --- DEFAULTS ---
    product = "Steel Shot"
    category = "Shots"
    type_ = "High Carbon" # Default for standard S-series

    # --- 1. PRODUCT & CATEGORY LOGIC ---
    if model_upper.startswith("G"):
        category = "Grits"
    elif "CUT WIRE" in desc_upper or model_upper.startswith("CW"):
        product = "Cut Wire Shot"
        category = "Cut Wire"
    elif model_upper.startswith("AZ") or "ZINC" in desc_upper:
        product = "Zinc Shot"
        category = "Shots"
        type_ = "Zinc" # Override type for Zinc
    elif "STAINLESS" in desc_upper:
        category = "Shots"
        type_ = "Stainless Steel"

    # --- 2. TYPE LOGIC (Material/Grade) ---
    # Only apply if we haven't already fixed the type (like Zinc or Stainless)
    if type_ not in ["Zinc", "Stainless Steel"]:
        if "TSP" in model_upper:
            type_ = "Special Grade - " + model
        elif "LOW CARBON" in desc_upper:
            type_ = "Low Carbon"
        elif model_upper in ["S110", "S170", "S70"]:
             # Heuristic: S70-S170 are often Low Carbon/Softer
            type_ = "Low Carbon"
        elif model_upper.isdigit():
             # Numeric code (Winoa raw data) - usually High Carbon unless specified
            type_ = "High Carbon"
        elif model_upper.startswith("S") and len(model_upper) > 1:
            type_ = "High Carbon"

    return product, category, type_

def get_company_type(name, role="SELLER"):
    """
    Determines if a company is a Manufacturer, Trader, Reseller, or End Customer.
    """
    name = str(name).upper()
    
    # --- SELLER LOGIC ---
    if role == "SELLER":
        # Known Manufacturers in Steel Shots Industry
        known_mfrs = ["WINOA", "SINTO", "ERVIN", "KAISER", "FROHN", "TOYOKU", "AIRBLAST", "PAN ABRASIVES"]
        # Generic Manufacturing Keywords
        mfr_keywords = ["MANUFACTURING", "WORKS", "INDUSTRIES", "TECHNOL", "PRODUC"]
        
        if any(k in name for k in known_mfrs + mfr_keywords):
            return "Manufacturer"
        return "Trader"

    # --- BUYER LOGIC ---
    elif role == "BUYER":
        # Keywords implying trading/reselling
        reseller_keywords = ["TRADING", "ENTERPRISE", "GLOBAL", "SUPPLY", "IMPEX", "EXPORT", "ABRASIVES", "STEEL"]
        
        if any(k in name for k in reseller_keywords):
            return "Reseller"
        return "End Customer"

def enrich_steel_shot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestration function called by extractor.py
    Expects df to have 'Model', 'product_description', 'SELLER', 'BUYER'.
    """
    def process_row(row):
        # 1. Get Extracted Data (populated by generic extractor)
        model = row.get("Model")
        desc = row.get("normalized_description") 
        if not desc or str(desc) == 'nan':
            desc = row.get("product_description", "")

        # 2. Run Classification Logic
        product, category, type_ = classify_steel_shot_logic(model, desc)

        # 3. Run Company Logic
        seller = str(row.get('SELLER', '')).upper()
        buyer = str(row.get('BUYER', '')).upper()
        
        seller_group = "Winoa Group" if "WINOA" in seller else seller 
        seller_type = get_company_type(seller, role="SELLER")
        buyer_type = get_company_type(buyer, role="BUYER")

        return pd.Series([product, category, type_, seller_group, seller_type, buyer_type])

    # Apply Logic
    cols = ['Product', 'Category', 'Type', 'Seller Group', 'Seller_Type', 'Buyer_Type']
    df[cols] = df.apply(process_row, axis=1)
    
    return df