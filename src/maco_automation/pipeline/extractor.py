# src/maco_automation/pipeline/extractor.py

import re
import os
import joblib
import pandas as pd
from typing import Optional, Union, List  # [CHANGED] Added for Python 3.8 support

from .reference_dicts import (
    MODEL_DICTIONARY,
    MODEL_PATTERNS,
    PATTERN_HINTS,
    APPLICATION_MAP,
    MANUFACTURER_KEYWORDS,
    CONDITION_KEYWORDS,
    SPARE_KEYWORDS,
    UNIT_KEYWORDS,
    OTHER_MACHINE_KEYWORDS,
)

# [EXISTING] Import Lift Specific References
try:
    from .reference_lift import FUEL_KEYWORDS, CATEGORY_KEYWORDS
except ImportError:
    print("⚠️ Warning: reference_lift.py not found. Lift specific features will be empty.")
    FUEL_KEYWORDS = {}
    CATEGORY_KEYWORDS = {}

# [NEW] Import Steel Shot Specific Logic
try:
    from .reference_steel_shot import enrich_steel_shot_data
except ImportError:
    print("⚠️ Warning: reference_steel_shot.py not found. Skipping specific logic.")
    enrich_steel_shot_data = lambda df: df  # No-op fallback
    
try:
    from .reference_ds import enrich_ds_data
except ImportError:
    print("⚠️ Warning: reference_ds.py not found. Skipping specific logic.")
    enrich_ds_data = lambda df: df

# -----------------------------------------------------------------
# Load ML Model (TF-IDF + Logistic Regression pipeline)
# -----------------------------------------------------------------
_PRODUCT_PIPELINE = None

def load_product_model(model_path: str = "models/product_pipeline.pkl"):
    """Load trained ML model for Product classification (fallback)."""
    global _PRODUCT_PIPELINE
    if os.path.exists(model_path):
        try:
            _PRODUCT_PIPELINE = joblib.load(model_path)
            print(f"[+] Product classification model loaded: {model_path}")
        except Exception as e:
            print(f"[!] Error loading product model: {e}")
    else:
        print(f"[!] Product model not found at: {model_path}")


# -----------------------------------------------------------------
# Text Cleaning Utility
# -----------------------------------------------------------------
def clean_desc(text: str) -> str:
    """Normalize text for pattern matching."""
    if not isinstance(text, str):
        return ""
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------------------------------------------
# Enhanced Model Extraction (Manufacturer-aware)
# -----------------------------------------------------------------
# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_model(desc: str, manufacturer: Optional[str] = None) -> Optional[str]:
    """
    Extract and normalize model numbers.
    Combines manufacturer-specific and global regex patterns with alias mapping.
    """
    desc = clean_desc(desc)
    manufacturer = (manufacturer or "").upper()

    # Step 1: Try manufacturer-based targeted patterns (for precision)
    manu_patterns = {
        # Rotary
        "DEUBLIN": [r"\b\d{3,4}-\d{3}-\d{3,6}\b", r"\bBC-\d{5}-\d{2}-\d{2}\b"],
        "MOOG": [r"\bA\d{2,3}-\d{4,}-\d{2}[A-Z]?\b", r"\bD0\d{2,}-\d{4,}\b"],
        "KADANT": [r"\bKD\d{2,4}\b", r"\bRIN-\d{2}-\d{2}-\d{4}\b"],
        "MAIER": [r"\bMJ\d{3}\b"],
        "GIROL": [r"\bS\d{2,}-\d{4,}-\d{2}[A-Z]?\b", r"\bE-\d{4}-\d{2}-\w{2}\b"],
        "HBS": [r"\bSR\d{2,3}\b"],
        "MOFLON": [r"\bMK\d{3,}[A-Z0-9\-]*\b"],
        # Lift
        "JLG": [r"\b\d{3,4}AJ\b", r"\b\d{3,4}S\b", r"\b\d{3,4}SJ\b", r"\bE\d{3,4}\w*\b"],
        "GENIE": [r"\bGS-\d{2,4}\b", r"\bZ-\d{2,3}\/\d{2}\b", r"\bS-\d{2,3}\b"],
        "DINGLI": [r"\bJCPT\d{4}\w*\b", r"\bGTbz\d{2}\w*\b"],
    }

    if manufacturer in manu_patterns:
        for pat in manu_patterns[manufacturer]:
            match = re.search(pat, desc)
            if match:
                model_candidate = match.group(0)
                return MODEL_DICTIONARY.get(model_candidate, model_candidate)

    # Step 2: Global regex fallback
    for pat in MODEL_PATTERNS:
        match = re.search(pat, desc)
        if match:
            model_candidate = match.group(0)
            return MODEL_DICTIONARY.get(model_candidate, model_candidate)

    # Step 3: Fallback alias check
    for alias, master_model in MODEL_DICTIONARY.items():
        if alias in desc:
            return master_model

    return None


# -----------------------------------------------------------------
# Product Extraction Logic (Regex + ML Fallback)
# -----------------------------------------------------------------
# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_product_regex(desc: str) -> Optional[str]:
    """
    Legacy detection logic (Regex/ML) - Primarily used for ROTARY_UNION
    to distinguish between 'Rotary Union', 'Rotary Joint', 'Swivel', etc.
    """
    desc = clean_desc(desc)

    # Step 1: Keyword-based detection
    for spare_word in SPARE_KEYWORDS:
        if re.search(r"\b" + re.escape(spare_word.upper()) + r"\b", desc):
            return "Spare"

    # Step 2: Regex-based detection
    for pattern, label in PATTERN_HINTS.items():
        if re.search(pattern, desc):
            return label

    # Step 3: ML Fallback
    try:
        if _PRODUCT_PIPELINE is not None:
            pred = _PRODUCT_PIPELINE.predict([desc])[0]
            prob = _PRODUCT_PIPELINE.predict_proba([desc]).max()
            if prob >= 0.6:
                return pred
            else:
                return None
    except Exception as e:
        pass
    return None


# -----------------------------------------------------------------
# Lift Specific Helper Functions
# -----------------------------------------------------------------
# [CHANGED] Updated signature from `str | None` to `Optional[str]`
# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_lift_category(desc: str) -> Optional[str]:
    desc = clean_desc(desc)
    
    # 1. Check for specific lift categories first (Scissor, Boom, etc. from reference_lift)
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in desc:
                return category.upper()
                
    # 2. Fallback check explicitly for AWP variants
    awp_keywords = ["AWP", "AERIAL WORK PLATFORM", "AERIAL WORKING PLATFORM"]
    if any(kw in desc for kw in awp_keywords):
        return "AWP"
        
    return None

# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_fuel(desc: str) -> Optional[str]:
    desc = clean_desc(desc)
    for fuel_type, keywords in FUEL_KEYWORDS.items():
        for kw in keywords:
            if kw in desc:
                return fuel_type.lower()
    return None

# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_height(desc: str) -> Optional[str]:
    if not isinstance(desc, str):
        return None
        
    # We DO NOT use clean_desc() here because it strips decimal points (e.g., 13.8 -> 13 8)
    desc_upper = desc.upper()

    # Priority 1: Explicitly stated "WORKING HEIGHT" (e.g., "WORKING HEIGHT: 13.8M")
    work_ft = re.search(r"WORKING\s*HEIGHT[^0-9]*(\d{2,3}(?:\.\d+)?)\s?(?:FT|FEET|')", desc_upper)
    if work_ft:
        return work_ft.group(1)
        
    work_m = re.search(r"WORKING\s*HEIGHT[^0-9]*(\d{1,3}(?:\.\d+)?)\s?(?:M|METER)\b", desc_upper)
    if work_m:
        return work_m.group(1)

    # Priority 2: General FT matches (take the max if multiple heights are listed)
    ft_matches = re.findall(r"\b(\d{2,3}(?:\.\d+)?)\s?(?:FT|FEET|')\b", desc_upper)
    if ft_matches:
        max_ft = max([float(m) for m in ft_matches])
        # Return cleanly without trailing .0 if it's a whole number
        return str(int(max_ft)) if max_ft.is_integer() else str(max_ft)

    # Priority 3: General M matches (take the max if multiple heights are listed)
    m_matches = re.findall(r"\b(\d{1,3}(?:\.\d+)?)\s?(?:M|METER)\b", desc_upper)
    if m_matches:
        max_m = max([float(m) for m in m_matches])
        return str(int(max_m)) if max_m.is_integer() else str(max_m)
        
    return None
# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_yom(desc: str) -> Optional[str]:
    desc = clean_desc(desc)
    match = re.search(r"\b(?:YOM|YEAR|MFG|MFR)[\s:\.]?(\d{4})\b", desc)
    if match:
        year = int(match.group(1))
        if 1980 < year < 2030:
            return str(year)
    return None


# -----------------------------------------------------------------
# Standard Extractors
# -----------------------------------------------------------------
# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_application(desc: str) -> Optional[str]:
    """Infer industrial application or use-case."""
    desc = clean_desc(desc)
    for keyword, label in APPLICATION_MAP.items():
        if keyword.upper() in desc:
            return label
    return None


# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def clean_manufacturer(text: str, keywords: List[str]) -> Optional[str]:
    """Normalize manufacturer text using keyword list."""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return keyword.upper()
    return text.upper()


# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_manufacturer(desc: str) -> Optional[str]:
    """Identify manufacturer presence."""
    desc = clean_desc(desc)
    for manu in MANUFACTURER_KEYWORDS:
        if manu.upper() in desc:
            return desc
    return None


# [CHANGED] Updated signature from `str | None` to `Optional[str]`
def extract_condition(desc: str) -> Optional[str]:
    """Detect if item is New / Used / Refurbished."""
    if not isinstance(desc, str):
        return None
        
    desc = clean_desc(desc)
    
    # [CRITICAL FIX] Correctly iterate over the flat {keyword: target} map
    for keyword, target_condition in CONDITION_KEYWORDS.items():
        # Look for the exact keyword with word boundaries so "OLD" doesn't trigger on "FOLD"
        if re.search(r'\b' + re.escape(keyword.upper()) + r'\b', desc):
            return target_condition.capitalize()
            
    return None


def detect_context(desc: str, product_group: str = "") -> str:
    """
    Classifies row into categories based on User Hierarchy:
    1. Unit + Spares + Other Machines
    2. Unit + Other Machines
    3. Spare + Unit
    4. Spare
    5. Single Unit
    (Default: Others)
    """
    if not isinstance(desc, str):
        return "Others"
        
    desc_upper = desc.upper()
    
    # 1. Detect Presence of Key Concepts
    has_spare = any(keyword in desc_upper for keyword in SPARE_KEYWORDS)
    has_unit = any(keyword in desc_upper for keyword in UNIT_KEYWORDS)
    has_machine = any(keyword in desc_upper for keyword in OTHER_MACHINE_KEYWORDS)

    # [NEW LOGIC] Treat the product group name explicitly as a Unit Keyword
    if product_group:
        pg_clean = product_group.replace("_", " ").upper()
        # Word boundary search so "DS" doesn't trigger inside "HANDSAW"
        if re.search(r'\b' + re.escape(pg_clean) + r'\b', desc_upper):
            has_unit = True

    # 2. Apply Hierarchy (Most specific to least specific)

    # CASE: Unit + Spares + Other Machines
    if has_unit and has_spare and has_machine:
        return "Unit + Spares + Other Machines"

    # CASE: Unit + Other Machines
    if has_unit and has_machine and not has_spare:
        return "Unit + Other Machines"

    # CASE: Spare + Unit
    if has_unit and has_spare:
        if re.search(r'\b(AND|WITH|INCLUDING|\+|&)\b', desc_upper):
            return "Spare + Unit"
        else:
            return "Spare"

    # CASE: Spare
    if has_spare:
        return "Spare"

    # CASE: Single Unit
    if has_unit:
        return "Unit" 

    # CASE: Other Machine (Fallback)
    if has_machine:
        return "Other" 

    return "Others"

# -----------------------------------------------------------------
# Main Extraction Function with Logic Switch
# -----------------------------------------------------------------
def extract_features(df: pd.DataFrame, product_group: str = "ROTARY_UNION") -> pd.DataFrame:
    """
    Apply extraction functions based on the selected Product Group.
    """
    print(f"Extracting structured fields from {len(df)} rows using [{product_group}] logic...")

    # 1. Common Extractors
    # Manufacturer
    df["Manufacturer"] = df["product_description"].apply(extract_manufacturer)
    if "Seller Group" in df.columns:
        df["Manufacturer"] = df["Manufacturer"].fillna(df["Seller Group"])
    df["Manufacturer"] = df["Manufacturer"].apply(lambda x: clean_manufacturer(x, MANUFACTURER_KEYWORDS))

    # Model
    df["Model"] = df.apply(lambda r: extract_model(r["product_description"], r.get("Manufacturer")), axis=1)
    
    # Context (Spare/Unit) & Condition
    # [NEW LOGIC] Pass the product_group dynamically into the context detector
    df["Spare / Unit / Others"] = df["product_description"].apply(lambda x: detect_context(x, product_group))
    
    # 👇 [CRITICAL FIX] Stop forcing "Old/New" into the 'type' column globally
    df["Extracted_Condition"] = df["product_description"].apply(extract_condition)
    
    # Only use the 'type' column for Condition if the product is LIFT
    if product_group.upper() == "LIFT":
        df["type"] = df["Extracted_Condition"]

    # 2. PRODUCT COLUMN LOGIC
    # DEFAULT: Set Product = User Input (product_group)
    df["Product"] = product_group

    # 3. Logic Switch based on Input
    if product_group.upper() == "ROTARY_UNION":
        print("🔹 Applying ROTARY_UNION specific logic...")
        df["Product"] = df["product_description"].apply(extract_product_regex)
        df["Product"] = df["Product"].fillna("Rotary Union")
        df["Application"] = df["product_description"].apply(extract_application)

    elif product_group.upper() == "LIFT":
        print("🔹 Applying LIFT-specific extractors...")
        
        # 1. Category is populated independently (e.g., AWP, SCISSOR LIFT)
        df["category"] = df["product_description"].apply(extract_lift_category)
        
        # 2. Other specific specs
        df["Battery/Diesel"] = df["product_description"].apply(extract_fuel)
        df["Height (ft)"] = df["product_description"].apply(extract_height)
        df["YOM"] = df["product_description"].apply(extract_yom)
        
        # [FIX] Commented out the override below so Product strictly remains 'LIFT' for all rows
        # mask_lift_units = (df["category"].notna()) & (df["Spare / Unit / Others"].isin(["Unit", "Unit + Spare"]))
        # df.loc[mask_lift_units, "Product"] = "AWP"

    elif product_group.upper() == "STEEL_SHOT":
        print("🔹 Applying STEEL_SHOT specific logic...")
        df = enrich_steel_shot_data(df)
        
    elif product_group.upper() == "DS":
        print("🔹 Applying DS logic...")
        df = enrich_ds_data(df)
        
    elif product_group.upper() == "CARDAN_SHAFT":
        print("🔹 Applying CARDAN_SHAFT logic...")
        df["Application"] = df["product_description"].apply(extract_application)
    
    elif product_group.upper() == "BARREL_COUPLING":
        print("🔹 Applying BARREL_COUPLING logic...")
        df["Application"] = df["product_description"].apply(extract_application)
        
    elif product_group.upper() == "POLYMIDE_FILMS":
        print("🔹 Applying POLYMIDE_FILMS logic...")
        df["Application"] = df["product_description"].apply(extract_application)

    else:
        print(f"⚠️ Unknown product group '{product_group}'. Running base extraction.")
        df["Application"] = df["product_description"].apply(extract_application)

    # Ensure "product" column exists (lowercase p per target schema sometimes)
    df["product"] = df["Product"]

    # --- [NEW LOGIC] POST-PROCESSING FOR UNITS ---
    # If the row was NOT classified as a Spare/Machine (i.e. it is currently "Others")
    # and we have an established Product, default it to "Unit".
    mask_is_others = df["Spare / Unit / Others"] == "Others"
    mask_has_valid_product = df["Product"].notna() & (df["Product"].astype(str).str.strip() != "")
    
    df.loc[mask_is_others & mask_has_valid_product, "Spare / Unit / Others"] = "Unit"

    print("Feature extraction complete.")
    return df


# -----------------------------------------------------------------
# CLI Runner
# -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Extract product details from normalized dataset")
    parser.add_argument("--input", required=True, help="Path to normalized CSV file")
    parser.add_argument("--output", default="output/extracted.csv", help="Path for output CSV file")
    parser.add_argument("--model", default="models/product_pipeline.pkl", help="Path to product ML model")
    parser.add_argument("--group", default="ROTARY_UNION", help="Product group (ROTARY_UNION, LIFT, STEEL_SHOT)")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    load_product_model(args.model)

    df = pd.read_csv(input_path)
    
    df = extract_features(df, product_group=args.group)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved extracted CSV -> {output_path}")