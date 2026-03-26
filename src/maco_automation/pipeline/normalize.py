# src/maco_automation/pipeline/normalize.py
import re
import pandas as pd
from rapidfuzz import fuzz, process
from .reference_dicts import (
    COMPANY_DICTIONARY,
    MODEL_DICTIONARY,
    PATTERN_HINTS,
    SPARE_KEYWORDS,
    UNIT_KEYWORDS,
    OTHER_MACHINE_KEYWORDS,
    UNIT_MAP,
    MODEL_PATTERNS,
)

# ---------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    """Standardize text for robust matching."""
    if not isinstance(text, str):
        return ""
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------------------------------------------
# COMPANY NORMALIZATION
# ---------------------------------------------------------------
def normalize_company(name: str):
    """Normalize company using dictionary and fuzzy logic."""
    if pd.isna(name) or str(name).strip() == "":
        return None, 0
        
    clean_name = clean_text(str(name))
    if not clean_name:
        return None, 0

    # 1. Exact match first
    if clean_name in COMPANY_DICTIONARY:
        return COMPANY_DICTIONARY[clean_name], 100

    # 2. BULLETPROOF SUBSTRING MATCH
    # Sort keys by length descending (e.g., check "ZOOMLION HEAVY" before "ZOOMLION")
    sorted_keys = sorted(COMPANY_DICTIONARY.keys(), key=len, reverse=True)
    
    for dict_key in sorted_keys:
        # \b ensures we match whole words only
        if re.search(r'\b' + re.escape(dict_key) + r'\b', clean_name):
            return COMPANY_DICTIONARY[dict_key], 90

    # 3. Fuzzy match fallback
    choices = list(COMPANY_DICTIONARY.keys())
    if not choices:
        return clean_name, 0

    match, score, _ = process.extractOne(clean_name, choices, scorer=fuzz.token_sort_ratio)
    
    if score >= 20:
        return COMPANY_DICTIONARY[match], score

    return clean_name, score

# ---------------------------------------------------------------
# MODEL NORMALIZATION
# ---------------------------------------------------------------
def normalize_model(desc: str):
    """Normalize model name using dictionary and regex hints."""
    clean_desc = clean_text(desc)
    if not clean_desc:
        return None, 0

    # Direct match in dictionary
    for key, val in MODEL_DICTIONARY.items():
        if key in clean_desc:
            return val, 100

    # Regex extraction via model patterns
    for pattern in MODEL_PATTERNS:
        match = re.search(pattern, clean_desc)
        if match:
            code = match.group(0)
            if code in MODEL_DICTIONARY:
                return MODEL_DICTIONARY[code], 95
            return code, 80  # Unlisted model found

    # Pattern hints (generic keywords)
    for pattern, val in PATTERN_HINTS.items():
        if re.search(pattern, clean_desc):
            return val, 90

    # Fuzzy fallback
    if MODEL_DICTIONARY:
        match, score, _ = process.extractOne(clean_desc, list(MODEL_DICTIONARY.keys()), scorer=fuzz.partial_ratio)
        if score > 75:
            return MODEL_DICTIONARY[match], score

    return None, 0

# ---------------------------------------------------------------
# UNIT NORMALIZATION
# ---------------------------------------------------------------
def normalize_unit(unit: str) -> str:
    """Standardize measurement units."""
    if not isinstance(unit, str):
        return "nos"
    unit = unit.strip().lower()
    return UNIT_MAP.get(unit, unit)

# ---------------------------------------------------------------
# PRODUCT TYPE CLASSIFICATION
# ---------------------------------------------------------------
def classify_product_type(description: str) -> str:
    """Classify product as Spare / Unit / Combined / Other."""
    desc = clean_text(description)

    if any(k in desc for k in SPARE_KEYWORDS):
        if any(u in desc for u in UNIT_KEYWORDS):
            return "Spare + Unit"
        elif any(o in desc for o in OTHER_MACHINE_KEYWORDS):
            return "Spare + Other Machine"
        return "Spare"
    elif any(u in desc for u in UNIT_KEYWORDS):
        if any(o in desc for o in OTHER_MACHINE_KEYWORDS):
            return "Unit + Other Machine"
        return "Unit"
    elif any(o in desc for o in OTHER_MACHINE_KEYWORDS):
        return "Other Machine"
    return "Unknown"

# ---------------------------------------------------------------
# MAIN DATASET NORMALIZATION
# ---------------------------------------------------------------
def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization logic on dataset."""
    print("Running normalization...")

    norm_sellers, seller_scores = [], []
    norm_buyers, buyer_scores = [], []
    norm_models, model_scores = [], []
    product_types, norm_units = [], []

    for _, row in df.iterrows():
        # --- 1. Seller Normalization (Separated) ---
        raw_seller = row.get("seller")
        n_seller, s_score = normalize_company(raw_seller)
        norm_sellers.append(n_seller)
        seller_scores.append(s_score)

        # --- 2. Buyer Normalization (Separated) ---
        raw_buyer = row.get("buyer")
        n_buyer, b_score = normalize_company(raw_buyer)
        norm_buyers.append(n_buyer)
        buyer_scores.append(b_score)

        # --- 3. Model Normalization ---
        desc = row.get("product_description", "")
        n_model, m_score = normalize_model(desc)
        norm_models.append(n_model)
        model_scores.append(m_score)

        # --- 4. Product Type classification ---
        product_types.append(classify_product_type(desc))

        # --- 5. Unit normalization ---
        norm_units.append(normalize_unit(row.get("unit", "")))

    # --- ASSIGN COLUMNS ---
    df["Seller Group"] = norm_sellers
    df["Buyer Group"] = norm_buyers  # New Column
    df["Normalized_Model_Internal"] = norm_models
    
    # --- MATCH SCORE CALCULATION ---
    # Calculates confidence based on Model Score + Best available Company Score
    final_match_scores = []
    for s_scr, b_scr, m_scr in zip(seller_scores, buyer_scores, model_scores):
        # Use the highest score between Seller and Buyer as the "Company Confidence"
        best_company_score = max(s_scr, b_scr)
        
        # Average with Model Score
        avg_score = (best_company_score + m_scr) / 2
        final_match_scores.append(round(avg_score, 2))

    df["Match_Score"] = final_match_scores
    df["Spare / Unit / Others"] = product_types
    df["Unit_Normalized"] = norm_units

    print("Normalization complete.")
    return df

# ---------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Normalize cleaned dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed CSV")
    parser.add_argument("--output", type=str, default="output/normalized.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    print(f"Loading cleaned data: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    df_norm = normalize_dataset(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(out_path, index=False)
    print(f"Normalized CSV saved: {out_path}")