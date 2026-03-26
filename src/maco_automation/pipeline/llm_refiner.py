import os
import json
import re
import pandas as pd
from openai import OpenAI
from typing import List, Dict

# [NEW] Import the shared dictionary for Company and Model Checks
from .reference_dicts import COMPANY_DICTIONARY, MODEL_DICTIONARY

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_dynamic_examples(product_group: str) -> List[Dict]:
    """Reads 'Raw' and 'Clean' CSVs from 'reference_data/' folder."""
    
    file_map = {
        "STEEL_SHOT":   ("Steel Shots - RAW.csv", "Steel Shots - Clean.csv"),
        "LIFT":         ("Access AWP INT (lift) - Raw.csv", "Access AWP INT (lift) - Clean.csv"),
        "DS":           ("DS - Raw.csv", "DS - Processed.csv"),
        "CARDAN_SHAFT": ("Cardan Shaft - Raw.csv", "Cardan Shaft - Clean.csv"),
        "ROTARY_UNION": ("Additional Rotary Union - RAW DATA.csv", "Additional Rotary Union - Clean.csv")
    }
    
    key_map = {
        "Steel Shots": "STEEL_SHOT", "Access AWP INT (lift)": "LIFT",
        "Cardan Shaft": "CARDAN_SHAFT", "Additional Rotary Union": "ROTARY_UNION", "DS": "DS"
    }
    pg_key = key_map.get(product_group, product_group)
    
    raw_file, clean_file = file_map.get(pg_key, (None, None))
    if not raw_file: return []

    base_dir = os.path.join(os.getcwd(), "reference_data")
    raw_path = os.path.join(base_dir, raw_file)
    clean_path = os.path.join(base_dir, clean_file)

    examples = []
    try:
        if os.path.exists(raw_path) and os.path.exists(clean_path):
            df_raw = pd.read_csv(raw_path).head(6)
            df_clean = pd.read_csv(clean_path).head(6)
            
            # Identify columns to train on (union of all potential columns)
            target_cols = [
                'Product', 'Model', 'Category', 'Manufacturer', 'quantity', 'unit', 'unit_price', 
                'Type', 'TYPE', 'Application', 'Seller_Type', 'Buyer_Type', 'Spare / Unit / Others', 'Spare/Unit/Others',
                'Height (ft)', 'YOM', 'Battery/Diesel', 'Relevancy', 'SBU', 'Imp Companies',
                'Seller Group', 'Buyer Group'
            ]

            for i in range(min(len(df_raw), len(df_clean))):
                r_row = df_raw.iloc[i]
                c_row = df_clean.iloc[i]
                
                # Find description column safely
                desc_col = next((c for c in r_row.index if 'desc' in c.lower()), 'product_description')
                
                # Input: Raw Data
                user_input = {
                    "desc": str(r_row.get(desc_col, "")),
                    "raw_unit": str(r_row.get('unit', "")),
                    "raw_qty": str(r_row.get('quantity', ""))
                }
                
                # Output: The CORRECTED Clean Data
                assistant_output = {k: v for k, v in c_row.items() if k in target_cols and pd.notna(v)}
                
                examples.append({
                    "user": json.dumps(user_input),
                    "assistant": json.dumps(assistant_output)
                })
    except Exception as e:
        print(f"Error loading refs: {e}")
        
    return examples

def refine_batch(df_batch: pd.DataFrame, product_group: str) -> pd.DataFrame:
    if df_batch.empty: return df_batch
    
    examples = load_dynamic_examples(product_group)
    
    # --- 1. INTELLIGENT CONTEXT: Filter Company & Model Maps ---
    # Only inject relevant rules found in this batch's text to save tokens.
    search_text = " ".join(df_batch["product_description"].fillna("").astype(str).tolist()).upper()
    
    # Add other columns to search text
    for col in ["seller", "buyer", "manufacturer", "Seller Group", "Buyer Group", "Manufacturer"]:
        found_col = next((c for c in df_batch.columns if c.lower() == col.lower()), None)
        if found_col:
            search_text += " " + " ".join(df_batch[found_col].fillna("").astype(str).tolist()).upper()

    # Filter Company Rules
    relevant_company_rules = {}
    if COMPANY_DICTIONARY:
        sorted_keys = sorted(COMPANY_DICTIONARY.keys(), key=len, reverse=True)
        for variation in sorted_keys:
            clean_var = str(variation).upper().strip()
            # \b ensures we only match whole words
            if re.search(r'\b' + re.escape(clean_var) + r'\b', search_text):
                relevant_company_rules[clean_var] = COMPANY_DICTIONARY[variation]
    
    company_map_str = json.dumps(relevant_company_rules) if relevant_company_rules else "{}"

    # Filter Model Rules
    relevant_model_rules = {}
    if MODEL_DICTIONARY:
        # Sort keys by length descending to match specific models first (e.g. 1350SJP over 1350)
        sorted_model_keys = sorted(MODEL_DICTIONARY.keys(), key=len, reverse=True)
        for variation in sorted_model_keys:
            clean_var = str(variation).upper().strip()
            # Direct substring check for models (as model numbers often have hyphens/no spaces)
            if clean_var in search_text:
                relevant_model_rules[clean_var] = MODEL_DICTIONARY[variation]
                
    model_map_str = json.dumps(relevant_model_rules) if relevant_model_rules else "{}"

    # --- 2. PREPARE SPECIFIC INSTRUCTIONS ---
    specific_instructions = ""
    
    if product_group == "LIFT":
        specific_instructions = (
            "   - **LIFT Category Rules** (Strict hierarchy based on Description keywords):\n"
            "     1. 'Scissor' -> 'SCISSOR LIFT'\n"
            "     2. 'Spider' -> 'SPIDER LIFT'\n"
            "     3. 'Articulating' OR 'Knuckle' AND 'Boom' -> 'ARTICULATING BOOM'\n"
            "     4. 'Telescopic' OR 'Straight' AND 'Boom' -> 'TELESCOPIC BOOM'\n"
            "     5. 'Boom' (only, without Articulating/Telescopic) -> 'BOOM OTHERS'\n"
            "     6. 'AWP' or 'Aerial Work Platform' (only, no boom/scissor) -> 'AWP'\n"
            "   - **LIFT HEIGHT (CRITICAL FALLBACK)**: The automation sometimes fails to extract the height. If 'Current_Height (ft)' is 'MISSING', you MUST scan the Description and extract the working/platform height. Do NOT automatically extract other missing fields unless explicitly asked.\n"
            "   - **LIFT Specs**: Extract 'YOM' (Year) if available.\n"
            "   - **Type**: Identify if 'New' or 'Used'.\n"
        )
    elif product_group == "STEEL_SHOT":
        specific_instructions = (
            "   - **STEEL_SHOT Rules**:\n"
            "     1. **Model**: Standardize (e.g., 'S-330' -> 'S330', 'S 550' -> 'S550', 'GH-18' -> 'GH18').\n"
            "     2. **Category**: Classify as 'Shots' (Spherical) or 'Grits' (Angular).\n"
            "     3. **Type**: Classify as 'High Carbon', 'Low Carbon', or 'Stainless Steel'.\n"
            "     4. **Seller/Buyer Type**: Infer 'Manufacturer', 'Reseller', 'End Customer', or 'Trader'.\n"
        )
    elif product_group == "DS":
        specific_instructions = (
            "   - **DS (Desulphurizing) Rules**:\n"
            "     1. **Model**: 'CA' (Calcium), 'MG' (Magnesium), 'LIME', 'CALCIUM CARBIDE', 'SODIUM'.\n"
            "     2. **TYPE**: Extract Size (e.g., '0.5-2MM') or Purity (e.g., '98.5%').\n"
        )
    elif product_group == "ROTARY_UNION":
        specific_instructions = (
            "   - **ROTARY_UNION Rules**:\n"
            "     1. **Application**: Infer industry (e.g., 'Paper', 'Steel', 'Machine Tool', 'Rubber').\n"
            "     2. **Product**: Differentiate 'Rotary Union' vs 'Swivel Joint' vs 'Repair Kit'.\n"
        )

    # --- 3. SYSTEM PROMPT ---
    system_content = (
        f"You are a Data Quality Auditor for {product_group}.\n"
        "Audit the 'Current_Values' (from automation) against the 'Description'. Fix errors & normalize data.\n\n"
        f"### COMPANY NORMALIZATION MAP (Use exact mappings):\n{company_map_str}\n\n"
        f"### MODEL NORMALIZATION MAP (Use exact mappings):\n{model_map_str}\n\n"
        "### CHECKLIST:\n"
        "1. **Company Names (Seller/Buyer Group)** (STRICT ENFORCEMENT):\n"
        "   - CHECK 'Seller Group' and 'Buyer Group' against the COMPANY MAP above.\n"
        "   - IF a variation from the map is found anywhere in the raw text, you MUST REPLACE the output with the exact target value from the map.\n"
        "   - DO NOT be creative. DO NOT use outside intelligence to guess or normalize company names.\n"
        "   - IF there is no match in the map, leave the original cleaned name unchanged.\n"
        "2. **Model Extraction** (STRICT ENFORCEMENT):\n"
        "   - CHECK the Description against the MODEL MAP above.\n"
        "   - IF a source key from the MODEL MAP is found in the text, you MUST REPLACE 'Model' with the exact target value from the map.\n"
        "   - CRITICAL: Do NOT confuse Serial Numbers (e.g., '0300178158') with Models (e.g., '1350SJP' or 'SR220'). If a known model exists in the text, it overrides any extracted serial number.\n"
        "3. **Product Column**: DO NOT CHANGE. Leave as '{product_group}'.\n"
        "4. **Unit & Price (CRITICAL)**: \n"
        "   - If 'Current_Unit' is KGS/MTS but Description says count (e.g. '100 PCS'), FORCE Unit='NOS', Quantity=100.\n"
        "   - If Unit is 'NOS' but Qty=1 while Description says 'Box of 50', CHANGE Qty to 50.\n"
        "   - **ALWAYS** recalculate 'unit_price' if Quantity changes.\n"
        "5. **Core Attributes**: Verify 'Manufacturer', 'Spare / Unit / Others'. \n"
        "   - Scan Description to correct WRONG values.\n"
        f"{specific_instructions}" 
        "6. **Relevancy**: Verify if item is relevant to the product group.\n\n"
        "Output JSON with a 'data' list of corrected rows."
    )

    messages = [{"role": "system", "content": system_content}]
    
    for ex in examples:
        messages.append({"role": "user", "content": f"Example In: {ex['user']}"})
        messages.append({"role": "assistant", "content": f"Example Out: {ex['assistant']}"})

    # --- 4. BUILD PAYLOAD ---
    batch_data = []
    
    # Comprehensive list of columns to audit
    cols_to_check = [
        "Manufacturer", "Model", "Category", "Type", "TYPE", "Application", "Product",
        "Seller_Type", "Buyer_Type", "Spare / Unit / Others", "Relevancy",
        "Height (ft)", "YOM", "Battery/Diesel", "SBU", "Imp Companies",
        "Seller Group", "Buyer Group"
    ]

    # [NEW] Aggressive JSON Sanitizer to prevent OpenAI 400 Errors
    def sanitize_for_json(val):
        if pd.isna(val):
            return "MISSING"
        
        text = str(val).strip()
        if text.lower() in ["nan", "none", "nat", "<na>", ""]:
            return "MISSING"
            
        # Strip null bytes and unescaped control characters that crash JSON parsers
        text = text.replace('\x00', '')
        # Encode/Decode ignores broken UTF-8 surrogate pairs (Mojibake)
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        return text

    # [FIX 3A] Guarantee sequential integer indexes so we don't lose rows on mapping
    final_batch = df_batch.reset_index(drop=True).copy()

    for idx, row in final_batch.iterrows():
        item = {
            "id": str(idx), # Lock the ID strictly to the dataframe index
            "Description": sanitize_for_json(row.get("product_description")),
            "Current_Unit": sanitize_for_json(row.get("unit")),
            "Current_Qty": sanitize_for_json(row.get("quantity")),
            "Current_Value_USD": sanitize_for_json(row.get("valueusd")),
        }
        
        # Include current automation output for grading
        for col in cols_to_check:
            found_col = next((c for c in final_batch.columns if c.lower() == col.lower()), None)
            
            if found_col:
                item[f"Current_{col}"] = sanitize_for_json(row.get(found_col))
            else:
                item[f"Current_{col}"] = "MISSING"

        # Pass Seller/Buyer raw columns for normalization check
        item["Raw_Seller"] = sanitize_for_json(row.get("seller", row.get("SELLER", "")))
        item["Raw_Buyer"] = sanitize_for_json(row.get("buyer", row.get("BUYER", "")))

        batch_data.append(item)

    # Use allow_nan=False to strictly catch any rogue floating point numbers before they hit OpenAI
    try:
        payload_str = json.dumps(batch_data, allow_nan=False)
    except ValueError as e:
        print(f"JSON Serialization Warning: {e}. Falling back to aggressive string cast.")
        payload_str = str(batch_data)

    messages.append({"role": "user", "content": f"Audit Batch:\n{payload_str}"})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            response_format={"type": "json_object"}, 
            temperature=0.0,
            max_tokens=16384 # [NEW] Prevent cutoff
        )
        
        if response.choices[0].finish_reason == "length":
            print("⚠️ WARNING: LLM output was cut off due to length!")
            
        result = json.loads(response.choices[0].message.content)
        clean_df = pd.DataFrame(result.get("data", []))
        
        # [FIX 3B] Safely map rows back by their exact ID to prevent blank overwrites
        if len(clean_df) > 0 and "id" in clean_df.columns:
            for _, llm_row in clean_df.iterrows():
                try:
                    row_id = int(llm_row["id"]) # Parse the index back
                    if row_id in final_batch.index:
                        for col in clean_df.columns:
                            if col != "id":
                                match_col = next((c for c in final_batch.columns if c.lower() == col.lower()), col)
                                val = llm_row[col]
                                # Only update if LLM provided a valid answer
                                if pd.notna(val) and str(val).upper() != "MISSING":
                                    final_batch.at[row_id, match_col] = val
                except (ValueError, TypeError):
                    continue # Skip malformed row IDs

        # --- 5. PRODUCT COLUMN LOCK ---
        # Force the 'Product' column to match the uploaded product_group exactly.
        if "Product" in final_batch.columns:
             final_batch["Product"] = product_group
        else:
             final_batch["Product"] = product_group 

        return final_batch

    except Exception as e:
        print(f"LLM Error: {e}")
        # Even on error, ensure consistency
        if "Product" in final_batch.columns:
             final_batch["Product"] = product_group
        return final_batch