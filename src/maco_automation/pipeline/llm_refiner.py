import os
import json
import re
import time 
import pandas as pd
from openai import OpenAI
import random
from typing import List, Dict
from functools import lru_cache

# Import the shared dictionary for Company and Model Checks
from .reference_dicts import COMPANY_DICTIONARY, MODEL_DICTIONARY

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cache the reference files in RAM so threads don't bottleneck on disk reads
@lru_cache(maxsize=10)
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
            
            target_cols = [
                'Product', 'Model', 'Category', 'Manufacturer', 'quantity', 'unit', 'unit_price', 
                'Type', 'TYPE', 'Application', 'Seller_Type', 'Buyer_Type', 'Spare / Unit / Others', 'Spare/Unit/Others',
                'Height (ft)', 'YOM', 'Battery/Diesel', 'Relevancy',  
                'Seller Group', 'Buyer Group'
            ]

            for i in range(min(len(df_raw), len(df_clean))):
                r_row = df_raw.iloc[i]
                c_row = df_clean.iloc[i]
                
                desc_col = next((c for c in r_row.index if 'desc' in c.lower()), 'product_description')
                
                user_input = {
                    "desc": str(r_row.get(desc_col, "")),
                    "raw_unit": str(r_row.get('unit', "")),
                    "raw_qty": str(r_row.get('quantity', ""))
                }
                
                assistant_output = {k: v for k, v in c_row.items() if k in target_cols and pd.notna(v)}
                
                examples.append({
                    "user": json.dumps(user_input),
                    "assistant": json.dumps(assistant_output)
                })
    except Exception as e:
        print(f"Error loading refs: {e}")
        
    return examples


def deep_clean_payload(obj):
    """
    Recursively strips server-breaking control characters and unpaired surrogates 
    that cause OpenAI's HTTP 400 "We could not parse the JSON body" error.
    """
    if isinstance(obj, str):
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', obj)
        return clean.encode('utf-8', 'ignore').decode('utf-8')
    elif isinstance(obj, dict):
        return {k: deep_clean_payload(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean_payload(v) for v in obj]
    return obj


def refine_batch(df_batch: pd.DataFrame, product_group: str) -> pd.DataFrame:
    if df_batch.empty: return df_batch
    
    examples = load_dynamic_examples(product_group)
    
    # --- 1. INTELLIGENT CONTEXT: Filter Company & Model Maps ---
    search_text = " ".join(df_batch["product_description"].fillna("").astype(str).tolist()).upper()
    
    for col in ["seller", "buyer", "manufacturer", "Seller Group", "Buyer Group", "Manufacturer"]:
        found_col = next((c for c in df_batch.columns if c.lower() == col.lower()), None)
        if found_col:
            search_text += " " + " ".join(df_batch[found_col].fillna("").astype(str).tolist()).upper()

    relevant_company_rules = {}
    if COMPANY_DICTIONARY:
        sorted_keys = sorted(COMPANY_DICTIONARY.keys(), key=len, reverse=True)
        for variation in sorted_keys:
            clean_var = str(variation).upper().strip()
            if clean_var in search_text:
                relevant_company_rules[clean_var] = COMPANY_DICTIONARY[variation]
    
    company_map_str = json.dumps(relevant_company_rules) if relevant_company_rules else "{}"

    relevant_model_rules = {}
    if MODEL_DICTIONARY:
        sorted_model_keys = sorted(MODEL_DICTIONARY.keys(), key=len, reverse=True)
        for variation in sorted_model_keys:
            clean_var = str(variation).upper().strip()
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
            "   - **LIFT HEIGHT (CRITICAL FALLBACK)**: The automation sometimes fails to extract the height. If 'Current_Height (ft)' is 'MISSING', you MUST scan the Description and extract the working/platform height. Do NOT automatically extract other fields unless explicitly asked.\n"
            "   - **LIFT Specs**: Extract 'YOM' (Year) if available.\n"
            "   - **Type (Condition)**: Identify if 'New' or 'Used'. Everything is 'New' by default. ONLY mark as 'Used' if words like 'used', 'old', or 'refurbished' explicitly describe the machine's condition (e.g., 'used boom lift'). Ignore false positives like 'used for construction'.\n"
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
            "     2. **Type**: Extract Size (e.g., '0.5-2MM') or Purity (e.g., '98.5%'). CRITICAL: NEVER output 'Old', 'New', or 'Used' in this column for DS.\n"
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
        "Audit the 'Current_Values' (from automation) against the 'Description' and Raw fields. Fix errors & normalize data.\n\n"
        f"### COMPANY NORMALIZATION MAP (Use exact mappings):\n{company_map_str}\n\n"
        f"### MODEL NORMALIZATION MAP (Use exact mappings):\n{model_map_str}\n\n"
        "### CHECKLIST:\n"
        "1. **Company Names (Seller Group / Buyer Group)** (STRICT ENFORCEMENT):\n"
        "   - Evaluate the raw 'SELLER' and 'BUYER' fields against the COMPANY MAP above.\n"
        "   - IF a variation from the map is found in 'SELLER', set 'Seller Group' to the exact target value.\n"
        "   - IF a variation from the map is found in 'BUYER', set 'Buyer Group' to the exact target value.\n"
        "   - IF NO MATCH IS FOUND: You must clean and normalize the raw name yourself. Remove legal suffixes (e.g., 'LTD', 'INC', 'SRL', 'CO', 'PRIVATE LIMITED', 'LLC', 'GMBH'), remove city/country names, and strip punctuation.\n"
        "   - BATCH CONSISTENCY (CRITICAL): If you see slight variations of the same unmapped company across multiple rows (e.g., 'ROTOFLUX SRL' and 'ROTOFLUX ROTARY JOINTS'), you MUST group them under a single, clean, unified 'Seller Group' or 'Buyer Group' name (e.g., 'ROTOFLUX').\n"
        "2. **Model Extraction** (STRICT ENFORCEMENT):\n"
        "   - CHECK the Description against the MODEL MAP above.\n"
        "   - IF a source key from the MODEL MAP is found in the text, you MUST REPLACE 'Model' with the exact target value from the map.\n"
        "   - CRITICAL: Do NOT confuse Serial Numbers (e.g., '0300178158') with Models (e.g., '1350SJP' or 'SR220'). If a known model exists in the text, it overrides any extracted serial number.\n"
        "3. **Product Column**: DO NOT CHANGE. Leave as '{product_group}'.\n"
        "4. **Unit & Price (CRITICAL)**: \n"
        "   - If 'Current_Unit' is KGS/MTS but Description says count (e.g. '100 PCS'), FORCE Unit='NOS', Quantity=100.\n"
        "   - If Unit is 'NOS' but Qty=1 while Description says 'Box of 50', CHANGE Qty to 50.\n"
        "   - **ALWAYS** recalculate 'unit_price' if Quantity changes.\n"
        "5. **Core Attributes**: Verify 'Manufacturer', 'Spare / Unit / Others', and 'Type/Condition'. \n"
        "   - Scan Description to correct WRONG values.\n"
        "   - **CONDITION DEFAULT (CRITICAL)**: If determining 'New' vs 'Used'/'Old', ALWAYS default to 'New'. ONLY override to 'Used' or 'Old' if the description explicitly contains words like 'used', 'old', 'pre-owned', or 'refurbished' AND those words genuinely describe the physical condition of the item (e.g., correct automation false positives if it says 'used for lifting' or 'used in manufacturing'). Never leave it MISSING.\n"
        f"{specific_instructions}" 
        "6. **Relevancy (CRITICAL)**: Review the 'Current_Relevancy' value provided by our ML model. You MUST strongly trust it. If it says 'Relevant', output 'Relevant'. If it says 'Irrelevant', output 'Irrelevant'. ONLY change it if it is blatantly wrong based on the description. Do NOT leave this MISSING.\n\n"
        "Output JSON with a 'data' list of corrected rows. **CRITICAL: Ensure your JSON keys EXACTLY match the requested names** (e.g., strictly use 'Seller Group' and 'Buyer Group')."
    )

    messages = [{"role": "system", "content": system_content}]
    
    for ex in examples:
        messages.append({"role": "user", "content": f"Example In: {ex['user']}"})
        messages.append({"role": "assistant", "content": f"Example Out: {ex['assistant']}"})

    # --- 4. BUILD PAYLOAD ---
    batch_data = []
    
    cols_to_check = [
        "Manufacturer", "Model", "Category", "Type", "TYPE", "Application", "Product",
        "Seller_Type", "Buyer_Type", "Spare / Unit / Others", "Relevancy",
        "Height (ft)", "YOM", "Battery/Diesel", "SBU", "Imp Companies",
        "Seller Group", "Buyer Group"
    ]

    def sanitize_for_json(val):
        if pd.isna(val): return "MISSING"
        text = str(val).strip()
        if text.lower() in ["nan", "none", "nat", "<na>", ""]: return "MISSING"
        return text

    final_batch = df_batch.reset_index(drop=True).copy()

    seller_header = next((c for c in final_batch.columns if c.lower() == "seller"), None)
    buyer_header = next((c for c in final_batch.columns if c.lower() == "buyer"), None)

    for idx, row in final_batch.iterrows():
        item = {
            "id": str(idx),
            "Description": sanitize_for_json(row.get("product_description")),
            "Current_Unit": sanitize_for_json(row.get("unit")),
            "Current_Qty": sanitize_for_json(row.get("quantity")),
            "Current_Value_USD": sanitize_for_json(row.get("valueusd")),
        }
        
        for col in cols_to_check:
            found_col = next((c for c in final_batch.columns if c.lower() == col.lower()), None)
            if found_col:
                item[f"Current_{col}"] = sanitize_for_json(row.get(found_col))
            else:
                item[f"Current_{col}"] = "MISSING"

        item["SELLER"] = sanitize_for_json(row.get(seller_header)) if seller_header else "MISSING"
        item["BUYER"] = sanitize_for_json(row.get(buyer_header)) if buyer_header else "MISSING"

        batch_data.append(item)

    try:
        payload_str = json.dumps(batch_data, allow_nan=False)
    except (ValueError, TypeError) as e:
        print(f"JSON Serialization Warning: {e}. Falling back to aggressive string cast.")
        payload_str = str(batch_data)

    messages.append({"role": "user", "content": f"Audit Batch:\n{payload_str}"})
    messages = deep_clean_payload(messages)

    # --- 5. THE RETRY LOOP ---
    max_retries = 3
    result = {}
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages, 
                response_format={"type": "json_object"}, 
                temperature=0.0,
                max_tokens=16384 
            )
            
            if response.choices[0].finish_reason == "length":
                print("⚠️ WARNING: LLM output was cut off due to length!")
                
            result = json.loads(response.choices[0].message.content)
            break 
            
        except Exception as e:
            print(f"⚠️ OpenAI Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 15 + random.uniform(1, 10) 
                print(f"⏳ Thread sleeping for {sleep_time:.1f}s to bypass Rate Limit...")
                time.sleep(sleep_time) 
            else:
                print("❌ Max retries reached. Returning unrefined batch.")
                final_batch["Product"] = product_group
                final_batch["product"] = product_group 
                return final_batch

    data_list = []
    if "data" in result:
        data_list = result["data"]
    else:
        for key, val in result.items():
            if isinstance(val, list):
                data_list = val
                break
                
    clean_df = pd.DataFrame(data_list)
    
    # --- 6. THE ANTI-BLANK SHIELD ---
    # Expanded list of bad AI outputs to catch everything
    bad_outputs = ["MISSING", "NAN", "NULL", "N/A", "NONE", "UNKNOWN", "UNSPECIFIED", "-", ""]
    
    if len(clean_df) > 0 and "id" in clean_df.columns:
        for _, llm_row in clean_df.iterrows():
            try:
                row_id = int(llm_row["id"])
                if row_id in final_batch.index:
                    for col in clean_df.columns:
                        if col != "id":
                            match_col = next((c for c in final_batch.columns if c.lower() == col.lower()), col)
                            val = llm_row[col]
                            
                            if pd.notna(val):
                                clean_val = str(val).strip()
                                if clean_val and clean_val.upper() not in bad_outputs:
                                    final_batch.at[row_id, match_col] = val
                                    
            except (ValueError, TypeError):
                continue 

    # --- 7. EXPLICIT FALLBACK MECHANISM ---
    # If the LLM left things blank, and the manual automation also didn't catch it, 
    # we forcefully apply raw data fallbacks so the final CSV has no empty holes.
    
    for idx, row in final_batch.iterrows():
        # Helper to check if a cell is truly blank
        def is_blank(val):
            return pd.isna(val) or str(val).strip().upper() in bad_outputs

        # 1. Seller Group Fallback -> Inject Raw Seller Name
        seller_col = next((c for c in final_batch.columns if c.lower() == 'seller'), None)
        seller_group_col = next((c for c in final_batch.columns if c.lower() == 'seller group'), 'Seller Group')
        
        if is_blank(row.get(seller_group_col)) and seller_col and not is_blank(row.get(seller_col)):
            final_batch.at[idx, seller_group_col] = str(row[seller_col]).upper().strip()

        # 2. Buyer Group Fallback -> Inject Raw Buyer Name
        buyer_col = next((c for c in final_batch.columns if c.lower() == 'buyer'), None)
        buyer_group_col = next((c for c in final_batch.columns if c.lower() == 'buyer group'), 'Buyer Group')
        
        if is_blank(row.get(buyer_group_col)) and buyer_col and not is_blank(row.get(buyer_col)):
            final_batch.at[idx, buyer_group_col] = str(row[buyer_col]).upper().strip()
            
        # 3. Manufacturer Fallback -> Defaults to whatever is in Seller Group
        manu_col = next((c for c in final_batch.columns if c.lower() == 'manufacturer'), 'Manufacturer')
        if is_blank(row.get(manu_col)):
            final_batch.at[idx, manu_col] = final_batch.at[idx, seller_group_col]
            
        # 4. Spare / Unit / Others -> Default to "Others"
        suo_col = next((c for c in final_batch.columns if c.lower() == 'spare / unit / others'), 'Spare / Unit / Others')
        if is_blank(row.get(suo_col)):
            final_batch.at[idx, suo_col] = "Others"
            
        # 5. Relevancy -> Default to "Relevant"
        rel_col = next((c for c in final_batch.columns if c.lower() == 'relevancy'), 'Relevancy')
        if is_blank(row.get(rel_col)):
            final_batch.at[idx, rel_col] = "Relevant"

        # 6. LIFT Condition Default -> "New"
        if product_group.upper() == "LIFT":
            type_col = next((c for c in final_batch.columns if c.lower() == 'type'), 'Type')
            if is_blank(row.get(type_col)):
                final_batch.at[idx, type_col] = "New"

    # Force BOTH 'Product' and 'product' columns to match the uploaded product_group exactly.
    final_batch["Product"] = product_group
    final_batch["product"] = product_group 

    return final_batch