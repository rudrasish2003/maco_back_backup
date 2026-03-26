import pandas as pd
import re
import logging

def standardize_units(df):
    """
    Standardizes 'unit' to 'NOS' ONLY IF a proper conversion can be found.
    Leaves units like 'KGS' intact if no discrete item count is found in the description.
    
    NOTE: Expects standardized snake_case column names from preprocess.py:
          - unit
          - quantity
          - valueusd
          - product_description
    """
    logging.info("Starting Unit Standardization...")
    
    # 1. Define Uniform Mapping (Target: NOS)
    unit_map = {
        'PCS': 'NOS',
        'PIECE': 'NOS',
        'PIECES': 'NOS',
        'SET': 'NOS',
        'SETS': 'NOS',
        'UNT': 'NOS',
        'UNIT': 'NOS',
        'NO': 'NOS',
        'PACK': 'NOS',
        'BOX': 'NOS',
        'NUM': 'NOS',
        'DOZ': 'NOS'
    }

    # 2. Regex Patterns
    # Matches: "QTY: 10", "10 PCS", "10 NOS", "10 X 500ML"
    qty_patterns = [
        r'QTY\s*[:\-\.]?\s*(\d+)',       
        r'QUANTITY\s*[:\-\.]?\s*(\d+)',  
        r'(\d+)\s*(?:NOS|PCS|SETS?|UNT|PIECES?)', 
    ]

    def process_row(row):
        # Normalize inputs (using standardized snake_case names)
        current_unit = str(row.get('unit', '')).upper().strip()
        current_qty = pd.to_numeric(row.get('quantity', 0), errors='coerce')
        current_val = pd.to_numeric(row.get('valueusd', 0), errors='coerce')
        desc = str(row.get('product_description', '')).upper()

        new_unit = current_unit
        new_qty = current_qty
        new_unit_price = row.get('unit_price', 0) # Default to existing price

        # Logic 1: Direct Mapping for discrete units
        if current_unit in unit_map:
            new_unit = 'NOS'
        
        # Logic 2: Conditional Conversion (KGS -> NOS) ONLY if explicit count is found
        elif current_unit in ['KGS', 'KG', 'MTR', 'METER', 'LITERS'] or pd.isna(current_unit) or current_unit == 'NAN':
            extracted_qty = None
            for pattern in qty_patterns:
                match = re.search(pattern, desc)
                if match:
                    try:
                        found_qty = float(match.group(1))
                        if found_qty > 0:
                            extracted_qty = found_qty
                            break 
                    except ValueError:
                        continue
            
            # ONLY update if a proper conversion was found in the description
            if extracted_qty:
                new_qty = extracted_qty
                new_unit = 'NOS'
                
                # Logic 3: Price Adjustment
                # Recalculate price ONLY if we successfully extracted a new NOS count
                if new_qty > 0 and pd.notna(current_val) and current_val > 0:
                    new_unit_price = current_val / new_qty
            
            # [UPDATED LOGIC]: The fallback block has been removed. 
            # If no count is extracted, the unit remains KGS/MTR, and original qty/price are kept.

        # Return named series to ensure safe assignment
        return pd.Series(
            [new_unit, new_qty, new_unit_price], 
            index=['unit', 'quantity', 'unit_price']
        )

    # Check for required columns before applying (using snake_case)
    req_cols = ['unit', 'quantity', 'valueusd', 'product_description']
    missing_cols = [col for col in req_cols if col not in df.columns]
    
    if missing_cols:
        logging.warning(f"⚠️ Missing columns for unit conversion: {missing_cols}. Skipping step.")
        return df

    # Apply Logic
    result = df.apply(process_row, axis=1)
    df[['unit', 'quantity', 'unit_price']] = result
    
    return df