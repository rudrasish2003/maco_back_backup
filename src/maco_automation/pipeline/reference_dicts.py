import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv() 

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise ValueError("MONGO_URI not found")

DB_NAME = "maco_db"
COLLECTION_NAME = "product_dictionaries"

# --- GLOBAL VARIABLES (The "10+" Standard + Specifics) ---

# 1-7 (Standard Logic)
COMPANY_DICTIONARY = {}
MODEL_DICTIONARY = {}
SPARE_KEYWORDS = []
UNIT_KEYWORDS = []
MANUFACTURER_KEYWORDS = []
MODEL_PATTERNS = []
PATTERN_HINTS = {}

# 8-11 (Additional Standard Logic)
OTHER_MACHINE_KEYWORDS = []
CONDITION_KEYWORDS = {} # Loaded as Map: {keyword: target}
APPLICATION_MAP = {}
UNIT_MAP = {}

# Product Specific (Extras)
LIFT_FUEL_MAP = {}
LIFT_CATEGORY_MAP = {}
SHOT_MANUFACTURER_LIST = []
SHOT_RESELLER_LIST = []
DS_CATEGORY_MAP = {}

PRODUCT_KEYWORDS = {"relevant": [], "irrelevant": []}

def load_from_db(product_groups=None):
    global COMPANY_DICTIONARY, MODEL_DICTIONARY, SPARE_KEYWORDS, UNIT_KEYWORDS
    global MANUFACTURER_KEYWORDS, MODEL_PATTERNS, PATTERN_HINTS, PRODUCT_KEYWORDS
    global OTHER_MACHINE_KEYWORDS, CONDITION_KEYWORDS, APPLICATION_MAP, UNIT_MAP
    global LIFT_FUEL_MAP, LIFT_CATEGORY_MAP, SHOT_MANUFACTURER_LIST, SHOT_RESELLER_LIST, DS_CATEGORY_MAP
    
    if product_groups is None: product_groups = ["ROTARY_UNION"]
        
    print(f"🔌 Fetching Independent Rules for: {product_groups}...")
    
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # This will fetch the specific copy for the requested product group(s)
        cursor = collection.find({"product_group": {"$in": product_groups}})
        data = list(cursor)
        
        if not data:
            print("⚠️ Warning: No dictionary data found.")
            return

        # Clear existing
        COMPANY_DICTIONARY.clear(); MODEL_DICTIONARY.clear(); SPARE_KEYWORDS.clear()
        UNIT_KEYWORDS.clear(); MANUFACTURER_KEYWORDS.clear(); MODEL_PATTERNS.clear()
        PATTERN_HINTS.clear(); OTHER_MACHINE_KEYWORDS.clear(); CONDITION_KEYWORDS.clear()
        APPLICATION_MAP.clear(); UNIT_MAP.clear()
        PRODUCT_KEYWORDS = {"relevant": [], "irrelevant": []}
        LIFT_FUEL_MAP.clear(); LIFT_CATEGORY_MAP.clear(); 
        SHOT_MANUFACTURER_LIST.clear(); SHOT_RESELLER_LIST.clear(); DS_CATEGORY_MAP.clear()

        # Populate
        for doc in data:
            dtype = doc.get("dictionary_type")
            key = doc.get("source_key")
            val = doc.get("target_value")

            # Skip empty keys
            if not key:
                continue

            # --- STANDARD 10+ ---
            if dtype == "company_map": 
                # FIX: Force uppercase and remove hidden spaces from DB entries
                clean_key = str(key).upper().strip()
                COMPANY_DICTIONARY[clean_key] = val
                
            elif dtype == "model_map": 
                clean_key = str(key).upper().strip()
                MODEL_DICTIONARY[clean_key] = val
                
            elif dtype == "classification_keywords":
                clean_key = str(key).upper().strip()
                if val == "Spare": SPARE_KEYWORDS.append(clean_key)
                elif val == "Unit": UNIT_KEYWORDS.append(clean_key)
                
            elif dtype == "manufacturer_keywords": MANUFACTURER_KEYWORDS.append(str(key).upper().strip())
            elif dtype == "model_patterns": MODEL_PATTERNS.append(key) # Keep original for Regex
            elif dtype == "pattern_hints": PATTERN_HINTS[str(key).upper().strip()] = val
            elif dtype == "other_machine_keywords": OTHER_MACHINE_KEYWORDS.append(str(key).upper().strip())
            elif dtype == "condition_map": CONDITION_KEYWORDS[str(key).upper().strip()] = val
            elif dtype == "application_map": APPLICATION_MAP[str(key).upper().strip()] = val
            elif dtype == "unit_map": UNIT_MAP[str(key).lower().strip()] = val # Units are usually lowercase in normalize.py
            
            # --- LIFT ---
            elif dtype == "lift_fuel_map": LIFT_FUEL_MAP[str(key).upper().strip()] = val
            elif dtype == "lift_category_map": LIFT_CATEGORY_MAP[str(key).upper().strip()] = val
            
            # --- STEEL SHOT ---
            elif dtype == "shot_manufacturer_list": SHOT_MANUFACTURER_LIST.append(str(key).upper().strip())
            elif dtype == "shot_reseller_list": SHOT_RESELLER_LIST.append(str(key).upper().strip())
            
            # --- DS ---
            elif dtype == "ds_category_map": DS_CATEGORY_MAP[str(key).upper().strip()] = val

        print(f"✅ Loaded {len(data)} rules.")
        client.close()

    except Exception as e:
        print(f"❌ Error loading dictionaries: {e}")