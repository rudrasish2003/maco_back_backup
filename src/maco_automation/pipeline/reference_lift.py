from .reference_dicts import load_from_db
from .reference_dicts import (
    # Import the raw containers populated by reference_dicts.py
    LIFT_FUEL_MAP,
    LIFT_CATEGORY_MAP,
    # Standard dicts
    COMPANY_DICTIONARY,
    MODEL_DICTIONARY,
    SPARE_KEYWORDS,
    UNIT_KEYWORDS,
    MANUFACTURER_KEYWORDS,
    MODEL_PATTERNS,
    PRODUCT_KEYWORDS
)

# 1. Load Data from DB
# This fetches "lift_fuel_map" and "lift_category_map" entries into the imported dicts
print("🔵 Initializing LIFT Context (Inheriting Rotary Patterns)...")
load_from_db(product_groups=["ROTARY_UNION", "LIFT"])

# 2. Transform DB Data for Extractor
# The DB stores flat pairs: {"ELECTRIC": "Battery", "VRLA": "Battery"}
# The Extractor needs groups: {"Battery": ["ELECTRIC", "VRLA"]}

FUEL_KEYWORDS = {}
for keyword, target_type in LIFT_FUEL_MAP.items():
    if target_type not in FUEL_KEYWORDS:
        FUEL_KEYWORDS[target_type] = []
    FUEL_KEYWORDS[target_type].append(keyword)

CATEGORY_KEYWORDS = {}
for keyword, category in LIFT_CATEGORY_MAP.items():
    if category not in CATEGORY_KEYWORDS:
        CATEGORY_KEYWORDS[category] = []
    CATEGORY_KEYWORDS[category].append(keyword)

# Fallback/Debug: If DB was empty, prevent crashes
if not FUEL_KEYWORDS:
    print("⚠️ Warning: No Lift Fuel Map found in DB. Using Defaults.")
    FUEL_KEYWORDS = {
        "Diesel": ["DIESEL", "ENGINE POWERED", "4WD", "GAS"],
        "Battery": ["ELECTRIC", "BATTERY", "DC ", "LITHIUM", "VRLA"]
    }

if not CATEGORY_KEYWORDS:
    print("⚠️ Warning: No Lift Category Map found in DB. Using Defaults.")
    # UPDATED DEFAULTS based on your request
    CATEGORY_KEYWORDS = {
        # Specific Types First (Priority)
        "SCISSOR LIFT": ["SCISSOR"],
        "SPIDER LIFT": ["SPIDER"],
        "ARTICULATING BOOM": ["ARTICULATING", "ARTICULATED", "KNUCKLE"],
        "TELESCOPIC BOOM": ["TELESCOPIC", "STRAIGHT BOOM"],
        
        # Generic / Fallback Types Last
        "BOOM OTHERS": ["BOOM"],  # Matches "Boom" if Articulating/Telescopic weren't found
        "AWP": ["AWP", "AERIAL WORK PLATFORM"]
    }