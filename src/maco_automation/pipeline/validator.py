"""
====================================================================
MACO AUTOMATION – STEP 5: VALIDATION & BUSINESS RULE CHECKS
====================================================================
Performs both field integrity checks and business-rule validations
after classification.

Checks:
    1. Missing critical fields
    2. Duplicate shipment IDs
    3. Invalid numeric values
    4. Missing relevancy predictions
    5. Invalid HS codes
    6. Value mismatch (quantity * unit_price ~ valueusd)
    7. Low-price anomaly detection
    8. Shipment type inference (Spare / Unit / Mixed / Unknown)

Outputs:
    - Returns (validated_df, audit_issues_df)
    - Saves detailed audit report CSV in /output/audit/
    - Saves summary report CSV (issue counts, top categories)
====================================================================
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
CRITICAL_FIELDS = [
    "product_description",
    "buyer_country",
    "unit_price",
    "quantity",
    "hs_code",
]

AUDIT_DIR = os.path.join("output", "audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

VALUE_TOLERANCE = 0.10     # 10% difference allowed for value mismatch
LOW_PRICE_THRESHOLD = 50   # Flag if unit_price < 50 USD


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_float(val):
    """Convert to float, handling symbols and commas."""
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except Exception:
        return np.nan


def detect_shipment_type(desc: str) -> str:
    """Infer shipment type based on keywords."""
    if not isinstance(desc, str):
        return "Unknown"
    text = desc.lower()

    spare_keywords = ["seal kit", "bearing", "spare", "gasket", "o-ring", "connector", "bolt", "kit"]
    unit_keywords = ["rotary union", "rotary joint", "slip ring", "machine", "equipment", "assembly"]
    mixed_keywords = ["including", "with", "plus", "along with"]

    if any(k in text for k in mixed_keywords):
        return "Mixed"
    if any(k in text for k in spare_keywords):
        return "Spare"
    if any(k in text for k in unit_keywords):
        return "Unit"
    return "Unknown"


# ---------------------------------------------------------------------
# MAIN VALIDATION FUNCTION
# ---------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame):
    """Perform all validation checks and create audit logs."""

    issues = []
    print("\n[5] Running data validation and business checks...")

    # --------------------------------------------------
    # 1. Missing critical fields
    # --------------------------------------------------
    for col in CRITICAL_FIELDS:
        if col not in df.columns:
            continue
        missing_rows = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        for _, row in missing_rows.iterrows():
            issues.append({
                "shipment_id": row.get("shipment_id", "N/A"),
                "issue_type": "Missing Value",
                "column": col,
                "details": "Empty or invalid value"
            })

    # --------------------------------------------------
    # 2. Duplicate shipment IDs
    # --------------------------------------------------
    if "shipment_id" in df.columns:
        dupes = df[df["shipment_id"].duplicated(keep=False)]
        for sid in dupes["shipment_id"].unique():
            issues.append({
                "shipment_id": sid,
                "issue_type": "Duplicate",
                "column": "shipment_id",
                "details": "Appears multiple times"
            })

    # --------------------------------------------------
    # 3. Invalid numeric values
    # --------------------------------------------------
    for num_col in ["unit_price", "quantity"]:
        if num_col not in df.columns:
            continue
        invalid = df[~df[num_col].astype(str)
                     .str.replace(r"[^0-9.\-]", "", regex=True)
                     .str.match(r"^\d+(\.\d+)?$")]
        for _, row in invalid.iterrows():
            issues.append({
                "shipment_id": row.get("shipment_id", "N/A"),
                "issue_type": "Invalid Format",
                "column": num_col,
                "details": f"Non-numeric or malformed value '{row[num_col]}'"
            })

    # Convert numeric fields
    for col in ["unit_price", "quantity", "valueusd", "value(usd)"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    # --------------------------------------------------
    # 4. Missing relevancy predictions
    # --------------------------------------------------
    # --- UPDATED COLUMN NAME ---
    if "Relevancy" in df.columns:
        missing_rel = df[df["Relevancy"].isna()]
        for _, row in missing_rel.iterrows():
            issues.append({
                "shipment_id": row.get("shipment_id", "N/A"),
                "issue_type": "Missing Prediction",
                "column": "Relevancy",
                "details": "Relevancy prediction not available"
            })
    # ---------------------------

    # --------------------------------------------------
    # 5. Invalid HS code format
    # --------------------------------------------------
    if "hs_code" in df.columns:
        bad_hs = df[~df["hs_code"].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.isdigit()]
        for _, row in bad_hs.iterrows():
            issues.append({
                "shipment_id": row.get("shipment_id", "N/A"),
                "issue_type": "Invalid Format",
                "column": "hs_code",
                "details": f"HS code contains invalid characters: {row['hs_code']}"
            })

        short_hs = df[df["hs_code"].astype(str).str.len() < 6]
        for _, row in short_hs.iterrows():
            issues.append({
                "shipment_id": row.get("shipment_id", "N/A"),
                "issue_type": "Invalid Length",
                "column": "hs_code",
                "details": f"HS code too short: {row['hs_code']}"
            })

    # --------------------------------------------------
    # 6. Value mismatch check
    # --------------------------------------------------
    if all(col in df.columns for col in ["quantity", "unit_price"]):
        value_col = "valueusd" if "valueusd" in df.columns else "value(usd)" if "value(usd)" in df.columns else None
        if value_col:
            df["calc_value"] = df["quantity"] * df["unit_price"]
            df["value_diff_pct"] = np.abs(df["calc_value"] - df[value_col]) / df[value_col]
            mismatched = df[df["value_diff_pct"] > VALUE_TOLERANCE]
            for _, row in mismatched.iterrows():
                issues.append({
                    "shipment_id": row.get("shipment_id", "N/A"),
                    "issue_type": "Value Mismatch",
                    "column": "valueusd",
                    "details": f"Calculated vs reported value differs by >{VALUE_TOLERANCE*100:.0f}%"
                })

    # --------------------------------------------------
    # 7. Low price anomaly
    # --------------------------------------------------
    if "unit_price" in df.columns:
        low_price = df[df["unit_price"] < LOW_PRICE_THRESHOLD]
        for _, row in low_price.iterrows():
            issues.append({
                "shipment_id": row.get("shipment_id", "N/A"),
                "issue_type": "Low Price",
                "column": "unit_price",
                "details": f"Suspiciously low unit price ({row['unit_price']})"
            })

    # --------------------------------------------------
    # 8. Shipment Type Inference
    # --------------------------------------------------
    if "product_description" in df.columns:
        df["shipment_type"] = df["product_description"].apply(detect_shipment_type)

    # --------------------------------------------------
    # SAVE AUDIT REPORT
    # --------------------------------------------------
    issues_df = pd.DataFrame(issues)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_file = f"audit_report_{timestamp}.csv"
    summary_file = f"audit_summary_{timestamp}.csv"
    audit_path = os.path.join(AUDIT_DIR, audit_file)
    summary_path = os.path.join(AUDIT_DIR, summary_file)

    if not issues_df.empty:
        issues_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
        summary = issues_df["issue_type"].value_counts().reset_index()
        summary.columns = ["issue_type", "count"]
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"Validation complete with {len(issues_df)} issues")
        print(f"   Detailed report: {audit_path}")
        print(f"   Summary report:  {summary_path}")
    else:
        print("No validation issues found.")

    return df, issues_df