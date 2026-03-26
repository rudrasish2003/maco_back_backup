# src/maco_automation/pipeline/preprocess.py
import re
import json
import chardet
from ftfy import fix_text
from unidecode import unidecode
from dateutil import parser as dateparser
import pandas as pd
from rich import print

# ----------------------------
# Utilities
# ----------------------------
def detect_encoding(path, nbytes=200000):
    """Detect encoding using chardet on a sample of bytes."""
    with open(path, "rb") as f:
        raw = f.read(nbytes)
    res = chardet.detect(raw)
    enc = res.get("encoding") or "latin1"
    print(f"detect_encoding: detected => {enc} (confidence {res.get('confidence')})")
    return enc


def normalize_colnames(df):
    """Convert column names to snake_case and strip junk characters."""
    
    # Map against space-stripped keys for extreme resilience
    COLUMN_MAP = {
        "VALUE(USD)": "valueusd",
        "UNITPRICE": "unit_price",
        "QTY": "quantity",
        "QUANTITY": "quantity",
        "PRODUCTDESCRIPTION": "product_description",
        "HSCODE": "hs_code",
        "SHIPMENTID": "shipment_id",
        "BUYERCOUNTRY": "buyer_country",
        "DESTINATIONPORT": "destination_port",
        "SELLERCOUNTRY": "seller_country",
        "ORIGINPORT": "origin_port",
        "SELLER": "seller",
        "BUYER": "buyer",
        "INDUSTRY": "industry",
        "DATE": "date",
        "UNIT": "unit",
    }
    
    def _norm(c):
        c_str = str(c).strip()
        c_nospace = c_str.upper().replace(" ", "")
        
        # Check resilient map first
        if c_nospace in COLUMN_MAP:
            return COLUMN_MAP[c_nospace]
        
        # Standard snake_case conversion fallback
        c_str = re.sub(r"\s+", "_", c_str)
        c_str = re.sub(r"[^\w_]", "", c_str)
        return c_str.lower()
        
    df = df.rename(columns=lambda c: _norm(c))
    return df

def clean_text(val):
    """Clean text fields: fix encoding, strip, normalize spaces."""
    if pd.isna(val):
        return val
    s = str(val)
    s = fix_text(s)                 # fix mojibake / encoding issues
    s = unidecode(s)                # normalize accents
    s = s.replace("\\", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_currency(val):
    """Parse currency-like strings into floats."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r"[^\d\.-]", "", s)
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_int_like(val):
    """Parse int-like values safely handling Excel floats."""
    if pd.isna(val):
        return None
    try:
        # This safely handles string floats like "10.0" -> 10.0 -> 10
        return int(float(str(val).strip()))
    except Exception:
        # Fallback for messy text like "10 pcs"
        s = re.sub(r"[^\d\-]", "", str(val))
        try:
            return int(s)
        except Exception:
            return None

def parse_date(val, dayfirst=False):
    """Parse dates using dateutil and format as YYYY-MM-DD."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        dt = dateparser.parse(str(val), dayfirst=dayfirst)
        return dt.strftime('%Y-%m-%d') # Formats cleanly to yyyy-mm-dd
    except Exception:
        return None


# ----------------------------
# Core preprocess function
# ----------------------------
def preprocess_csv(path, sample_n=5, encoding=None, dayfirst=False, drop_unnamed=True):
    """Load and clean a raw file (CSV or Excel) with basic normalization."""

    path_str = str(path).lower()

    # 1. Handle Excel Files
  # 1. Handle Excel Files
    if path_str.endswith(('.xlsx', '.xls')):
        try:
            print(f"Reading as Excel: {path}")
            df = pd.read_excel(path)
        except Exception as e:
            print(f"Failed to read as Excel: {e}. Falling back to CSV reader...")
            try:
                # Fallback in case it's a CSV with a bad .xlsx extension
                enc = encoding or detect_encoding(path)
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            except Exception as e2:
                raise ValueError(f"Failed to read file as both Excel and CSV: {e2}")

    # 2. Handle CSV Files
    else:
        # Detect encoding if not supplied
        enc = encoding or detect_encoding(path)

        # Try reading with fallback encodings
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            print(f"Failed with detected encoding '{enc}'. Retrying with latin1...")
            df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        except Exception as e:
            print(f"Failed reading CSV with {enc}: {e}")
            print("Trying fallback encodings utf-8-sig → cp1252 → latin1")
            for fallback in ["utf-8-sig", "cp1252", "latin1"]:
                try:
                    df = pd.read_csv(path, encoding=fallback, on_bad_lines="skip")
                    print(f"Fallback successful with {fallback}")
                    break
                except Exception:
                    continue
    
    if 'df' not in locals():
        raise Exception("All file reading attempts failed.")

    print(f"File loaded: rows={len(df)} cols={len(df.columns)}")

    # 3. Normalize column names
    df = normalize_colnames(df)

    # 4. Drop unnamed junk columns
    if drop_unnamed:
        unnamed = [c for c in df.columns if c.startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
            print(f"Dropped columns: {unnamed}")

    # 5. Clean all text columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        # Skip numeric columns that may have been read as object type
        if c not in ['valueusd', 'unit_price', 'quantity']:
            df[c] = df[c].apply(clean_text)

    # 6. Attempt numeric parsing for likely numeric columns
    # Using the consistently named columns thanks to normalize_colnames
    numeric_name_candidates = [
        "value", "value_usd", "valueusd",
        "unit_price", 
        "qty", "quantity",
        "hs_code", "shipment_id"
    ]
    for candidate in numeric_name_candidates:
        if candidate in df.columns:
            if "value" in candidate or "price" in candidate:
                df[candidate] = df[candidate].apply(parse_currency)
            elif "qty" in candidate or "quantity" in candidate:
                df[candidate] = df[candidate].apply(parse_int_like)
            # hs_code and shipment_id are often better as strings
            elif candidate not in ["hs_code", "shipment_id"]:
                 df[candidate] = df[candidate].apply(parse_int_like)


    # 7. Parse date columns (heuristic)
    date_cols = [c for c in df.columns if "date" in c]
    for c in date_cols:
        df[c] = df[c].apply(lambda v: parse_date(v, dayfirst=dayfirst))

    # 8. Derived consistency checks
    if all(col in df.columns for col in ["quantity", "unit_price", "valueusd"]):
        df["_derived_value_calc"] = df["quantity"].astype(float) * df["unit_price"].astype(float)
        df["_value_mismatch"] = (
            (~df["_derived_value_calc"].isna()) &
            (~df["valueusd"].isna()) &
            (abs(df["_derived_value_calc"] - df["valueusd"]) > 1.0)
        )
    else:
        df["_value_mismatch"] = False

    # 9. Return full DF and sample
    sample = df.head(sample_n)
    return df, sample