# src/maco_automation/api/main.py

import os
import shutil
import pandas as pd
import numpy as np
import json
import io
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import pdfplumber
import gridfs
import tempfile
import concurrent.futures

# --- FASTAPI IMPORTS ---
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Body, Depends, status,BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# --- SECURITY IMPORTS ---
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- MONGODB IMPORTS ---
from pydantic import BaseModel, Field
from pymongo import MongoClient, errors
import certifi

# Import pipeline functions
from maco_automation.pipeline.run import run_pipeline
from maco_automation.pipeline.preprocess import preprocess_csv
from maco_automation.pipeline import reference_dicts
from bson.objectid import ObjectId
# ⬇️ NEW IMPORT: Standard Dictionaries for Cloning ⬇️
from maco_automation.pipeline.reference_dicts import (
    COMPANY_DICTIONARY, MODEL_DICTIONARY, PATTERN_HINTS, 
    SPARE_KEYWORDS, UNIT_KEYWORDS, MANUFACTURER_KEYWORDS, 
    MODEL_PATTERNS, OTHER_MACHINE_KEYWORDS, CONDITION_KEYWORDS,
    APPLICATION_MAP, UNIT_MAP
)
import bcrypt
import time

# passlib relies on an attribute '__about__' that was removed in bcrypt 4.0
# We inject a dummy object so passlib finds what it looks for.
if not hasattr(bcrypt, "__about__"):
    class MockAbout:
        __version__ = bcrypt.__version__
    bcrypt.__about__ = MockAbout()
import asyncio
import uuid
from maco_automation.pipeline.llm_refiner import refine_batch
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "api" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SECURITY CONFIGURATION ---
# ⚠️ IMPORTANT: Change this secret key for production!
SECRET_KEY = os.environ.get("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- MONGODB ATLAS CONFIGURATION ---
MONGO_URI = " "

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")
DB_NAME = "maco_db"

# Collection Names
COLLECTION_DICTS = "product_dictionaries"
COLLECTION_HISTORY = "upload_history"
COLLECTION_INPUT = "input_data_store"
COLLECTION_PROCESSED = "processed_data_store"
COLLECTION_USERS = "users" 
COLLECTION_LEGAL_ENTITIES = "legal_entities"
COLLECTION_FINANCIAL = "financial_data"

# --- DATABASE CONNECTION ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    
    collection_dicts = db[COLLECTION_DICTS]
    collection_history = db[COLLECTION_HISTORY]
    collection_input = db[COLLECTION_INPUT]
    collection_processed = db[COLLECTION_PROCESSED]
    collection_users = db[COLLECTION_USERS]
    collection_legal_entities = db[COLLECTION_LEGAL_ENTITIES]
    collection_financial = db[COLLECTION_FINANCIAL]
    
    # Create indexes
    collection_input.create_index("upload_id")
    collection_processed.create_index("upload_id")
    collection_users.create_index("username", unique=True) # Ensure unique usernames
    collection_legal_entities.create_index("tax_id", unique=True) 
    collection_financial.create_index([("tax_id", 1), ("financial_year", 1)])
    
    print(f"✅ Connected to MongoDB Atlas!")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to MongoDB Atlas: {e}")
    fs=None
    collection_history = None
    collection_input = None
    collection_processed = None
    collection_dicts = None
    collection_users = None
    collection_legal_entities = None
    collection_financial = None

JOB_STORE = {}
# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class SaveRequest(BaseModel):
    status: str  # "working" or "complete"
    data: List[dict]

# --- Auth Models ---
class UserPermissions(BaseModel):
    """Permissions settings for a user"""
    allowed_products: List[str] = Field(default_factory=list, description="List of product groups user can view")
    can_view_finance: bool = False
    can_view_crm: bool = False

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"  # 'admin' or 'user'
    permissions: UserPermissions = Field(default_factory=UserPermissions)

class UserUpdate(BaseModel):
    """Model for updating an existing user (Optional fields)"""
    role: Optional[str] = None
    permissions: Optional[UserPermissions] = None
    password: Optional[str] = None # Allow admin to reset password if needed

class UserResponse(BaseModel):
    username: str
    role: str
    permissions: UserPermissions
    
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
#---------------------------------------------------------
#Financial Data Models
#---------------------------------------------------------
class LegalEntityCreate(BaseModel):
    entity_name: str
    tax_id: str
    mapped_companies: List[str] = []

class LegalEntityUpdate(BaseModel):
    entity_name: Optional[str] = None
    mapped_companies: Optional[List[str]] = None

class LegalEntityResponse(BaseModel):
    id: str
    entity_name: str
    tax_id: str
    mapped_companies: List[str]
    created_at: Optional[datetime] = None

class FinancialRecordResponse(BaseModel):
    upload_id: str
    tax_id: str
    entity_name: str
    financial_year: str
    file_type: str
    filename: str
    uploaded_at: datetime
    data_preview: List[dict]
# ---------------------------------------------------------
# Helper: Sanitization for MongoDB
# ---------------------------------------------------------

def sanitize_for_json(df: pd.DataFrame) -> List[dict]:
    """
    Replaces NaN, Infinity, and -Infinity with None (JSON null).
    """
    if df.empty:
        return []
    
    # Create a copy to avoid modifying the original if needed
    df_clean = df.copy()
    
    # Replace Infinity
    df_clean.replace([np.inf, -np.inf], None, inplace=True)
    
    # Replace NaN with None
    # object conversion is often needed to allow None in float columns
    df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)
    
    return df_clean.to_dict(orient="records")

def sanitize_dataframe_for_mongo(df: pd.DataFrame) -> List[Dict]:  # <--- Changed list[dict] to List[Dict]
    """
    Converts a Pandas DataFrame to a list of dicts safe for MongoDB insertion.
    """
    if df.empty:
        return []

    # 1. Clean Column Names (No dots allowed in Mongo keys)
    # Note: We create a copy of columns to avoid modifying the original DF in-place if possible, 
    # but assigning to df.columns modifies the index. 
    df.columns = [str(c).replace(".", "_") for c in df.columns]

    # 2. Convert to Object type (crucial for replacing floats/dates with None)
    df_clean = df.astype(object)

    # 3. Replace NaT (Time), NaN (Float), Inf with None
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    # 4. Handle any lingering Infinity values
    df_clean.replace([np.inf, -np.inf], None, inplace=True)

    return df_clean.to_dict("records")

# ---------------------------------------------------------
# Security Utilities
# ---------------------------------------------------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    if collection_users is None:
        raise HTTPException(status_code=500, detail="DB Error")

    user = collection_users.find_one({"username": token_data.username})
    if user is None:
        raise credentials_exception
    return user

def get_current_admin(current_user: dict = Depends(get_current_user)):
    """
    Dependency to restrict route to Admins only.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Not authorized (Admin privileges required)"
        )
    return current_user

# --- New Permission Dependencies ---

def check_finance_access(current_user: dict = Depends(get_current_user)):
    """Ensure user has finance viewing rights"""
    perms = current_user.get("permissions", {})
    if current_user.get("role") != "admin" and not perms.get("can_view_finance", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Requires Finance permissions")
    return current_user

def check_crm_access(current_user: dict = Depends(get_current_user)):
    """Ensure user has CRM viewing rights"""
    perms = current_user.get("permissions", {})
    if current_user.get("role") != "admin" and not perms.get("can_view_crm", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Requires CRM permissions")
    return current_user

def get_allowed_products_filter(current_user: dict):
    """
    Returns a MongoDB query filter for product_group based on user permissions.
    Admins see all. Users see only their allowed_products.
    """
    if current_user.get("role", "").lower() == "admin":
        return {} # No filter, see all
    
    allowed = current_user.get("permissions", {}).get("allowed_products", [])
    if not allowed:
        # If list is empty, user sees nothing
        # We return a filter that matches nothing
        return {"product_group": "__NO_ACCESS__"} 
    
    return {"product_group": {"$in": allowed}}


#---------------------------------------------------------
#Helper function for finacial data upload
#---------------------------------------------------------
# ---------------------------------------------------------
# Helper function for financial data upload
# ---------------------------------------------------------
def parse_financial_file(file_bytes: bytes, filename: str) -> List[dict]:
    """Parses Excel/CSV/PDF financial files with 'Metric vs Year' structure."""
    try:
        df = None
        filename_lower = filename.lower()

        # 1. Load File into DataFrame based on extension
        if filename_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        
        elif filename_lower.endswith(".pdf"):
            # PDF Table Extraction using pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                all_rows = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        # table is a list of lists e.g. [ ["Metric", "2021"], ["Rev", "100"] ]
                        if table:
                            all_rows.extend(table)
                
                if all_rows:
                    df = pd.DataFrame(all_rows)
        
        else:
            # Fallback (CSV)
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
            except:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")

        if df is None or df.empty:
            return []

        # 2. Dynamic Header Detection (Look for Year or 'March')
        header_idx = -1
        # Convert to string to avoid type errors during search
        df_str = df.astype(str)
        
        for idx, row in df_str.iterrows():
            row_vals = row.str.lower().values
            # Heuristic: finding a year "202x" or month "march" often indicates the header row
            if any("202" in s for s in row_vals) or any("march" in s for s in row_vals):
                header_idx = idx
                break
        
        if header_idx != -1:
            df.columns = df.iloc[header_idx]
            df = df.iloc[header_idx + 1:].reset_index(drop=True)

        parsed_data = []
        
        # 3. Identify Year Columns (Simple heuristic: Contains digits)
        # We assume columns like "FY 2021", "2022", "Mar-23" contain digits
        year_cols = [c for c in df.columns if isinstance(c, str) and any(char.isdigit() for char in c)]
        
        # Fallback if no digits found in headers (unlikely but possible)
        if not year_cols: 
            year_cols = df.columns[1:]

        metric_col = df.columns[0]
        
        # 4. Extract Data Points
        for _, row in df.iterrows():
            metric = str(row[metric_col]).strip()
            if not metric or metric.lower() in ['nan', 'none', '', 'metric', 'particulars']: 
                continue
            
            for year in year_cols:
                val = row[year]
                if pd.notnull(val) and str(val).strip() != "":
                    # Clean currency/formatting characters
                    clean_val = str(val).replace(",", "").replace("(", "-").replace(")", "")
                    try: 
                        clean_val = float(clean_val)
                    except: 
                        pass # Keep as string if it's text (e.g. "Audited")
                    
                    parsed_data.append({
                        "metric": metric, 
                        "period": str(year).strip(), 
                        "value": clean_val
                    })
                    
        return parsed_data

    except Exception as e:
        print(f"Parser Error: {e}")
        return []

# ---------------------------------------------------------
# App Definition
# ---------------------------------------------------------
app = FastAPI(
    title="MACO Data Automation API",
    description="Automated pipeline for rotary union and mechanical product data cleaning & classification",
    version="1.7.0",
    root_path="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Routes: Authentication & User Management
# ---------------------------------------------------------

@app.post("/auth/signup", response_model=UserResponse)
def signup(user: UserCreate):
    """
    Register a new user with Permissions.
    """
    if collection_users is None:
        raise HTTPException(status_code=500, detail="DB Error")
    
    if collection_users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Store permissions along with user data
    user_dict = {
        "username": user.username,
        "hashed_password": get_password_hash(user.password),
        "role": user.role, 
        "permissions": user.permissions.dict(),
        "created_at": datetime.now()
    }
    collection_users.insert_one(user_dict)
    
    return {
        "username": user.username, 
        "role": user.role, 
        "permissions": user.permissions
    }

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint. Returns a JWT token.
    Username and Password must be sent as Form Data (standard OAuth2).
    """
    if collection_users is None:
        raise HTTPException(status_code=500, detail="DB Error")
        
    user = collection_users.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Get details of the currently logged-in user.
    """
    return {
        "username": current_user["username"], 
        "role": current_user["role"],
        "permissions": current_user.get("permissions", {})
    }

# --- ADMIN USER MANAGEMENT ROUTES ---

@app.get("/admin/users", response_model=List[UserResponse])
def get_all_users(current_user: dict = Depends(get_current_admin)):
    """
    Admin Only: List all registered users.
    """
    if collection_users is None:
        raise HTTPException(status_code=500, detail="DB Error")
    
    cursor = collection_users.find({})
    users = []
    for u in cursor:
        users.append({
            "username": u["username"],
            "role": u.get("role", "user"),
            "permissions": u.get("permissions", {})
        })
    return users

@app.put("/admin/users/{username}", response_model=UserResponse)
def update_user(
    username: str, 
    user_update: UserUpdate, 
    current_user: dict = Depends(get_current_admin)
):
    """
    Admin Only: Update user permissions, role, or reset password.
    """
    if collection_users is None:
        raise HTTPException(status_code=500, detail="DB Error")
    
    user = collection_users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    update_data = {}
    
    if user_update.role:
        update_data["role"] = user_update.role
    
    if user_update.permissions:
        update_data["permissions"] = user_update.permissions.dict()
        
    if user_update.password:
        update_data["hashed_password"] = get_password_hash(user_update.password)
        
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
        
    collection_users.update_one({"username": username}, {"$set": update_data})
    
    # Return updated document
    updated_user = collection_users.find_one({"username": username})
    return {
        "username": updated_user["username"],
        "role": updated_user.get("role", "user"),
        "permissions": updated_user.get("permissions", {})
    }

@app.delete("/admin/users/{username}")
def delete_user(username: str, current_user: dict = Depends(get_current_admin)):
    """
    Admin Only: Delete a user account.
    """
    if collection_users is None:
        raise HTTPException(status_code=500, detail="DB Error")
    
    # Prevent admin from deleting themselves
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    
    result = collection_users.delete_one({"username": username})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {"message": f"User '{username}' successfully deleted"}


# ---------------------------------------------------------
# Routes: Pipeline Processing
# ---------------------------------------------------------

@app.get("/")
def index():
    return {"message": "Welcome to MACO Automation API. Use /process to upload a file."}


@app.post("/process")
def process_file(
    file: UploadFile = File(...),
    product_group: str = Form(...) 
):
    """
    Process CSV or Excel and store Input/Processed data in MongoDB using GridFS.
    """
    if collection_history is None or collection_input is None or fs is None:
        raise HTTPException(status_code=500, detail="Database connection not available")

    upload_id = str(uuid.uuid4())
    
    # Use a temporary directory for processing to avoid local storage persistence issues
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        try:
            timestamp = datetime.now()
            # Original filename with correct extension (e.g. data.xlsx)
            safe_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            
            # --- 1. Save RAW file to MongoDB GridFS ---
            file.file.seek(0)
            raw_file_id = fs.put(
                file.file, 
                filename=safe_filename, 
                upload_id=upload_id,
                type="raw",
                product_group=product_group
            )

            # --- 2. Create Temp Copy for Pipeline ---
            temp_raw_path = temp_dir_path / safe_filename
            file.file.seek(0)
            with open(temp_raw_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # --- 3. Initial Preprocessing ---
            # Now supports CSV and Excel thanks to updated preprocess.py
            input_df, _ = preprocess_csv(str(temp_raw_path))
            input_df = input_df.reset_index(drop=True) 
            initial_row_count = len(input_df)

            # --- DUPLICATE CHECK (STRICT 12 COLUMNS) ---
            check_cols = [
                "industry", "product_description", "buyer_country", "buyer", 
                "destination_port", "seller_country", "seller", "origin_port", 
                "unit", "quantity", "valueusd", "unit_price"
            ]
            
            projection = {col: 1 for col in check_cols}
            projection["_id"] = 0
            
            existing_cursor = collection_input.find(
                {"product_group": product_group, "data_type": "input"},
                projection
            )
            existing_data = list(existing_cursor)

            duplicates_removed = 0
            if existing_data:
                existing_df = pd.DataFrame(existing_data)
                valid_cols = [c for c in check_cols if c in input_df.columns and c in existing_df.columns]
                
                if valid_cols:
                    # Fingerprinting Function
                    def create_fingerprint(df_subset):
                        temp = df_subset.copy()
                        numeric_cols = ['quantity', 'valueusd', 'unit_price']
                        for col in numeric_cols:
                            if col in temp.columns:
                                temp[col] = (
                                    pd.to_numeric(temp[col], errors='coerce')
                                    .fillna(0).round(2).apply(lambda x: "{:.2f}".format(x))
                                )
                        text_cols = [c for c in temp.columns if c not in numeric_cols]
                        for col in text_cols:
                            temp[col] = (
                                temp[col].astype(str).str.lower()
                                .str.replace(r'[^a-z0-9]', '', regex=True)
                                .replace(['nan', 'none', ''], 'empty')
                            )
                        return temp

                    input_fing = create_fingerprint(input_df[valid_cols])
                    existing_fing = create_fingerprint(existing_df[valid_cols]).drop_duplicates()
                    merged = input_fing.merge(existing_fing, on=valid_cols, how='left', indicator=True)
                    
                    if '_indicator' in merged.columns:
                        mask = (merged['_indicator'] == 'left_only')
                        input_df = input_df[mask.values].copy().reset_index(drop=True)
                        duplicates_removed = initial_row_count - len(input_df)

            if input_df.empty:
                return JSONResponse({
                    "status": "skipped",
                    "message": "All rows in this file are duplicates based on business data.",
                    "duplicates_removed": duplicates_removed
                })
            
            # --- SAVE FILTERED INPUT FOR PIPELINE ---
            # ⚠️ CRITICAL CHANGE: Force intermediate file to be .csv
            # We must use .stem to strip the original extension (e.g. .xlsx) and append .csv
            # This ensures the pipeline reads the temp file correctly as CSV.
            filtered_filename = f"filtered_{Path(safe_filename).stem}.csv"
            temp_filtered_path = temp_dir_path / filtered_filename
            input_df.to_csv(temp_filtered_path, index=False)
            
            # --- 4. Pipeline Execution ---
            processed_df, audit_df, temp_output_path, temp_audit_path = run_pipeline(str(temp_filtered_path), product_group=product_group)
            
            final_processed_filename = os.path.basename(temp_output_path)
            final_audit_filename = os.path.basename(temp_audit_path)

            # --- 5. Upload Processed Result & Audit Report to GridFS ---
            
            # A. Processed File
            with open(temp_output_path, "rb") as f:
                processed_file_id = fs.put(
                    f,
                    filename=final_processed_filename,
                    upload_id=upload_id,
                    type="processed",
                    product_group=product_group
                )

            # B. Audit Report
            with open(temp_audit_path, "rb") as f:
                audit_file_id = fs.put(
                    f,
                    filename=final_audit_filename,
                    upload_id=upload_id,
                    type="audit",
                    product_group=product_group
                )

            # --- 6. Data Storage (Mongo Collections) ---
            input_df["upload_id"] = upload_id
            input_df["source_filename"] = file.filename
            input_df["upload_timestamp"] = timestamp
            input_df["data_type"] = "input"
            input_df["product_group"] = product_group

            input_records = sanitize_dataframe_for_mongo(input_df)
            if input_records:
                collection_input.insert_many(input_records)

            processed_df["upload_id"] = upload_id
            processed_df["source_filename"] = file.filename
            processed_df["upload_timestamp"] = timestamp
            processed_df["data_type"] = "processed"
            processed_df["product_group"] = product_group

            processed_records = sanitize_dataframe_for_mongo(processed_df)
            if processed_records:
                collection_processed.insert_many(processed_records)

            # --- 7. History Log ---
            history_entry = {
                "upload_id": upload_id,
                "filename": file.filename,
                "gridfs_raw_id": raw_file_id,
                "gridfs_processed_id": processed_file_id,
                "gridfs_audit_id": audit_file_id,
                "timestamp": timestamp,
                "rows_input": len(input_df),
                "rows_processed": len(processed_df),
                "audit_issues": len(audit_df),
                "duplicates_removed": duplicates_removed,
                "status": "success",
                "review_status": "working",
                "product_group": product_group
            }
            collection_history.insert_one(history_entry)

            return JSONResponse({
                "status": "success",
                "upload_id": upload_id,
                "rows_processed": len(processed_df),
                "audit_issues": len(audit_df),
                "duplicates_removed": duplicates_removed,
                "processed_file": final_processed_filename,
                "audit_file": final_audit_filename,
                "product_group": product_group,
                "review_status": "working"
            })

        except Exception as e:
            if collection_history is not None:
                 collection_history.insert_one({
                    "upload_id": upload_id,
                    "filename": file.filename,
                    "timestamp": datetime.now(),
                    "status": "failed",
                    "error": str(e),
                    "product_group": product_group
                })
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/process/start-refinement")
def start_refinement(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    product_group: str = Form(...)
):
    job_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    # 1. Read Content Once (Synchronously)
    file_bytes = file.file.read()
    
    # 2. Save to GridFS (So it appears in history downloads)
    if fs:
        # We need a BytesIO wrapper because fs.put expects a file-like object or bytes
        raw_file_id = fs.put(
            file_bytes, 
            filename=f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{file.filename}", 
            upload_id=job_id,
            type="raw",
            product_group=product_group
        )
    else:
        raw_file_id = None

    # 3. Save to Temp Disk for Pipeline
    safe_filename = f"{job_id}_{file.filename}"
    temp_path = UPLOAD_DIR / safe_filename
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    # 4. Run Standard Automation
    # Note: We treat the output of this as our "Input" for the AI refinement
    processed_df, _, _, _ = run_pipeline(str(temp_path), product_group=product_group)
    
    # 5. Create Initial History Record (Fixes "Missing Fields" / "Zero Rows")
    history_entry = {
        "upload_id": job_id,
        "filename": file.filename,
        "gridfs_raw_id": str(raw_file_id) if raw_file_id else None,
        "timestamp": timestamp,
        "rows_input": len(processed_df), # The input to the Refiner is the output of the Pipeline
        "rows_processed": 0, # Will update as we go
        "status": "processing",
        "review_status": "working",
        "product_group": product_group,
        "ai_refined": True
    }
    if collection_history is not None:
        collection_history.insert_one(history_entry)
    
    # 6. Setup Job
    JOB_STORE[job_id] = {
        "status": "processing",
        "total_rows": len(processed_df),
        "processed_rows": 0,
        "data_queue": processed_df,
        "refined_results": []
    }
    
    # [FIX 1] Removed the synchronous 50-row processing block.
    # Send all rows directly to the background task to prevent 504 Timeouts.
    background_tasks.add_task(process_remaining_batches, job_id, product_group)
    
    return {
        "status": "started",
        "job_id": job_id,
        "first_batch": [] # Frontend will now fetch initial data via the standard polling endpoint
    }
    
@app.get("/process/poll/{job_id}")
def poll_results(job_id: str, offset: int = 0):
    job = JOB_STORE.get(job_id)
    if not job: return {"error": "Job not found"}
    
    raw_new_data = job["refined_results"][offset:]
    
    # FIX: Sanitize here to prevent JSON Error
    if raw_new_data:
        # Convert list of dicts to DF temporarily to leverage our sanitizer
        df_new = pd.DataFrame(raw_new_data)
        clean_new_data = sanitize_for_json(df_new)
    else:
        clean_new_data = []
    
    return {
        "status": job["status"],
        "processed": job["processed_rows"],
        "total": job["total_rows"],
        "new_data": clean_new_data
    }
    
def process_remaining_batches(job_id: str, product_group: str):
    job = JOB_STORE[job_id]
    df = job["data_queue"]
    
    batch_size = 10
    
    # 1. Split the dataframe into a list of smaller chunk dataframes
    batches = [df.iloc[i:i + batch_size].copy() for i in range(0, len(df), batch_size)]
    
    print(f"🚀 Job {job_id}: Starting parallel refinement for {len(batches)} batches...")

    # Helper function for the thread pool
    def process_chunk(chunk_df):
        return refine_batch(chunk_df, product_group)

    # 2. Process up to 5 batches concurrently!
    # Removing 'list()' allows it to yield results continuously to your UI poll
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for refined_batch in executor.map(process_chunk, batches):
            job["refined_results"].extend(refined_batch.to_dict(orient="records"))
            job["processed_rows"] += len(refined_batch)

    # B. SAVE TO MONGODB & GRIDFS
    print(f"💾 Saving Job {job_id} to MongoDB...")
    try:
        final_df = pd.DataFrame(job["refined_results"])
        final_df["upload_id"] = job_id
        final_df["data_type"] = "processed"
        final_df["product_group"] = product_group
        final_df["upload_timestamp"] = datetime.now()
        
        # 1. Save Processed Data to Collections
        records = sanitize_dataframe_for_mongo(final_df)
        if collection_processed is not None:
            collection_processed.delete_many({"upload_id": job_id})
            collection_processed.insert_many(records)
            
        # 2. Save Processed File to GridFS (NEW - Fixes missing download)
        processed_file_id = None
        if fs:
            # Create CSV string
            csv_buffer = io.StringIO()
            final_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            
            processed_file_id = fs.put(
                csv_bytes,
                filename=f"refined_{job_id}.csv",
                upload_id=job_id,
                type="processed",
                product_group=product_group
            )

        # 3. Update History
        if collection_history is not None:
            update_fields = {
                "rows_processed": len(final_df), 
                "status": "success", 
                "review_status": "complete",
                "timestamp": datetime.now() # Update completion time
            }
            # Attach GridFS ID if we saved it
            if processed_file_id:
                update_fields["gridfs_processed_id"] = str(processed_file_id)
                
            collection_history.update_one(
                {"upload_id": job_id}, 
                {"$set": update_fields}
            )
            
    except Exception as e:
        print(f"❌ Save Error: {e}")

    job["status"] = "completed"

@app.post("/history/{upload_id}/save")
def save_processed_data(upload_id: str, payload: SaveRequest, current_user: dict = Depends(get_current_user)):
    """
    Update the processed data and review status for a given upload.
    """
    if collection_processed is None or collection_history is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        # Check permissions for this upload
        record = collection_history.find_one({"upload_id": upload_id})
        if not record:
             raise HTTPException(status_code=404, detail="Upload ID not found in history")
        
        # Verify Product Group Access
        allowed_filter = get_allowed_products_filter(current_user)
        if "product_group" in allowed_filter:
            # If there's a restriction, ensure this record matches
            if record.get("product_group") not in allowed_filter["product_group"]["$in"]:
                 raise HTTPException(status_code=403, detail="Access denied for this product group")

        print(f"Saving edits for {upload_id} - Status: {payload.status}")

        # 1. Update History Status
        collection_history.update_one(
            {"upload_id": upload_id},
            {"$set": {"review_status": payload.status}}
        )

        # 2. Process the Data
        df = pd.DataFrame(payload.data)
        df["upload_id"] = upload_id
        df["data_type"] = "processed"
        
        # 3. Replace Records in MongoDB
        collection_processed.delete_many({"upload_id": upload_id})
        
        records = sanitize_dataframe_for_mongo(df)
        if records:
            collection_processed.insert_many(records)
            
        collection_history.update_one(
            {"upload_id": upload_id},
            {"$set": {"rows_processed": len(records)}}
        )

        return {"message": "Data saved successfully", "status": payload.status}

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        print(f"Error saving data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{upload_id}")
def delete_upload(upload_id: str, current_user: dict = Depends(get_current_admin)):
    """
    Completely remove an upload: DB records + GridFS files.
    """
    if collection_history is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        record = collection_history.find_one({"upload_id": upload_id})
        if not record:
            raise HTTPException(status_code=404, detail="Upload not found")
            
        # Delete from GridFS
        if fs:
            if "gridfs_raw_id" in record:
                try: fs.delete(record["gridfs_raw_id"])
                except: pass
            if "gridfs_processed_id" in record:
                try: fs.delete(record["gridfs_processed_id"])
                except: pass

        # Delete DB Records
        if collection_input is not None: 
            collection_input.delete_many({"upload_id": upload_id})
        
        if collection_processed is not None:
            collection_processed.delete_many({"upload_id": upload_id})
            
        collection_history.delete_one({"upload_id": upload_id})
        
        return {"message": "Upload deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
def download_file(filename: str):
    """
    Stream file directly from GridFS.
    """
    if fs is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
        
    try:
        # Get the latest version of the file from GridFS
        grid_out = fs.get_last_version(filename)
        
        return StreamingResponse(
            grid_out, 
            media_type="text/csv", 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="File not found in GridFS.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Routes: History & Data Retrieval
# ---------------------------------------------------------

@app.get("/history")
def get_upload_history(
    product_group: Optional[str] = None, 
    current_user: dict = Depends(get_current_user)
):
    if collection_history is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        # 1. Permission Logic
        query = get_allowed_products_filter(current_user)
        
        # 2. Filter Logic
        if product_group and product_group != "no-products":
            if "product_group" in query:
                allowed_list = query["product_group"]["$in"]
                if product_group not in allowed_list:
                    return {"count": 0, "history": []}
            query["product_group"] = product_group
            
        cursor = collection_history.find(query, {"_id": 0}).sort("timestamp", -1)
        history = list(cursor)

        # 3. Cleanup & Formatting Loop
        for h in history:
            # Tag Orphans
            if "product_group" not in h:
                h["product_group"] = "NEEDS_MIGRATION" 
            
            # FIX: Convert GridFS ObjectIds to strings to prevent JSON serialization errors
            for field in ["gridfs_raw_id", "gridfs_processed_id", "gridfs_audit_id"]:
                if field in h and h[field] is not None:
                    h[field] = str(h[field])

        return {"count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{data_type}/batch")
def get_batch_data(data_type: str, ids: str, format: str = "json", current_user: dict = Depends(get_current_user)):
    """
    Retrieve data for multiple upload IDs combined.
    """
    if data_type not in ["input", "processed"]:
        raise HTTPException(status_code=400, detail="Invalid data type")
        
    collection = collection_input if data_type == "input" else collection_processed
    if collection is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
        
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    
    try:
        # Basic Query
        query = {"upload_id": {"$in": id_list}}
        
        # Add Permissions Filter
        perm_filter = get_allowed_products_filter(current_user)
        query.update(perm_filter)
        
        cursor = collection.find(query, {"_id": 0})
        data = list(cursor)
        
        if not data:
            if format == "csv":
                df = pd.DataFrame()
                stream = io.StringIO()
                df.to_csv(stream, index=False)
                response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
                response.headers["Content-Disposition"] = f"attachment; filename=batch_{data_type}.csv"
                return response
            return {"count": 0, "data": []}

        if format == "csv":
            df = pd.DataFrame(data)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response.headers["Content-Disposition"] = f"attachment; filename=batch_{data_type}_{timestamp}.csv"
            return response
            
        return {"count": len(data), "data": data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/input/all")
def get_all_input_data(format: str = "json", current_user: dict = Depends(get_current_user)):
    if collection_input is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        # Apply permission filter
        query = get_allowed_products_filter(current_user)
        
        cursor = collection_input.find(query, {"_id": 0})
        data = list(cursor)
        if not data: return {"message": "No input data found."}
        
        if format == "csv":
            df = pd.DataFrame(data)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=all_input_data.csv"
            return response
        return {"count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/processed/all")
def get_all_processed_data(format: str = "json", current_user: dict = Depends(get_current_user)):
    if collection_processed is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        # Apply permission filter
        query = get_allowed_products_filter(current_user)
        
        cursor = collection_processed.find(query, {"_id": 0})
        data = list(cursor)
        if not data: return {"message": "No processed data found."}

        if format == "csv":
            df = pd.DataFrame(data)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=all_processed_data.csv"
            return response
        return {"count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{upload_id}/input")
def get_upload_input_data(
    upload_id: str, 
    format: str = "json",
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=1000, description="Rows per page"), 
    current_user: dict = Depends(get_current_user)
):
    if collection_history is None or collection_input is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    # 1. Fetch History Record
    history_record = collection_history.find_one({"upload_id": upload_id})
    if not history_record:
        raise HTTPException(status_code=404, detail="Upload ID not found in history.")

    # 2. Security Check
    allowed_filter = get_allowed_products_filter(current_user)
    if "product_group" in allowed_filter:
        allowed = allowed_filter["product_group"]["$in"]
        if history_record.get("product_group", "ROTARY_UNION") not in allowed:
             raise HTTPException(status_code=403, detail="Access denied.")

    # 3. Helper for GridFS ID & Filename
    raw_id = history_record.get("gridfs_raw_id")
    if isinstance(raw_id, str): raw_id = ObjectId(raw_id)
    original_filename = history_record.get('filename', 'input.csv')

    # --- SCENARIO A: CSV DOWNLOAD (No Pagination - Stream Full File) ---
    if format == "csv":
        if raw_id and fs.exists(raw_id):
            grid_out = fs.get(raw_id)
            
            # FIX 1: Provide correct media type based on original file extension
            m_type = "text/csv"
            if original_filename.lower().endswith(".xlsx"):
                m_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif original_filename.lower().endswith(".xls"):
                m_type = "application/vnd.ms-excel"

            return StreamingResponse(
                grid_out, 
                media_type=m_type, 
                headers={"Content-Disposition": f"attachment; filename={original_filename}"}
            )
        
        # Fallback to DB if GridFS missing
        query = {"upload_id": upload_id}
        cursor = collection_input.find(query, {"_id": 0})
        # Note: If DB is huge, streaming cursor to CSV is better, but keeping simple for now
        data = list(cursor)
        if data:
            df = pd.DataFrame(data)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = f"attachment; filename=input_{upload_id}.csv"
            return response

        raise HTTPException(status_code=404, detail="File not found.")

    # --- SCENARIO B: JSON RESPONSE (PAGINATED) ---
    else:
        # Calculate Skip
        skip = (page - 1) * limit

        # Strategy 1: Database Pagination (Efficient)
        query = {"upload_id": upload_id}
        total_count = collection_input.count_documents(query)

        if total_count > 0:
            cursor = collection_input.find(query, {"_id": 0}).skip(skip).limit(limit)
            data = list(cursor)
            
            return {
                "upload_id": upload_id,
                "data": data,
                "pagination": {
                    "current_page": page,
                    "limit": limit,
                    "total_items": total_count,
                    "total_pages": (total_count + limit - 1) // limit
                },
                "source": "database"
            }

        # Strategy 2: GridFS Fallback (Slice in Memory)
        # If DB is empty, we load the file but ONLY return the requested chunk
        if raw_id and fs.exists(raw_id):
            try:
                grid_out = fs.get(raw_id)
                
                # FIX 2: Check extension to support both CSV and Excel Fallbacks
                if original_filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(grid_out)
                else:
                    # FIX 3: Add fallback for weird character encodings
                    try:
                        df = pd.read_csv(grid_out)
                    except UnicodeDecodeError:
                        grid_out.seek(0)
                        df = pd.read_csv(grid_out, encoding="latin1")
                
                total_count = len(df)
                # Slice the DataFrame for pagination
                paginated_df = df.iloc[skip : skip + limit]
                
                clean_data = sanitize_for_json(paginated_df)
                
                return {
                    "upload_id": upload_id, 
                    "data": clean_data, 
                    "pagination": {
                        "current_page": page,
                        "limit": limit,
                        "total_items": total_count,
                        "total_pages": (total_count + limit - 1) // limit
                    },
                    "source": "gridfs_fallback"
                }
            except Exception as e:
                print(f"Error parsing GridFS file: {e}")

        raise HTTPException(status_code=404, detail="No data found.")

@app.get("/history/{upload_id}/processed")
def get_upload_processed_data(upload_id: str, format: str = "json", current_user: dict = Depends(get_current_user)):
    if collection_processed is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        # Basic Query
        query = {"upload_id": upload_id}
        # Apply permission filter
        query.update(get_allowed_products_filter(current_user))

        cursor = collection_processed.find(query, {"_id": 0})
        data = list(cursor)
        
        if format == "csv":
            df = pd.DataFrame(data)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = f"attachment; filename=processed_{upload_id}.csv"
            return response
        return {"upload_id": upload_id, "count": len(data), "data": data}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/data/completed")
def get_completed_rows(
    product_group: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=10000, description="Rows per page"),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve all processed rows where the parent upload status is 'complete'.
    Supports filtering by product_group and pagination.
    """
    if collection_history is None or collection_processed is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        # 1. Build Query for History (Find completed uploads)
        history_query = {"review_status": "complete"}

        # 2. Apply Security & Product Filters
        # Get user's allowed products
        permission_filter = get_allowed_products_filter(current_user)
        
        # If user requests a specific group, validate it against permissions
        if product_group:
            if "product_group" in permission_filter:
                # If user has restricted access, ensure requested group is allowed
                allowed = permission_filter["product_group"]["$in"]
                if product_group not in allowed:
                    raise HTTPException(status_code=403, detail="Access denied for this product group")
            history_query["product_group"] = product_group
        else:
            # If no specific group requested, apply generic permission filter
            history_query.update(permission_filter)

        # 3. Get relevant Upload IDs from History
        # We only want the IDs to filter the processed collection
        valid_uploads = list(collection_history.find(history_query, {"upload_id": 1, "_id": 0}))
        
        if not valid_uploads:
            return {
                "count": 0, 
                "total_pages": 0, 
                "current_page": page, 
                "data": []
            }

        valid_ids = [u["upload_id"] for u in valid_uploads]

        # 4. Query Processed Data Store
        data_query = {"upload_id": {"$in": valid_ids}}
        
        # Calculate pagination
        skip = (page - 1) * limit
        
        total_count = collection_processed.count_documents(data_query)
        cursor = collection_processed.find(data_query, {"_id": 0}).skip(skip).limit(limit)
        data = list(cursor)

        return {
            "count": total_count,
            "total_pages": (total_count + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
            "data": data
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# ADMIN API (PROTECTED)
# ---------------------------------------------------------
class DictionaryEntry(BaseModel):
    product_group: str
    dictionary_type: str
    source_key: str
    target_value: str

class DictionaryDeleteRequest(BaseModel):
    product_group: str
    dictionary_type: str
    source_key: str

class CreateEntityRequest(BaseModel):
    name: str

class DeleteEntityRequest(BaseModel):
    name: str

class DictionaryRenameRequest(BaseModel):
    product_group: str
    dictionary_type: str
    old_source_key: str
    new_source_key: str

# --- 1. PRODUCT MANAGEMENT ---

@app.get("/admin/products")
def get_product_groups(current_user: dict = Depends(get_current_admin)):
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    try: 
        return {"product_groups": collection_dicts.distinct("product_group")}
    except errors.PyMongoError as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/products")
def create_product_group(req: CreateEntityRequest, current_user: dict = Depends(get_current_admin)):
    """
    Creates a new product group AND populates it with independent copies 
    of all standard dictionaries (10+ tables).
    """
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    
    new_group = req.name.strip().upper().replace(" ", "_")
    
    # Check if exists
    if collection_dicts.find_one({"product_group": new_group}):
         raise HTTPException(status_code=400, detail=f"Product group '{new_group}' already exists.")

    docs_to_insert = []
    
    # A. Metadata
    docs_to_insert.append({
        "product_group": new_group, 
        "dictionary_type": "__metadata__", 
        "source_key": "created_at", 
        "target_value": datetime.now().isoformat()
    })

    # B. Clone Standard Dictionaries
    # 1. Maps
    for k, v in COMPANY_DICTIONARY.items():
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "company_map", "source_key": k, "target_value": v})
    for k, v in MODEL_DICTIONARY.items():
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "model_map", "source_key": k, "target_value": v})
    for k, v in PATTERN_HINTS.items():
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "pattern_hints", "source_key": k, "target_value": v})
    for k, v in APPLICATION_MAP.items():
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "application_map", "source_key": k, "target_value": v})
    for k, v in UNIT_MAP.items():
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "unit_map", "source_key": k, "target_value": v})
    
    # 2. Lists (Value = "true" or "Spare"/"Unit")
    for k in SPARE_KEYWORDS:
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "classification_keywords", "source_key": k, "target_value": "Spare"})
    for k in UNIT_KEYWORDS:
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "classification_keywords", "source_key": k, "target_value": "Unit"})
    for k in MANUFACTURER_KEYWORDS:
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "manufacturer_keywords", "source_key": k, "target_value": "true"})
    for k in MODEL_PATTERNS:
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "model_patterns", "source_key": k, "target_value": "true"})
    for k in OTHER_MACHINE_KEYWORDS:
        docs_to_insert.append({"product_group": new_group, "dictionary_type": "other_machine_keywords", "source_key": k, "target_value": "true"})

    # 3. Flattened Maps (Condition)
    for target, keywords in CONDITION_KEYWORDS.items():
        for k in keywords:
            docs_to_insert.append({"product_group": new_group, "dictionary_type": "condition_map", "source_key": k, "target_value": target})

    try:
        collection_dicts.insert_many(docs_to_insert)
        return {"message": f"Created product '{new_group}' with {len(docs_to_insert)} default rules."}
    except errors.PyMongoError as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/products")
def delete_product_group(req: DeleteEntityRequest, current_user: dict = Depends(get_current_admin)):
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    try:
        res = collection_dicts.delete_many({"product_group": req.name.strip().upper().replace(" ", "_")})
        if res.deleted_count > 0: return {"message": "Deleted"}
        raise HTTPException(status_code=404, detail="Not found")
    except errors.PyMongoError as e: raise HTTPException(status_code=500, detail=str(e))

# --- 2. DICTIONARY TYPES & REFRESH ---

@app.get("/admin/dictionary-types")
def get_dictionary_types(product_group: Optional[str] = None, current_user: dict = Depends(get_current_admin)):
    """
    Get available dictionary types. 
    If product_group is provided, shows types relevant to that group.
    """
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    try:
        query = {}
        if product_group:
            query["product_group"] = product_group
            
        types = collection_dicts.distinct("dictionary_type", query)
        return {"dictionary_types": [t for t in types if t != "__metadata__"]}
    except errors.PyMongoError as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/refresh-dictionaries")
def refresh_dicts(current_user: dict = Depends(get_current_admin)):
    """
    Force reload of ALL dictionaries from MongoDB into memory.
    Now correctly detects all product groups.
    """
    try:
        # 1. Discover all groups
        all_groups = collection_dicts.distinct("product_group")
        # 2. Reload everything
        reference_dicts.load_from_db(product_groups=all_groups)
        return {"message": f"Dictionaries reloaded successfully for: {all_groups}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload dictionaries: {str(e)}")

# --- 3. DICTIONARY ENTRIES (CRUD) ---

@app.get("/admin/dictionary/{product_group}")
def get_dictionary_entries(product_group: str, dictionary_type: Optional[str] = Query(None), current_user: dict = Depends(get_current_admin)):
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    
    query = {"product_group": product_group, "dictionary_type": {"$ne": "__metadata__"}}
    if dictionary_type: 
        query["dictionary_type"] = dictionary_type
        
    try: 
        entries = list(collection_dicts.find(query, {"_id": 0}))
        return {"count": len(entries), "entries": entries}
    except errors.PyMongoError as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/dictionary")
def upsert_dictionary_entry(
    entry: DictionaryEntry, 
    current_user: dict = Depends(get_current_admin)
):
    """
    Adds a dictionary rule (e.g., Company Map) to ALL product groups automatically.
    This ensures consistency across the entire ecosystem.
    """
    if collection_dicts is None: 
        raise HTTPException(status_code=500, detail="DB Error")
    
    try:
        # 1. Identify all target groups
        # We fetch all existing groups to "broadcast" this rule to everyone.
        all_groups = collection_dicts.distinct("product_group")
        
        # Edge case: If this is the very first entry for a new group, ensure it's included
        if entry.product_group not in all_groups:
            all_groups.append(entry.product_group)

        # 2. Loop and Save to EVERY group
        for group in all_groups:
            doc = entry.dict()
            doc["product_group"] = group  # <--- Override the group here
            
            collection_dicts.replace_one(
                {
                    "product_group": group, 
                    "dictionary_type": entry.dictionary_type, 
                    "source_key": entry.source_key
                }, 
                doc, 
                upsert=True
            )
        
        # 3. Auto-Refresh Memory
        # Reloads the rules so the /process endpoint sees them immediately.
        reference_dicts.load_from_db(product_groups=all_groups)
        
        return {
            "message": f"Saved rule to {len(all_groups)} product groups & Dictionaries Refreshed", 
            "entry": entry.dict()
        }

    except errors.PyMongoError as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/dictionary")
def delete_dictionary_entry(req: DictionaryDeleteRequest, current_user: dict = Depends(get_current_admin)):
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    try:
        res = collection_dicts.delete_one({
            "product_group": req.product_group, 
            "dictionary_type": req.dictionary_type, 
            "source_key": req.source_key
        })
        if res.deleted_count == 1: return {"message": "Deleted"}
        raise HTTPException(status_code=404, detail="Not found")
    except errors.PyMongoError as e: raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/dictionary/rename")
def rename_dictionary_key(req: DictionaryRenameRequest, current_user: dict = Depends(get_current_admin)):
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    
    # Filter strictly by product_group
    old_query = {"product_group": req.product_group, "dictionary_type": req.dictionary_type, "source_key": req.old_source_key}
    new_query = {"product_group": req.product_group, "dictionary_type": req.dictionary_type, "source_key": req.new_source_key}

    if not collection_dicts.find_one(old_query): raise HTTPException(status_code=404, detail="Not found")
    if collection_dicts.find_one(new_query): raise HTTPException(status_code=400, detail="New key already exists for this product")

    try:
        collection_dicts.update_one(old_query, {"$set": {"source_key": req.new_source_key}})
        return {"message": "Renamed"}
    except errors.PyMongoError as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/dictionary/{dictionary_type}/target_value")
def get_target_values_list(dictionary_type: str, product_group: Optional[str] = Query(None), current_user: dict = Depends(get_current_admin)):
    """
    Get distinct target values for a dictionary type.
    Crucial: Now filters by product_group to avoid mixing contexts (e.g. Lift vs Rotary).
    """
    if collection_dicts is None: raise HTTPException(status_code=500, detail="DB Error")
    
    key = dictionary_type.strip().lower().replace("/", "_").replace(" ", "_").replace("__", "_")
    query = {"dictionary_type": key}
    if product_group:
        query["product_group"] = product_group

    try: 
        return sorted(collection_dicts.distinct("target_value", query))
    except errors.PyMongoError as e: 
        raise HTTPException(status_code=500, detail=str(e))


#---------------------------------------------------------
# Financial Routes
#---------------------------------------------------------
# 1. CREATE Legal Entity
@app.post("/admin/legal-entities", response_model=LegalEntityResponse)
def create_legal_entity(entity: LegalEntityCreate, current_user: dict = Depends(get_current_admin)):
    if collection_legal_entities is None: raise HTTPException(500, "DB Error")
    if collection_legal_entities.find_one({"tax_id": entity.tax_id}):
        raise HTTPException(400, f"Tax ID '{entity.tax_id}' already exists.")
        
    doc = entity.dict()
    doc["created_at"] = datetime.now()
    res = collection_legal_entities.insert_one(doc)
    doc["id"] = str(res.inserted_id)
    return doc

# 2. UPDATE Legal Entity
@app.put("/admin/legal-entities/{tax_id}", response_model=LegalEntityResponse)
def update_legal_entity(tax_id: str, update: LegalEntityUpdate, current_user: dict = Depends(get_current_admin)):
    if collection_legal_entities is None: raise HTTPException(500, "DB Error")
    
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    if not update_data: raise HTTPException(400, "No data provided")

    res = collection_legal_entities.find_one_and_update(
        {"tax_id": tax_id}, {"$set": update_data}, return_document=True
    )
    if not res: raise HTTPException(404, "Entity not found")
    res["id"] = str(res["_id"])
    return res

# 3. DELETE Legal Entity
@app.delete("/admin/legal-entities/{tax_id}")
def delete_legal_entity(tax_id: str, current_user: dict = Depends(get_current_admin)):
    if collection_legal_entities is None: raise HTTPException(500, "DB Error")
    res = collection_legal_entities.delete_one({"tax_id": tax_id})
    if res.deleted_count == 0: raise HTTPException(404, "Entity not found")
    
    # Cascade delete financial records
    fin_res = collection_financial.delete_many({"tax_id": tax_id})
    return {"message": "Deleted", "records_deleted": fin_res.deleted_count}

# 4. GET ALL Legal Entities
@app.get("/admin/legal-entities", response_model=List[LegalEntityResponse])
def get_all_legal_entities(current_user: dict = Depends(get_current_user)):
    if collection_legal_entities is None: raise HTTPException(500, "DB Error")
    return [{"id": str(d["_id"]), **d} for d in collection_legal_entities.find().sort("created_at", -1)]

# 5. UPLOAD Financial File
@app.post("/finance/upload", response_model=FinancialRecordResponse)
def upload_financial_data(
    file: UploadFile = File(...),
    tax_id: str = Form(...),
    financial_year: str = Form(...),
    file_type: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    if collection_financial is None: raise HTTPException(500, "DB Error")
    
    # Verify Entity
    entity = collection_legal_entities.find_one({"tax_id": tax_id})
    if not entity: raise HTTPException(404, "Legal Entity not found")
    
    # Parse File
    content = file.read()
    parsed_data = parse_financial_file(content, file.filename)
    if not parsed_data: raise HTTPException(400, "Could not parse file data.")

    # Save Record
    doc = {
        "upload_id": str(uuid.uuid4()),
        "tax_id": tax_id,
        "financial_year": financial_year,
        "file_type": file_type,
        "filename": file.filename,
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.now(),
        "data": parsed_data
    }
    collection_financial.insert_one(doc)

    return {
        "upload_id": doc["upload_id"],
        "tax_id": doc["tax_id"],
        "entity_name": entity["entity_name"],
        "financial_year": doc["financial_year"],
        "file_type": doc["file_type"],
        "filename": doc["filename"],
        "uploaded_at": doc["uploaded_at"],
        "data_preview": doc["data"][:5]
    }

# 6. GET Financial File (Search/Dashboard)
@app.get("/finance/entity/{tax_id}")
def get_entity_dashboard(tax_id: str, current_user: dict = Depends(get_current_user)):
    """Returns Entity Details + Aggregated Financial Data."""
    if collection_legal_entities is None: raise HTTPException(500, "DB Error")
    
    entity = collection_legal_entities.find_one({"tax_id": tax_id})
    if not entity: raise HTTPException(404, "Entity not found")
    
    # Find Aliases from Dictionary
    aliases = []
    if entity.get("mapped_companies"):
        dict_entries = collection_dicts.find({
            "dictionary_type": "company_map", 
            "target_value": {"$in": entity["mapped_companies"]}
        })
        aliases = list(set([d["source_key"] for d in dict_entries]))

    # Get Financial Data grouped by File Type
    fin_cursor = collection_financial.find({"tax_id": tax_id})
    financial_data = {}
    
    for doc in fin_cursor:
        ftype = doc["file_type"]
        if ftype not in financial_data: financial_data[ftype] = []
        financial_data[ftype].append({
            "upload_id": doc["upload_id"],
            "year": doc["financial_year"],
            "filename": doc["filename"],
            "uploaded_at": doc["uploaded_at"],
            "metrics": doc["data"] # Full data for display
        })

    return {
        "entity": {
            "id": str(entity["_id"]),
            "entity_name": entity["entity_name"],
            "tax_id": entity["tax_id"],
            "mapped_companies": entity.get("mapped_companies", []),
            "company_variations": aliases
        },
        "financial_data": financial_data
    }

# 7. DELETE Financial File
@app.delete("/finance/file/{upload_id}")
def delete_financial_file(upload_id: str, current_user: dict = Depends(get_current_admin)):
    if collection_financial is None: raise HTTPException(500, "DB Error")
    res = collection_financial.delete_one({"upload_id": upload_id})
    if res.deleted_count == 0: raise HTTPException(404, "File not found")
    return {"message": "Deleted successfully"}

# ---------------------------------------------------------
# MIGRATION UTILITY
# ---------------------------------------------------------
@app.post("/admin/migrate-status")
def migrate_status(current_user: dict = Depends(get_current_admin)):
    """
    Utility to migrate old records:
    - Set review_status = 'working'
    - Set product_group = 'ROTARY_UNION'
    """
    if collection_history is None:
        raise HTTPException(status_code=500, detail="DB Error")
    
    try:
        # Update History
        res_hist = collection_history.update_many(
            {}, 
            {"$set": {"review_status": "working", "product_group": "ROTARY_UNION"}}
        )
        
        # Update Input (for consistency)
        if collection_input is not None:
            collection_input.update_many(
                {}, 
                {"$set": {"product_group": "ROTARY_UNION"}}
            )
            
        # Update Processed (for consistency)
        if collection_processed is not None:
            collection_processed.update_many(
                {}, 
                {"$set": {"product_group": "ROTARY_UNION"}}
            )
            
        return {
            "message": "Migration complete",
            "modified_history": res_hist.modified_count
        }
    except Exception as e:
        print(f"Migration Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))