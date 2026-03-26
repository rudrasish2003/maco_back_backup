import os
import certifi
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load your MongoDB URI from the .env file
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "maco_db"
COLLECTION_NAME = "product_dictionaries"

def update_company_map_for_all(excel_file_path):
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Dynamically fetch ALL existing product groups from the database
    all_product_groups = collection.distinct("product_group")
    
    if not all_product_groups:
        print("⚠️ No product groups found in the database. Make sure your DB has data.")
        client.close()
        return
        
    print(f"Connecting to DB to update company_map for ALL groups: {all_product_groups}...")

    # 1. Delete the existing company_map data for ALL these product groups
    delete_result = collection.delete_many({
        "dictionary_type": "company_map",
        "product_group": {"$in": all_product_groups}
    })
    print(f"🗑️ Deleted {delete_result.deleted_count} existing company mappings.")

    # 2. Read the new Excel (.xlsx) file directly
    print(f"📄 Reading Excel file: {excel_file_path}")
    df = pd.read_excel(excel_file_path)
    
    # 3. Format the data to match your database schema
    documents = []
    for index, row in df.iterrows():
        # Get the keys from your specific Excel headers
        source = str(row.get('Source Key', '')).strip()
        target = str(row.get('Target value', '')).strip()
        
        # Skip if the row is completely empty or invalid
        if not source or source == 'nan' or source == 'None' or not target or target == 'nan' or target == 'None':
            continue

        # Create a database entry for EVERY active product group
        for group in all_product_groups:
            documents.append({
                "product_group": group,
                "dictionary_type": "company_map",
                "source_key": source,
                "target_value": target
            })

    # 4. Insert the new data into MongoDB
    if documents:
        insert_result = collection.insert_many(documents)
        print(f"✅ Successfully inserted {len(insert_result.inserted_ids)} new mappings across all groups.")
    else:
        print("⚠️ No valid data found in the Excel file.")

    client.close()

if __name__ == "__main__":
    # Ensure this exactly matches the name of your Excel file in the folder
    excel_file = "EXIM Grouping_LIFT & RU.xlsx"
    
    # Run the function for all groups
    update_company_map_for_all(excel_file)