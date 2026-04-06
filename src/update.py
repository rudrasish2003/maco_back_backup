import os
from datetime import datetime
from pymongo import MongoClient
import certifi

# Your MongoDB Setup
MONGO_URI = "mongodb+srv://info_db_user:T9j4ZOpejvbh6MA8@cluster0.8aptjw.mongodb.net/?appName=Cluster0"
DB_NAME = "maco_db"

def seed_new_product_groups_from_db():
    print("🔌 Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    collection_dicts = db["product_dictionaries"]

    # 1. Fetch the baseline rules from an existing product group (e.g., ROTARY_UNION)
    print("📥 Fetching baseline rules from 'ROTARY_UNION'...")
    base_rules = list(collection_dicts.find({
        "product_group": "ROTARY_UNION", 
        "dictionary_type": {"$ne": "__metadata__"} # Don't copy the old metadata
    }))

    if not base_rules:
        print("❌ ERROR: No rules found for 'ROTARY_UNION' in the database!")
        print("You must have at least one product group populated in the DB to clone from.")
        return

    print(f"✅ Found {len(base_rules)} base rules to clone.")

    # 2. The new product groups you want to fix/add
    new_groups = ["BARREL_COUPLING", "MAGNESSIUM", "MUD_PUMPS"]

    for group in new_groups:
        # --- DELETE EXISTING EMPTY RECORDS FIRST ---
        existing_count = collection_dicts.count_documents({"product_group": group})
        if existing_count > 0:
            print(f"⚠️ Found {existing_count} existing records for '{group}'. Deleting them now...")
            collection_dicts.delete_many({"product_group": group})
            print(f"🗑️ Deleted old records for '{group}'.")

        print(f"\n🚀 Creating '{group}' and cloning baseline dictionaries...")
        docs_to_insert = []

        # A. Create Metadata for the new group
        docs_to_insert.append({
            "product_group": group, 
            "dictionary_type": "__metadata__", 
            "source_key": "created_at", 
            "target_value": datetime.now().isoformat()
        })

        # B. Clone all the base rules, simply swapping out the product_group name
        for rule in base_rules:
            # Create a copy of the dictionary without the MongoDB '_id' field
            new_rule = {k: v for k, v in rule.items() if k != '_id'}
            new_rule["product_group"] = group
            docs_to_insert.append(new_rule)

        # C. Insert everything into the DB in one batch
        if docs_to_insert:
            collection_dicts.insert_many(docs_to_insert)
            print(f"✅ Successfully inserted {len(docs_to_insert)} populated rules for '{group}'.")

    print("\n🎉 Database seeding complete! Restart your FastAPI server.")

if __name__ == "__main__":
    seed_new_product_groups_from_db()