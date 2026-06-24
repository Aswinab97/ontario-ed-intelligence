import sqlite3
import os
import pandas as pd
from pathlib import Path

def build_warehouse():
    # 1. Set paths relative to this script's location
    base_dir = Path(__file__).resolve().parent
    db_dir = base_dir / "database"
    db_path = db_dir / "healthcare.db"
    csv_path = base_dir / "fact_operations.csv"
    
    # Ensure database directory exists inside the project folder
    db_dir.mkdir(exist_ok=True)
    
    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 2. CREATE DIMENSION TABLES
    print("Creating Dimension Tables...")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_hospital (
        HospitalID INTEGER PRIMARY KEY AUTOINCREMENT,
        HospitalName TEXT UNIQUE,
        Region TEXT,
        HospitalType TEXT
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_date (
        DateID INTEGER PRIMARY KEY,
        Date TEXT,
        Month TEXT,
        Quarter TEXT,
        Year INTEGER
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_icd10 (
        ICD10 TEXT PRIMARY KEY,
        Diagnosis TEXT,
        Category TEXT
    );
    """)
    
    # 3. CREATE FACT TABLES
    print("Creating Fact Tables...")
    
    # Executive / Aggregate Operations Fact Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fact_operations (
        RecordID INTEGER PRIMARY KEY AUTOINCREMENT,
        Hospital TEXT,  
        Date TEXT,
        ED_Visits INTEGER,
        Wait_Time_Hours REAL,
        LOS_Days REAL,
        Bed_Occupancy_Pct REAL,
        ALC_Patients INTEGER,
        Admission_Rate_Pct REAL,
        LWBS_Rate_Pct REAL,
        Readmission_Rate_Pct REAL
    );
    """)
    
    # Granular ED Visits Fact Table (Simulating NACRS layout)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fact_ed_visits (
        VisitID INTEGER PRIMARY KEY AUTOINCREMENT,
        HospitalID INTEGER,
        DateID INTEGER,
        CTAS_Level INTEGER,
        Wait_Time REAL,
        LOS REAL,
        Admitted_Flag INTEGER,
        ICD10 TEXT,
        FOREIGN KEY(HospitalID) REFERENCES dim_hospital(HospitalID),
        FOREIGN KEY(DateID) REFERENCES dim_date(DateID),
        FOREIGN KEY(ICD10) REFERENCES dim_icd10(ICD10)
    );
    """)

    conn.commit()
    
    # 4. LOAD EXISTING CSV DATA
    if csv_path.exists():
        print(f"Found {csv_path.name}. Ingesting staging data...")
        df = pd.read_csv(csv_path)
        
        # Write to the aggregate operations staging table
        df.to_sql("fact_operations", conn, if_exists="replace", index=False)
        print("✅ fact_operations populated successfully from CSV.")
        
        # Populate dim_hospital from your CSV data to seed the lookup
        if 'Hospital' in df.columns:
            distinct_hospitals = df[['Hospital']].drop_duplicates().dropna()
            for hosp in distinct_hospitals['Hospital']:
                # Parse localized regions and hospital classification definitions
                region = "GTA" if any(x in hosp for x in ["Toronto", "Sunnybrook", "Unity", "Humber"]) else "Ontario Central"
                h_type = "Academic" if any(x in hosp for x in ["HSC", "Health", "Teaching", "Sunnybrook"]) else "Community"
                
                cursor.execute("""
                INSERT INTO dim_hospital (HospitalName, Region, HospitalType)
                VALUES (?, ?, ?)
                ON CONFLICT(HospitalName) DO NOTHING
                """, (hosp, region, h_type))
            conn.commit()
            print("✅ dim_hospital lookup seeded from CSV unique entries.")
    else:
        print(f"⚠️ Warning: {csv_path.name} not found in root directory. Skipping CSV ingestion.")
        
    conn.close()
    print("🎉 Warehouse setup execution complete!")

if __name__ == '__main__':
    build_warehouse()