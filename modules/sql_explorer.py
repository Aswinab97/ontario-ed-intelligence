import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

def run_query(query):
    # Dynamically find the database file path relative to this script
    db_path = Path(__file__).resolve().parent.parent / "database" / "healthcare.db"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)

def show_sql_explorer():
    st.title("🔍 Healthcare Data Warehouse Explorer")
    st.markdown("""
    This interactive playground exposes an **Ontario ED Data Warehouse** designed using a Star Schema framework. 
    You can run custom SQL queries against performance metrics mimicking **NACRS (ED Visits)** and **DAD (Inpatient Admissions)** frameworks.
    """)
    
    # 1. Sidebar Schema Metadata Reference
    st.sidebar.subheader("Warehouse Schema Reference")
    table_choice = st.sidebar.selectbox("Inspect Architecture Schema:", ["fact_operations", "dim_hospital"])
    
    if table_choice:
        try:
            schema_df = run_query(f"PRAGMA table_info({table_choice});")
            st.sidebar.dataframe(schema_df[['name', 'type']], use_container_width=True)
        except Exception as e:
            st.sidebar.error("Could not read database schema.")

    # 2. Hardcoded Analytical Use-Case Query Templates (Step 8)
    st.subheader("Pre-configured Executive Query Templates")
    query_presets = {
        "Select a KPI template to auto-load...": "",
        "1. High-Strain Operational Bottlenecks (Wait Time > 4 hrs)": """SELECT 
    Hospital, 
    ROUND(AVG(Wait_Time_Hours), 2) as Avg_Wait_Hours, 
    ROUND(AVG(LOS_Days), 2) as Avg_Inpatient_LOS_Days 
FROM fact_operations 
GROUP BY Hospital 
HAVING Avg_Wait_Hours > 4.0
ORDER BY Avg_Wait_Hours DESC;""",
        
        "2. Alternate Level of Care (ALC) Impact on Occupancy": """SELECT 
    Date, 
    Hospital, 
    Bed_Occupancy_Pct, 
    ALC_Patients 
FROM fact_operations 
WHERE Bed_Occupancy_Pct > 85.0 
ORDER BY ALC_Patients DESC, Bed_Occupancy_Pct DESC;""",
        
        "3. Attrition Indicators (LWBS vs Wait Times)": """SELECT 
    Hospital, 
    ROUND(AVG(Wait_Time_Hours), 2) as Avg_Wait_Hours, 
    ROUND(AVG(LWBS_Rate_Pct), 2) as Avg_Left_Without_Being_Seen_Pct 
FROM fact_operations 
GROUP BY Hospital 
ORDER BY Avg_Left_Without_Being_Seen_Pct DESC;""",

        "4. System Capacity Cross-Join (Fact + Dimension Join)": """SELECT 
    h.HospitalType,
    h.Region,
    ROUND(AVG(f.Bed_Occupancy_Pct), 2) as Avg_Bed_Occupancy
FROM fact_operations f
JOIN dim_hospital h ON f.Hospital = h.HospitalName
GROUP BY h.HospitalType, h.Region;"""
    }

    selected_preset = st.selectbox("Load Standard Healthcare Template Query:", list(query_presets.keys()))
    
    # Determine what code goes into the editor box
    default_query = query_presets[selected_preset] if selected_preset != "Select a KPI template to auto-load..." else """-- Write custom SQLite code here
SELECT Hospital, ROUND(AVG(Wait_Time_Hours), 2) AS AvgWait 
FROM fact_operations 
GROUP BY Hospital;"""

    # 3. Main Text Area Entry
    query_input = st.text_area("SQL Input Window (SQLite Dialect)", value=default_query, height=180)
    
    # 4. Execution Engine
    if st.button("Execute Analytics Query", type="primary"):
        if query_input.strip():
            try:
                with st.spinner("Processing execution plan..."):
                    results_df = run_query(query_input)
                
                st.success(f"Returned {len(results_df)} structured records")
                st.dataframe(results_df, use_container_width=True)
                
                # Dynamic visual charting from query results if applicable
                if not results_df.empty and len(results_df.columns) >= 2:
                    st.subheader("Instant Query Visualization")
                    x_axis = results_df.columns[0]
                    y_axis = results_df.columns[1]
                    
                    # Convert to numeric safely if chartable
                    try:
                        results_df[y_axis] = pd.to_numeric(results_df[y_axis])
                        st.bar_chart(data=results_df, x=x_axis, y=y_axis)
                    except Exception:
                        st.info("Query output schema is not numeric; skipping bar chart rendering.")
                        
            except Exception as e:
                st.error(f"SQL Execution Engine Error: {e}")

if __name__ == "__main__":
    show_sql_explorer()
