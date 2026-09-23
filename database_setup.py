import sqlite3
import pandas as pd
import os

def initialize_database():
    print("--- Setting up SQL Database Layer ---")
    
    # 1. Load raw data
    csv_path = 'data/customer_churn.csv'
    if not os.path.exists(csv_path):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 2. Connect to local SQLite database (acts as your SQL storage layer)
    conn = sqlite3.connect('customer_churn.db')
    cursor = conn.cursor()
    
    # 3. Write dataframe to SQL table
    df.to_sql('customers', conn, if_exists='replace', index=False)
    print("Success: Customer records loaded into SQL database table 'customers'.")
    
    # 4. Execute sample analytical queries (Section 9 requirements)
    print("\n--- Executing SQL Analytical Queries ---")
    
    # Query 1: Overall Churn Summary
    query_summary = """
    SELECT 
        COUNT(*) as Total_Customers,
        SUM(CASE WHEN Churn = 1 OR Churn = 'True' THEN 1 ELSE 0 END) as Churned_Count,
        ROUND(AVG(CASE WHEN Churn = 1 OR Churn = 'True' THEN 1.0 ELSE 0.0 END) * 100, 2) as Churn_Rate_Percentage
    FROM customers;
    """
    summary_df = pd.read_sql(query_summary, conn)
    print("\n1. Overall Churn Summary:")
    print(summary_df.to_string(index=False))
    
    # Query 2: Churn by Customer Service Calls
    query_service = """
    SELECT 
        [Customer servicecalls], 
        COUNT(*) as Customer_Count,
        SUM(CASE WHEN Churn = 1 OR Churn = 'True' THEN 1 ELSE 0 END) as Churned_Count
    FROM customers
    GROUP BY [Customer servicecalls]
    ORDER BY [Customer servicecalls] DESC;
    """
    # Note: Adjust column name if your CSV uses 'Customer service calls' with a space
    try:
        service_df = pd.read_sql(query_service, conn)
        print("\n2. Churn Analysis by Customer Service Calls:")
        print(service_df.head(6).to_string(index=False))
    except Exception as e:
        print(f"Query note: {e}")
        
    conn.close()

if __name__ == "__main__":
    initialize_database()