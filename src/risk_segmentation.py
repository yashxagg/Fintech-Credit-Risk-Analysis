import pandas as pd
import numpy as np

def run_risk_segmentation():
    """
    Simulates SQL Window Functions using Pandas for Risk Segmentation.
    In a production enviroment, this would likely be:
    
    SELECT 
        loan_id,
        loan_amount,
        income,
        NTILE(4) OVER (ORDER BY loan_amount DESC) as loan_quartile,
        NTILE(4) OVER (ORDER BY income ASC) as income_quartile,
        CASE 
            WHEN loan_quartile = 1 AND income_quartile = 1 THEN 'High Risk'
            WHEN loan_quartile = 4 AND income_quartile = 4 THEN 'Low Risk'
            ELSE 'Medium Risk'
        END as risk_segment
    FROM applications;
    """
    print("Running Risk Segmentation (simulating SQL Window Functions)...")
    try:
        df = pd.read_csv("data/lending_data.csv")
    except FileNotFoundError:
        print("Data not found.")
        return None

    # 1. Calculate Risk Segment based on Loan-to-Income Ratio
    df['loan_to_income'] = df['loan_amount'] / df['income']
    
    # Simulate NTILE(3) OVER (ORDER BY loan_to_income DESC)
    # Using pandas qcut for quantile binning
    df['risk_segment_id'] = pd.qcut(df['loan_to_income'], 3, labels=[3, 2, 1]) 
    
    # Map to descriptive Strings
    segment_map = {1: 'High Risk (High LTI)', 2: 'Medium Risk', 3: 'Low Risk (Low LTI)'}
    df['risk_segment'] = df['risk_segment_id'].map(segment_map)
    
    # Calculate average defaults per segment
    segment_stats = df.groupby('risk_segment')['loan_status'].mean().reset_index()
    segment_stats.columns = ['Risk Segment', 'Default Rate']
    
    return segment_stats

if __name__ == "__main__":
    stats = run_risk_segmentation()
    print(stats)
