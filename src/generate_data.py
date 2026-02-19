import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_synthetic_data(n_samples=5000):
    """
    Generates synthetic lending data for credit risk modeling (Indian Context).
    Features:
    - income: Annual Income (INR)
    - loan_amount: Loan Amount (INR)
    - credit_score: CIBIL Score (300-900)
    - employment_length: Years of employment
    - dti: Debt-to-Income Ratio
    - age: Age of applicant
    - default: 0 (Paid), 1 (Default)
    """
    np.random.seed(42)
    
    # Generate base structure using make_classification for imbalanced data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5, # We will add derived features
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.97, 0.03], # 3% default rate (Realistic for Banks)
        random_state=42
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    
    # Transform features to realistic business metrics (Indian Market)
    
    # Income: Scaled f1 to 2.5 LPA - 30 LPA (Indian Rupees)
    # Norm: 3,00,000 - 30,00,000
    df['income'] = (df['f1'] - df['f1'].min()) / (df['f1'].max() - df['f1'].min()) * 2750000 + 250000
    df['income'] = df['income'].round(-3) # Round to nearest 1000
    
    # Loan Amount: Correlated with income but with variance
    # Personal loans usually 0.5x to 2x of annual income in India, sometimes less
    df['loan_amount'] = df['income'] * np.random.uniform(0.2, 0.8, n_samples)
    df['loan_amount'] = df['loan_amount'].round(-3) # Round to nearest 1000
    
    # Credit Score (CIBIL): Inverse correlation with default probability (f2)
    # Higher f2 -> Higher risk -> Lower score
    # CIBIL range 300-900
    df['credit_score'] = 900 - ((df['f2'] - df['f2'].min()) / (df['f2'].max() - df['f2'].min()) * 500)
    # Add noise
    df['credit_score'] += np.random.normal(0, 30, n_samples)
    df['credit_score'] = df['credit_score'].clip(300, 900).round(0)
    
    # Employment Length: 0-40 years
    df['employment_length'] = np.random.choice(range(0, 41), n_samples, p=[0.05] + [0.95/40]*40)
    
    # Debt-to-Income (DTI): Higher means riskier
    # Defaulting customers should have higher DTI on average
    df['dti'] = np.random.beta(2, 5, n_samples) * 45 # Base DTI
    # Increase DTI for defaulters (class 1)
    mask_default = y == 1
    df.loc[mask_default, 'dti'] += np.random.normal(15, 5, sum(mask_default))
    df['dti'] = df['dti'].clip(0, 100).round(2)
    
    # Age: 21-65 (Retirement age usually cap for loans)
    df['age'] = np.random.randint(21, 66, n_samples)
    
    # Assign Target
    df['loan_status'] = y # 0: Paid, 1: Default
    
    # Feature Cleanup
    features = ['income', 'loan_amount', 'credit_score', 'employment_length', 'dti', 'age', 'loan_status']
    final_df = df[features]
    
    return final_df

if __name__ == "__main__":
    print("Generating synthetic data (Indian Context)...")
    df = generate_synthetic_data()
    output_path = "data/lending_data.csv"
    # Ensure directory exists (handled by previous steps ideally, but safe check)
    import os
    os.makedirs("data", exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())
    print("\nClass Distribution:")
    print(df['loan_status'].value_counts(normalize=True))
