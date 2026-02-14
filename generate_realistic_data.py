import pandas as pd
import numpy as np
from datetime import datetime

"""
REALISTIC MEDICAL TRIAGE DATASET GENERATOR
===========================================
This creates a comprehensive dataset with proper medical correlations
that will train a high-confidence AI model.

Key Improvements:
1. Strong correlations between vitals and risk
2. Realistic vital sign ranges
3. Proper symptom-disease associations
4. Age-appropriate risk factors
5. 10,000 samples for robust training
"""

np.random.seed(42)  # Reproducible results

def generate_realistic_triage_data(n_samples=10000):
    """
    Generate medically accurate triage data with strong patterns
    """
    data = []
    
    # Define realistic ranges and correlations
    for i in range(n_samples):
        # First, decide the risk level (balanced distribution)
        risk_roll = np.random.rand()
        
        if risk_roll < 0.33:  # HIGH RISK - 33%
            # ============================================
            # HIGH RISK: Critical patients
            # ============================================
            
            # Age: Elderly more likely (60-95)
            age = int(np.random.normal(75, 10))
            age = np.clip(age, 55, 95)
            
            # Gender: Slight male bias for cardiac issues
            gender = 'M' if np.random.rand() < 0.6 else 'F'
            
            # Systolic BP: Hypertensive crisis range (160-220)
            bp_base = np.random.normal(180, 15)
            bp = int(np.clip(bp_base, 160, 220))
            
            # Heart Rate: Tachycardia (100-160)
            hr_base = np.random.normal(115, 15)
            hr = int(np.clip(hr_base, 95, 160))
            
            # Symptoms: Life-threatening
            symptoms = np.random.choice([
                'Chest Pain',           # 50% - Most critical
                'Shortness of Breath',  # 25%
                'Severe Headache',      # 15%
                'Confusion',            # 10%
            ], p=[0.50, 0.25, 0.15, 0.10])
            
            # Pre-existing: High-risk conditions
            pre_existing = np.random.choice([
                'Heart Disease',   # 60%
                'Diabetes',        # 30%
                'Hypertension',    # 10%
            ], p=[0.60, 0.30, 0.10])
            
            risk_level = 'High'
            
        elif risk_roll < 0.66:  # MEDIUM RISK - 33%
            # ============================================
            # MEDIUM RISK: Urgent but stable
            # ============================================
            
            # Age: Middle-aged (35-70)
            age = int(np.random.normal(52, 12))
            age = np.clip(age, 35, 75)
            
            # Gender: Balanced
            gender = 'M' if np.random.rand() < 0.5 else 'F'
            
            # Systolic BP: Elevated (130-165)
            bp_base = np.random.normal(145, 10)
            bp = int(np.clip(bp_base, 130, 165))
            
            # Heart Rate: Slightly elevated (85-105)
            hr_base = np.random.normal(92, 8)
            hr = int(np.clip(hr_base, 80, 110))
            
            # Symptoms: Concerning but not critical
            symptoms = np.random.choice([
                'Fever',              # 35%
                'Abdominal Pain',     # 30%
                'Dizziness',          # 20%
                'Nausea',             # 15%
            ], p=[0.35, 0.30, 0.20, 0.15])
            
            # Pre-existing: Mixed
            pre_existing = np.random.choice([
                'Diabetes',        # 40%
                'Hypertension',    # 35%
                'No_History',      # 25%
            ], p=[0.40, 0.35, 0.25])
            
            risk_level = 'Medium'
            
        else:  # LOW RISK - 34%
            # ============================================
            # LOW RISK: Stable, non-urgent
            # ============================================
            
            # Age: Younger (18-50)
            age = int(np.random.normal(32, 10))
            age = np.clip(age, 18, 55)
            
            # Gender: Balanced
            gender = 'M' if np.random.rand() < 0.5 else 'F'
            
            # Systolic BP: Normal (100-130)
            bp_base = np.random.normal(118, 8)
            bp = int(np.clip(bp_base, 100, 135))
            
            # Heart Rate: Normal (60-85)
            hr_base = np.random.normal(72, 8)
            hr = int(np.clip(hr_base, 55, 90))
            
            # Symptoms: Minor complaints
            symptoms = np.random.choice([
                'Cough',           # 30%
                'Sore Throat',     # 25%
                'Minor Injury',    # 20%
                'Rash',            # 15%
                'Headache',        # 10%
            ], p=[0.30, 0.25, 0.20, 0.15, 0.10])
            
            # Pre-existing: Mostly none
            pre_existing = np.random.choice([
                'No_History',  # 80%
                'Asthma',      # 15%
                'Allergies',   # 5%
            ], p=[0.80, 0.15, 0.05])
            
            risk_level = 'Low'
        
        # Add some realistic noise (5% of cases are edge cases)
        if np.random.rand() < 0.05:
            # Occasional young person with high risk
            if risk_level == 'High' and np.random.rand() < 0.3:
                age = np.random.randint(25, 40)
            # Occasional elderly with low risk
            elif risk_level == 'Low' and np.random.rand() < 0.3:
                age = np.random.randint(60, 75)
                bp = np.random.randint(125, 140)
        
        # Ensure values are in realistic bounds
        age = int(np.clip(age, 18, 95))
        bp = int(np.clip(bp, 80, 220))
        hr = int(np.clip(hr, 45, 180))
        
        data.append({
            'Age': age,
            'Gender': gender,
            'Systolic_BP': bp,
            'Heart_Rate': hr,
            'Symptoms': symptoms,
            'Pre_Existing': pre_existing,
            'Risk_Level': risk_level
        })
    
    df = pd.DataFrame(data)
    return df

# Generate the dataset
print("="*70)
print("GENERATING REALISTIC MEDICAL TRIAGE DATASET")
print("="*70)
print("\nCreating 10,000 patient records with proper medical correlations...")

df = generate_realistic_triage_data(10000)

# Display statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)

print("\nRisk Level Distribution:")
print(df['Risk_Level'].value_counts().sort_index())
print(f"\nPercentages:")
for level in ['High', 'Medium', 'Low']:
    pct = (df['Risk_Level'] == level).sum() / len(df) * 100
    print(f"  {level:6s}: {pct:.1f}%")

print("\n" + "-"*70)
print("Age Statistics by Risk Level:")
print("-"*70)
for level in ['High', 'Medium', 'Low']:
    ages = df[df['Risk_Level'] == level]['Age']
    print(f"{level:6s}: Mean={ages.mean():.1f}, Min={ages.min()}, Max={ages.max()}")

print("\n" + "-"*70)
print("Blood Pressure Statistics by Risk Level:")
print("-"*70)
for level in ['High', 'Medium', 'Low']:
    bp = df[df['Risk_Level'] == level]['Systolic_BP']
    print(f"{level:6s}: Mean={bp.mean():.1f}, Min={bp.min()}, Max={bp.max()}")

print("\n" + "-"*70)
print("Heart Rate Statistics by Risk Level:")
print("-"*70)
for level in ['High', 'Medium', 'Low']:
    hr = df[df['Risk_Level'] == level]['Heart_Rate']
    print(f"{level:6s}: Mean={hr.mean():.1f}, Min={hr.min()}, Max={hr.max()}")

print("\n" + "-"*70)
print("Symptom Distribution:")
print("-"*70)
print(df['Symptoms'].value_counts())

print("\n" + "-"*70)
print("Pre-Existing Conditions:")
print("-"*70)
print(df['Pre_Existing'].value_counts())

print("\n" + "-"*70)
print("Sample High-Risk Patients:")
print("-"*70)
print(df[df['Risk_Level'] == 'High'].head(3).to_string(index=False))

print("\n" + "-"*70)
print("Sample Low-Risk Patients:")
print("-"*70)
print(df[df['Risk_Level'] == 'Low'].head(3).to_string(index=False))

# Save the dataset
filename = 'realistic_triage_dataset.csv'
df.to_csv(filename, index=False)

print("\n" + "="*70)
print(f"✅ SUCCESS: Saved {len(df)} records to '{filename}'")
print("="*70)
print("\nKey Features of This Dataset:")
print("  ✓ Strong correlations between vitals and risk")
print("  ✓ Realistic age distributions per risk level")
print("  ✓ Proper symptom-disease associations")
print("  ✓ Medical accuracy in vital sign ranges")
print("  ✓ Balanced class distribution")
print("  ✓ 10,000 samples for robust model training")
print("\nNext Step: Run 'train_final.py' to train on this data!")
print("="*70)
