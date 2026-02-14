import pandas as pd
import numpy as np
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

# 1. CREATE THE EXPERT SEED (THE BLUEPRINT)
def generate_strong_seed(n=300):
    rows = []
    for _ in range(n):
        r = np.random.rand()
        if r < 0.33: # HIGH RISK: Old, High BP, Dangerous Symptoms
            rows.append([np.random.randint(60, 90), 'M', np.random.randint(165, 210), 
                         np.random.randint(100, 140), 'Chest Pain', 'Heart Disease', 'High'])
        elif r < 0.66: # MEDIUM RISK: Middle age, Elevated BP, General Symptoms
            rows.append([np.random.randint(35, 60), 'F', np.random.randint(135, 155), 
                         np.random.randint(85, 100), 'Fever', 'Diabetes', 'Medium'])
        else: # LOW RISK: Young, Normal BP, Minor Symptoms
            rows.append([np.random.randint(18, 35), 'M', np.random.randint(110, 125), 
                         np.random.randint(60, 80), 'Cough', 'None', 'Low'])
    
    df = pd.DataFrame(rows, columns=['Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 'Symptoms', 'Pre_Existing', 'Risk_Level'])
    df.to_csv('expert_seed.csv', index=False)
    return df

print("Step 1: Creating expert_seed.csv...")
seed_df = generate_strong_seed()

# 2. SETUP METADATA
metadata = Metadata.detect_from_dataframe(data=seed_df)

# 3. TRAINING WITH ENHANCED SETTINGS
# 'pac=10' helps the model group 'High Risk' rows together to see the trend
synthesizer = CTGANSynthesizer(
    metadata, 
    enforce_rounding=True, 
    epochs=1500, 
    pac=10 
)

print("Step 2: Training CTGAN (1500 Epochs)... This is the 90% push.")
synthesizer.fit(seed_df)

# 4. GENERATE 5000 ROWS
print("Step 3: Generating 5000 rows...")
synthetic_data = synthesizer.sample(num_rows=5000)

# 5. SAVE THE FINAL DATA
synthetic_data.to_csv('triage_v3_90plus.csv', index=False)
print("SUCCESS: 'triage_v3_90plus.csv' is ready!")