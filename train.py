import pandas as pd
import numpy as np
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. CREATE SHARPER SEED DATA (EXAGGERATED CONTRAST)
def generate_sharp_seed(n=400):
    rows = []
    for _ in range(n):
        r = np.random.rand()
        if r < 0.33: # HIGH: Must be very high BP/Age/Pain
            rows.append([np.random.randint(65, 95), 'M', np.random.randint(175, 220), 
                         np.random.randint(110, 150), 'Chest Pain', 'Heart Disease', 'High'])
        elif r < 0.66: # MEDIUM: Moderate symptoms
            rows.append([np.random.randint(35, 60), 'F', np.random.randint(135, 150), 
                         np.random.randint(85, 100), 'Fever', 'Diabetes', 'Medium'])
        else: # LOW: Healthy vitals
            rows.append([np.random.randint(18, 30), 'M', np.random.randint(110, 125), 
                         np.random.randint(60, 80), 'Cough', 'None', 'Low'])
    
    return pd.DataFrame(rows, columns=['Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 'Symptoms', 'Pre_Existing', 'Risk_Level'])

seed_df = generate_sharp_seed()
seed_df.to_csv('expert_seed.csv', index=False)

# 2. TRAIN ENCODERS
le_gender = LabelEncoder().fit(seed_df['Gender'])
le_symptoms = LabelEncoder().fit(seed_df['Symptoms'])
le_pre = LabelEncoder().fit(seed_df['Pre_Existing'])

# 3. PREPARE TRAINING DATA
train_df = seed_df.copy()
train_df['Gender'] = le_gender.transform(train_df['Gender'])
train_df['Symptoms'] = le_symptoms.transform(train_df['Symptoms'])
train_df['Pre_Existing'] = le_pre.transform(train_df['Pre_Existing'])

X = train_df.drop('Risk_Level', axis=1)
y = train_df['Risk_Level']

# 4. TRAIN MODEL (The "Confidence" Fix)
# We set max_depth to allow the trees to be more decisive
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# 5. SAVE EVERYTHING
joblib.dump(model, 'triage_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_symptoms, 'le_symptoms.pkl')
joblib.dump(le_pre, 'le_pre.pkl')

print("SUCCESS: Model retrained with high-contrast data. Test it in Streamlit now!")