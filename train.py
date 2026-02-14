import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. MANUALLY DEFINED MAPPINGS (Keep these exactly the same in app.py)
GENDER_MAP = {'M': 0, 'F': 1}
SYMPTOM_MAP = {'Chest Pain': 0, 'Fever': 1, 'Cough': 2}
HISTORY_MAP = {'Heart Disease': 0, 'Diabetes': 1, 'None': 2}

# 2. THE DATA (7 columns now: Age, Gender, BP, HR, Symptom, History, Risk)
data = [
    [80, 0, 190, 120, 0, 0, 'High'],   # Classic High
    [20, 1, 115, 70, 2, 2, 'Low'],     # Classic Low
    [45, 0, 145, 90, 1, 1, 'Medium'],  # Classic Medium
    [75, 1, 185, 110, 0, 0, 'High'],   # Another High
    [25, 0, 120, 75, 2, 2, 'Low']      # Another Low
]

# 3. THE COLUMNS (Added 'Risk_Level' to match the 7th item in 'data')
columns = ['Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 'Symptoms', 'Pre_Existing', 'Risk_Level']
train_df = pd.DataFrame(data, columns=columns)

# 4. SPLIT X and Y
X = train_df.drop('Risk_Level', axis=1) # Features (6 columns)
y = train_df['Risk_Level']              # Target (1 column)

# 5. TRAIN
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 6. SAVE
joblib.dump(model, 'triage_model.pkl')
print("SUCCESS: Model trained with perfect 7-column alignment!")