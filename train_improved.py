import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("="*60)
print("MEDTOUCH.AI - IMPROVED TRAINING PIPELINE")
print("="*60)

# 1. LOAD HIGH-QUALITY SYNTHETIC DATA
try:
    df = pd.read_csv('triage_v3_90plus.csv')
    # Fill NaN values with 'None' for consistency
    df['Pre_Existing'] = df['Pre_Existing'].fillna('None')
    print(f"\n[OK] Loaded {len(df)} training samples")
    print(f"\nRisk Distribution:")
    print(df['Risk_Level'].value_counts())
except FileNotFoundError:
    print("[ERROR] 'triage_v3_90plus.csv' not found. Run your data generation script first!")
    exit(1)

# 2. CREATE LABEL ENCODERS
le_gender = LabelEncoder()
le_symptoms = LabelEncoder()
le_pre = LabelEncoder()

# Fit encoders
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['Symptoms_Encoded'] = le_symptoms.fit_transform(df['Symptoms'])
df['Pre_Existing_Encoded'] = le_pre.fit_transform(df['Pre_Existing'])

# Print mapping for verification
print("\n" + "="*60)
print("ENCODING MAPPINGS")
print("="*60)
print("\nGender Mapping:", dict(zip(le_gender.classes_, range(len(le_gender.classes_)))))
print("Symptoms Mapping:", dict(zip(le_symptoms.classes_, range(len(le_symptoms.classes_)))))
print("Pre-Existing Mapping:", dict(zip(le_pre.classes_, range(len(le_pre.classes_)))))

# 3. PREPARE FEATURES AND TARGET
# Note: Use the exact names used in the app later
feature_cols = ['Age', 'Gender_Encoded', 'Systolic_BP', 'Heart_Rate', 'Symptoms_Encoded', 'Pre_Existing_Encoded']
X = df[feature_cols]
y = df['Risk_Level']

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[OK] Training set: {len(X_train)} samples")
print(f"[OK] Test set: {len(X_test)} samples")

# 5. TRAIN RANDOM FOREST
print("\n" + "="*60)
print("TRAINING MODEL...")
print("="*60)

model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)
print("\n[OK] Model training complete!")

# 6. EVALUATE PERFORMANCE
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. TEST WITH EXAMPLE CASES
print("\n" + "="*60)
print("VALIDATION: Testing Critical Cases")
print("="*60)

test_cases = [
    {'name': 'CRITICAL: High BP', 'data': [80, 'M', 190, 120, 'Chest Pain', 'Heart Disease'], 'expected': 'High'},
    {'name': 'STABLE: Healthy', 'data': [25, 'F', 115, 70, 'Cough', 'None'], 'expected': 'Low'}
]

for case in test_cases:
    # Transform test case using fitted encoders
    d = case['data']
    encoded_data = [
        d[0], 
        le_gender.transform([d[1]])[0], 
        d[2], 
        d[3], 
        le_symptoms.transform([d[4]])[0], 
        le_pre.transform([d[5]])[0]
    ]
    
    test_df = pd.DataFrame([encoded_data], columns=feature_cols)
    prediction = model.predict(test_df)[0]
    probs = model.predict_proba(test_df)[0]
    confidence = max(probs) * 100
    
    res_mark = "[PASS]" if prediction == case['expected'] else "[FAIL]"
    print(f"{res_mark} {case['name']} -> Predicted: {prediction} ({confidence:.1f}%)")

# 8. SAVE ASSETS
joblib.dump(model, 'triage_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_symptoms, 'le_symptoms.pkl')
joblib.dump(le_pre, 'le_pre.pkl')

print("\n" + "="*60)
print("ALL ASSETS SAVED - Ready for Streamlit!")
print("="*60)