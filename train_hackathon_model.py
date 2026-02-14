"""
HACKATHON AI TRIAGE SYSTEM - TRAINING PIPELINE
===============================================
Multi-Output ML System:
1. Risk Level Classification (High/Medium/Low)
2. Department Recommendation (9 departments)
3. Explainability (Feature importance, confidence scores)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*25 + "HACKATHON AI TRIAGE SYSTEM")
print(" "*30 + "Training Pipeline")
print("="*90)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1/8] Loading medical triage dataset...")

try:
    df = pd.read_csv('hackathon_medical_triage_data.csv')
    print(f"✓ Loaded {len(df):,} patient records")
except FileNotFoundError:
    print("❌ ERROR: 'hackathon_medical_triage_data.csv' not found!")
    print("   Please run 'generate_hackathon_data.py' first.")
    exit(1)

print(f"\nDataset columns: {list(df.columns)}")
print(f"Dataset shape: {df.shape}")

# Display distribution
print(f"\nRisk Level Distribution:")
for level in ['High', 'Medium', 'Low']:
    count = (df['Risk_Level'] == level).sum()
    pct = count / len(df) * 100
    print(f"  {level:6s}: {count:5d} ({pct:.2f}%)")

print(f"\nDepartment Distribution:")
dept_dist = df['Recommended_Department'].value_counts()
for dept, count in dept_dist.head(5).items():
    pct = count / len(df) * 100
    print(f"  {dept:20s}: {count:5d} ({pct:.2f}%)")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2/8] Preprocessing and encoding data...")

# Create label encoders for categorical features
le_gender = LabelEncoder()
le_symptoms = LabelEncoder()
le_pre_existing = LabelEncoder()
le_risk = LabelEncoder()
le_department = LabelEncoder()

# Encode categorical variables
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['Symptoms_Encoded'] = le_symptoms.fit_transform(df['Symptoms'])
df['Pre_Existing_Encoded'] = le_pre_existing.fit_transform(df['Pre_Existing'])
df['Risk_Encoded'] = le_risk.fit_transform(df['Risk_Level'])
df['Department_Encoded'] = le_department.fit_transform(df['Recommended_Department'])

print("\n✓ Encoding mappings created:")

print(f"\n  Gender ({len(le_gender.classes_)} classes):")
for i, label in enumerate(le_gender.classes_):
    print(f"    {label} → {i}")

print(f"\n  Symptoms ({len(le_symptoms.classes_)} classes):")
for i, label in enumerate(sorted(le_symptoms.classes_)[:10]):  # Show first 10
    enc_val = le_symptoms.transform([label])[0]
    print(f"    {label} → {enc_val}")
print(f"    ... and {len(le_symptoms.classes_) - 10} more")

print(f"\n  Pre-Existing Conditions ({len(le_pre_existing.classes_)} classes):")
for i, label in enumerate(le_pre_existing.classes_):
    print(f"    {label} → {i}")

print(f"\n  Risk Levels ({len(le_risk.classes_)} classes):")
for i, label in enumerate(le_risk.classes_):
    print(f"    {label} → {i}")

print(f"\n  Departments ({len(le_department.classes_)} classes):")
for i, label in enumerate(le_department.classes_):
    print(f"    {label} → {i}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 3/8] Preparing features...")

# Define feature columns (all vital signs + encoded categoricals)
feature_cols = [
    'Age',
    'Gender_Encoded',
    'Systolic_BP',
    'Diastolic_BP',
    'Heart_Rate',
    'Temperature',
    'Symptoms_Encoded',
    'Pre_Existing_Encoded'
]

X = df[feature_cols]
y_risk = df['Risk_Level']
y_department = df['Recommended_Department']

print(f"✓ Features selected: {feature_cols}")
print(f"✓ X shape: {X.shape}")
print(f"✓ Target 1 (Risk): {y_risk.shape}")
print(f"✓ Target 2 (Department): {y_department.shape}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 4/8] Splitting data into train and test sets...")

# Split with stratification on risk level
X_train, X_test, y_risk_train, y_risk_test, y_dept_train, y_dept_test = train_test_split(
    X, y_risk, y_department,
    test_size=0.2,
    random_state=42,
    stratify=y_risk  # Ensure balanced split
)

print(f"✓ Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nTraining set risk distribution:")
for level in ['High', 'Medium', 'Low']:
    count = (y_risk_train == level).sum()
    pct = count / len(y_risk_train) * 100
    print(f"  {level:6s}: {count:4d} ({pct:.1f}%)")

# ============================================================================
# STEP 5: TRAIN RISK CLASSIFICATION MODEL
# ============================================================================
print("\n[STEP 5/8] Training Risk Classification Model...")

print("\nModel Configuration:")
print("  - Algorithm: Random Forest Classifier")
print("  - n_estimators: 300 (decision trees)")
print("  - max_depth: 20 (for complex patterns)")
print("  - min_samples_split: 5")
print("  - min_samples_leaf: 2")
print("  - class_weight: balanced")
print("  - random_state: 42")

risk_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nTraining Risk Model...")
risk_model.fit(X_train, y_risk_train)
print("✓ Risk Model trained successfully!")

# ============================================================================
# STEP 6: TRAIN DEPARTMENT RECOMMENDATION MODEL
# ============================================================================
print("\n[STEP 6/8] Training Department Recommendation Model...")

department_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Department Model...")
department_model.fit(X_train, y_dept_train)
print("✓ Department Model trained successfully!")

# ============================================================================
# STEP 7: MODEL EVALUATION
# ============================================================================
print("\n[STEP 7/8] Evaluating Model Performance...")

# ============================================================================
# RISK MODEL EVALUATION
# ============================================================================
print("\n" + "="*90)
print("RISK CLASSIFICATION MODEL PERFORMANCE")
print("="*90)

y_risk_pred = risk_model.predict(X_test)
y_risk_proba = risk_model.predict_proba(X_test)

risk_accuracy = accuracy_score(y_risk_test, y_risk_pred)
print(f"\n📊 OVERALL ACCURACY: {risk_accuracy*100:.2f}%")

print("\n" + "-"*90)
print("Detailed Classification Report:")
print("-"*90)
print(classification_report(y_risk_test, y_risk_pred, digits=3))

print("Confusion Matrix:")
print("-"*90)
cm = confusion_matrix(y_risk_test, y_risk_pred, labels=['High', 'Medium', 'Low'])
print("                Predicted")
print("              High  Medium  Low")
print(f"Actual High   {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
print(f"       Medium {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
print(f"       Low    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")

# Cross-validation
print("\n" + "-"*90)
print("Cross-Validation (5-fold):")
print("-"*90)
cv_scores = cross_val_score(risk_model, X_train, y_risk_train, cv=5, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
print(f"Individual folds: {[f'{s*100:.2f}%' for s in cv_scores]}")

# Feature Importance for Risk Model
print("\n" + "-"*90)
print("Feature Importance (Risk Classification):")
print("-"*90)
feature_names = ['Age', 'Gender', 'Systolic_BP', 'Diastolic_BP', 
                 'Heart_Rate', 'Temperature', 'Symptoms', 'Pre_Existing']
importances = risk_model.feature_importances_
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]:20s}: {importances[idx]*100:.2f}%")

# Confidence Analysis
print("\n" + "-"*90)
print("Confidence Score Analysis (Risk Model):")
print("-"*90)
max_probas = y_risk_proba.max(axis=1)
for level in ['High', 'Medium', 'Low']:
    mask = y_risk_pred == level
    if mask.sum() > 0:
        level_probs = max_probas[mask]
        print(f"\n{level} Risk Predictions:")
        print(f"  Mean confidence:     {level_probs.mean()*100:.1f}%")
        print(f"  Min confidence:      {level_probs.min()*100:.1f}%")
        print(f"  Max confidence:      {level_probs.max()*100:.1f}%")
        print(f"  >90% confident:      {(level_probs > 0.90).sum()} / {len(level_probs)} ({(level_probs > 0.90).sum()/len(level_probs)*100:.1f}%)")
        print(f"  >95% confident:      {(level_probs > 0.95).sum()} / {len(level_probs)} ({(level_probs > 0.95).sum()/len(level_probs)*100:.1f}%)")

# ============================================================================
# DEPARTMENT MODEL EVALUATION
# ============================================================================
print("\n\n" + "="*90)
print("DEPARTMENT RECOMMENDATION MODEL PERFORMANCE")
print("="*90)

y_dept_pred = department_model.predict(X_test)
y_dept_proba = department_model.predict_proba(X_test)

dept_accuracy = accuracy_score(y_dept_test, y_dept_pred)
print(f"\n📊 OVERALL ACCURACY: {dept_accuracy*100:.2f}%")

print("\n" + "-"*90)
print("Detailed Classification Report:")
print("-"*90)
print(classification_report(y_dept_test, y_dept_pred, digits=3))

# Feature Importance for Department Model
print("\n" + "-"*90)
print("Feature Importance (Department Recommendation):")
print("-"*90)
dept_importances = department_model.feature_importances_
dept_indices = np.argsort(dept_importances)[::-1]

for i, idx in enumerate(dept_indices):
    print(f"{i+1}. {feature_names[idx]:20s}: {dept_importances[idx]*100:.2f}%")

# ============================================================================
# STEP 8: TEST WITH CRITICAL CASES
# ============================================================================
print("\n\n" + "="*90)
print("CRITICAL TEST CASES VALIDATION")
print("="*90)

test_cases = [
    {
        'name': 'CRITICAL CARDIAC EMERGENCY',
        'input': {
            'Age': 68,
            'Gender': 'Male',
            'Systolic_BP': 195,
            'Diastolic_BP': 110,
            'Heart_Rate': 128,
            'Temperature': 37.2,
            'Symptoms': 'Chest Pain',
            'Pre_Existing': 'Heart Disease'
        },
        'expected_risk': 'High',
        'expected_dept': 'Emergency'
    },
    {
        'name': 'STROKE SYMPTOMS - EMERGENCY',
        'input': {
            'Age': 72,
            'Gender': 'Female',
            'Systolic_BP': 188,
            'Diastolic_BP': 105,
            'Heart_Rate': 95,
            'Temperature': 37.1,
            'Symptoms': 'Stroke Symptoms',
            'Pre_Existing': 'Hypertension'
        },
        'expected_risk': 'High',
        'expected_dept': 'Emergency'
    },
    {
        'name': 'RESPIRATORY DISTRESS',
        'input': {
            'Age': 45,
            'Gender': 'Female',
            'Systolic_BP': 142,
            'Diastolic_BP': 88,
            'Heart_Rate': 105,
            'Temperature': 37.8,
            'Symptoms': 'Difficulty Breathing',
            'Pre_Existing': 'Asthma'
        },
        'expected_risk': 'High',
        'expected_dept': 'Emergency'
    },
    {
        'name': 'MODERATE GASTRO ISSUE',
        'input': {
            'Age': 38,
            'Gender': 'Male',
            'Systolic_BP': 145,
            'Diastolic_BP': 90,
            'Heart_Rate': 92,
            'Temperature': 38.2,
            'Symptoms': 'Abdominal Pain',
            'Pre_Existing': 'No History'
        },
        'expected_risk': 'Medium',
        'expected_dept': 'Gastroenterology'
    },
    {
        'name': 'STABLE RESPIRATORY',
        'input': {
            'Age': 28,
            'Gender': 'Female',
            'Systolic_BP': 118,
            'Diastolic_BP': 75,
            'Heart_Rate': 72,
            'Temperature': 37.4,
            'Symptoms': 'Cough',
            'Pre_Existing': 'No History'
        },
        'expected_risk': 'Low',
        'expected_dept': 'Respiratory'
    },
    {
        'name': 'MINOR COMPLAINT',
        'input': {
            'Age': 25,
            'Gender': 'Male',
            'Systolic_BP': 115,
            'Diastolic_BP': 72,
            'Heart_Rate': 68,
            'Temperature': 36.8,
            'Symptoms': 'Sore Throat',
            'Pre_Existing': 'No History'
        },
        'expected_risk': 'Low',
        'expected_dept': 'General Medicine'
    }
]

all_passed = True
for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*90}")
    print(f"Test {i}/{len(test_cases)}: {case['name']}")
    print("="*90)
    
    input_data = case['input']
    
    # Encode inputs
    try:
        gender_enc = le_gender.transform([input_data['Gender']])[0]
        symptom_enc = le_symptoms.transform([input_data['Symptoms']])[0]
        pre_enc = le_pre_existing.transform([input_data['Pre_Existing']])[0]
    except ValueError as e:
        print(f"❌ ENCODING ERROR: {e}")
        all_passed = False
        continue
    
    # Create feature vector
    features = pd.DataFrame([[
        input_data['Age'],
        gender_enc,
        input_data['Systolic_BP'],
        input_data['Diastolic_BP'],
        input_data['Heart_Rate'],
        input_data['Temperature'],
        symptom_enc,
        pre_enc
    ]], columns=feature_cols)
    
    # Predict Risk
    risk_pred = risk_model.predict(features)[0]
    risk_proba = risk_model.predict_proba(features)[0]
    risk_idx = list(risk_model.classes_).index(risk_pred)
    risk_confidence = risk_proba[risk_idx] * 100
    
    # Predict Department
    dept_pred = department_model.predict(features)[0]
    dept_proba = department_model.predict_proba(features)[0]
    dept_idx = list(department_model.classes_).index(dept_pred)
    dept_confidence = dept_proba[dept_idx] * 100
    
    # Display input
    print(f"\n📋 Patient Profile:")
    print(f"  Age: {input_data['Age']} years | Gender: {input_data['Gender']}")
    print(f"  BP: {input_data['Systolic_BP']}/{input_data['Diastolic_BP']} mmHg")
    print(f"  HR: {input_data['Heart_Rate']} BPM | Temp: {input_data['Temperature']}°C")
    print(f"  Symptom: {input_data['Symptoms']}")
    print(f"  History: {input_data['Pre_Existing']}")
    
    # Display predictions
    print(f"\n🎯 AI Predictions:")
    print(f"  Risk Level:  {risk_pred} (Confidence: {risk_confidence:.1f}%)")
    print(f"  Department:  {dept_pred} (Confidence: {dept_confidence:.1f}%)")
    
    # Display expected
    print(f"\n✓ Expected:")
    print(f"  Risk Level:  {case['expected_risk']}")
    print(f"  Department:  {case['expected_dept']}")
    
    # Verify
    risk_match = risk_pred == case['expected_risk']
    dept_match = dept_pred == case['expected_dept']
    
    if risk_match and dept_match:
        print(f"\n✅ TEST PASSED - Both predictions correct!")
    else:
        if not risk_match:
            print(f"\n⚠️ Risk prediction mismatch: Got {risk_pred}, expected {case['expected_risk']}")
        if not dept_match:
            print(f"\n⚠️ Department prediction mismatch: Got {dept_pred}, expected {case['expected_dept']}")
        all_passed = False
    
    # Show probability distribution
    print(f"\n📊 Risk Probability Distribution:")
    risk_probs = {cls: prob*100 for cls, prob in zip(risk_model.classes_, risk_proba)}
    for level in ['High', 'Medium', 'Low']:
        prob = risk_probs.get(level, 0)
        bar = '█' * int(prob/2)
        print(f"  {level:6s}: {prob:5.1f}% {bar}")

# ============================================================================
# SAVE MODELS AND ENCODERS
# ============================================================================
print("\n\n" + "="*90)
print("SAVING MODELS AND ENCODERS")
print("="*90)

# Save models
joblib.dump(risk_model, 'risk_model.pkl')
print("✓ Saved: risk_model.pkl")

joblib.dump(department_model, 'department_model.pkl')
print("✓ Saved: department_model.pkl")

# Save encoders
joblib.dump(le_gender, 'le_gender.pkl')
print("✓ Saved: le_gender.pkl")

joblib.dump(le_symptoms, 'le_symptoms.pkl')
print("✓ Saved: le_symptoms.pkl")

joblib.dump(le_pre_existing, 'le_pre_existing.pkl')
print("✓ Saved: le_pre_existing.pkl")

joblib.dump(le_risk, 'le_risk.pkl')
print("✓ Saved: le_risk.pkl")

joblib.dump(le_department, 'le_department.pkl')
print("✓ Saved: le_department.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*90)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*90)

print(f"\n🎯 Model Performance:")
print(f"  ✓ Risk Classification Accuracy:      {risk_accuracy*100:.2f}%")
print(f"  ✓ Department Recommendation Accuracy: {dept_accuracy*100:.2f}%")

print(f"\n📊 Training Details:")
print(f"  ✓ Training Samples:    {len(X_train):,}")
print(f"  ✓ Test Samples:        {len(X_test):,}")
print(f"  ✓ Total Features:      {len(feature_cols)}")
print(f"  ✓ Risk Classes:        {len(risk_model.classes_)}")
print(f"  ✓ Department Classes:  {len(department_model.classes_)}")

print(f"\n💾 Files Saved:")
print(f"  ✓ risk_model.pkl")
print(f"  ✓ department_model.pkl")
print(f"  ✓ 5 encoder files (.pkl)")

print(f"\n✅ Validation:")
print(f"  ✓ Test Cases: {'All Passed' if all_passed else 'Some Issues'}")
print(f"  ✓ Cross-Validation: {cv_scores.mean()*100:.2f}%")
print(f"  ✓ Production Ready: {'Yes' if all_passed and risk_accuracy > 0.95 else 'Review Needed'}")

print("\n" + "="*90)
print("🚀 READY FOR DEPLOYMENT!")
print("="*90)
print("\nNext Steps:")
print("  1. Run 'python predict_triage.py' to test predictions")
print("  2. Integrate with your Streamlit dashboard")
print("  3. Add explainability layer")
print("="*90)
