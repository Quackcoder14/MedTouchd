import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*20 + "MEDTOUCH.AI - FINAL TRAINING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD REALISTIC MEDICAL DATA
# ============================================================================
print("\n[1/7] Loading realistic medical triage dataset...")

try:
    df = pd.read_csv('realistic_triage_dataset.csv')
    print(f"✓ Loaded {len(df):,} patient records")
except FileNotFoundError:
    print("❌ ERROR: 'realistic_triage_dataset.csv' not found!")
    print("   Please run 'generate_realistic_data.py' first.")
    exit(1)

print(f"\nDataset shape: {df.shape}")
print(f"Risk distribution:")
for level in ['High', 'Medium', 'Low']:
    count = (df['Risk_Level'] == level).sum()
    pct = count / len(df) * 100
    print(f"  {level:6s}: {count:4d} ({pct:.1f}%)")

# ============================================================================
# STEP 2: CREATE AND SAVE LABEL ENCODERS
# ============================================================================
print("\n[2/7] Creating label encoders for categorical features...")

# Initialize encoders
le_gender = LabelEncoder()
le_symptoms = LabelEncoder()
le_pre = LabelEncoder()

# Fit encoders on the data
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['Symptoms_Encoded'] = le_symptoms.fit_transform(df['Symptoms'])
df['Pre_Existing_Encoded'] = le_pre.fit_transform(df['Pre_Existing'])

print("\n✓ Encoding mappings created:")
print(f"\n  Gender ({len(le_gender.classes_)} classes):")
for i, label in enumerate(le_gender.classes_):
    print(f"    {label} → {i}")

print(f"\n  Symptoms ({len(le_symptoms.classes_)} classes):")
for i, label in enumerate(le_symptoms.classes_):
    print(f"    {label} → {i}")

print(f"\n  Pre-Existing ({len(le_pre.classes_)} classes):")
for i, label in enumerate(le_pre.classes_):
    print(f"    {label} → {i}")

# ============================================================================
# STEP 3: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[3/7] Preparing features and target variable...")

# Select features in exact order
feature_cols = ['Age', 'Gender_Encoded', 'Systolic_BP', 'Heart_Rate', 
                'Symptoms_Encoded', 'Pre_Existing_Encoded']
X = df[feature_cols]
y = df['Risk_Level']

print(f"✓ Features: {feature_cols}")
print(f"✓ Target: Risk_Level (3 classes: High, Medium, Low)")
print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")

# ============================================================================
# STEP 4: SPLIT DATA WITH STRATIFICATION
# ============================================================================
print("\n[4/7] Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintains class balance
)

print(f"✓ Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nTraining set distribution:")
for level in ['High', 'Medium', 'Low']:
    count = (y_train == level).sum()
    pct = count / len(y_train) * 100
    print(f"  {level:6s}: {count:4d} ({pct:.1f}%)")

# ============================================================================
# STEP 5: TRAIN OPTIMIZED RANDOM FOREST
# ============================================================================
print("\n[5/7] Training Random Forest Classifier...")
print("Configuration:")
print("  - n_estimators: 300 (more trees = better confidence)")
print("  - max_depth: 15 (deeper trees for complex patterns)")
print("  - min_samples_split: 5 (finer decision boundaries)")
print("  - min_samples_leaf: 2 (more precise leaf nodes)")
print("  - class_weight: balanced (handle any imbalance)")
print("  - random_state: 42 (reproducible results)")

model = RandomForestClassifier(
    n_estimators=300,          # More trees = higher confidence
    max_depth=15,              # Deeper trees capture more patterns
    min_samples_split=5,       # More fine-grained splits
    min_samples_leaf=2,        # Smaller leaf nodes
    class_weight='balanced',   # Handle class imbalance
    random_state=42,
    n_jobs=-1                  # Use all CPU cores
)

print("\nTraining in progress...")
model.fit(X_train, y_train)
print("✓ Training complete!")

# ============================================================================
# STEP 6: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n[6/7] Evaluating model performance...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*80}")
print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*80}")

# Detailed classification report
print("\nDetailed Classification Report:")
print("-"*80)
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
print("Confusion Matrix:")
print("-"*80)
cm = confusion_matrix(y_test, y_pred)
print("                Predicted")
print("              High  Medium  Low")
print(f"Actual High   {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
print(f"       Medium {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
print(f"       Low    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")

# Cross-validation
print("\nCross-Validation (5-fold):")
print("-"*80)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
print(f"Individual folds: {[f'{s*100:.2f}%' for s in cv_scores]}")

# Feature Importance
print("\nFeature Importance:")
print("-"*80)
feature_names = ['Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 'Symptoms', 'Pre_Existing']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]:15s}: {importances[idx]*100:.2f}%")

# Confidence Analysis
print("\nConfidence Score Analysis:")
print("-"*80)
max_probas = y_pred_proba.max(axis=1)
for level in ['High', 'Medium', 'Low']:
    mask = y_pred == level
    if mask.sum() > 0:
        level_probs = max_probas[mask]
        print(f"{level:6s} predictions:")
        print(f"  Mean confidence: {level_probs.mean()*100:.1f}%")
        print(f"  Min confidence:  {level_probs.min()*100:.1f}%")
        print(f"  Max confidence:  {level_probs.max()*100:.1f}%")
        print(f"  >90% confident:  {(level_probs > 0.90).sum()} / {len(level_probs)} ({(level_probs > 0.90).sum()/len(level_probs)*100:.1f}%)")

# ============================================================================
# STEP 7: CRITICAL TEST CASES VALIDATION
# ============================================================================
print("\n" + "="*80)
print("CRITICAL TEST CASES VALIDATION")
print("="*80)

test_cases = [
    {
        'name': 'CRITICAL EMERGENCY: Elderly cardiac event',
        'input': {
            'Age': 82,
            'Gender': 'M',
            'Systolic_BP': 195,
            'Heart_Rate': 125,
            'Symptoms': 'Chest Pain',
            'Pre_Existing': 'Heart Disease'
        },
        'expected': 'High'
    },
    {
        'name': 'SEVERE: Elderly with respiratory distress',
        'input': {
            'Age': 78,
            'Gender': 'F',
            'Systolic_BP': 175,
            'Heart_Rate': 118,
            'Symptoms': 'Shortness of Breath',
            'Pre_Existing': 'Heart Disease'
        },
        'expected': 'High'
    },
    {
        'name': 'URGENT: Middle-aged with concerning symptoms',
        'input': {
            'Age': 52,
            'Gender': 'M',
            'Systolic_BP': 148,
            'Heart_Rate': 95,
            'Symptoms': 'Fever',
            'Pre_Existing': 'Diabetes'
        },
        'expected': 'Medium'
    },
    {
        'name': 'MODERATE: Young adult with pain',
        'input': {
            'Age': 38,
            'Gender': 'F',
            'Systolic_BP': 142,
            'Heart_Rate': 88,
            'Symptoms': 'Abdominal Pain',
            'Pre_Existing': 'No_History'
        },
        'expected': 'Medium'
    },
    {
        'name': 'STABLE: Young healthy patient',
        'input': {
            'Age': 24,
            'Gender': 'F',
            'Systolic_BP': 112,
            'Heart_Rate': 68,
            'Symptoms': 'Cough',
            'Pre_Existing': 'No_History'
        },
        'expected': 'Low'
    },
    {
        'name': 'STABLE: Minor complaint',
        'input': {
            'Age': 31,
            'Gender': 'M',
            'Systolic_BP': 118,
            'Heart_Rate': 72,
            'Symptoms': 'Sore Throat',
            'Pre_Existing': 'No_History'
        },
        'expected': 'Low'
    }
]

all_passed = True
for i, case in enumerate(test_cases, 1):
    print(f"\nTest {i}/6: {case['name']}")
    print("-"*80)
    
    # Prepare input
    input_data = case['input']
    
    # Encode categorical variables
    try:
        gender_enc = le_gender.transform([input_data['Gender']])[0]
        symptom_enc = le_symptoms.transform([input_data['Symptoms']])[0]
        pre_enc = le_pre.transform([input_data['Pre_Existing']])[0]
    except ValueError as e:
        print(f"❌ ENCODING ERROR: {e}")
        all_passed = False
        continue
    
    # Create feature vector
    features = pd.DataFrame([[
        input_data['Age'],
        gender_enc,
        input_data['Systolic_BP'],
        input_data['Heart_Rate'],
        symptom_enc,
        pre_enc
    ]], columns=feature_cols)
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get confidence for predicted class
    class_idx = list(model.classes_).index(prediction)
    confidence = probabilities[class_idx] * 100
    
    # Display all probabilities
    prob_dict = {cls: prob*100 for cls, prob in zip(model.classes_, probabilities)}
    
    # Check result
    status = "✅ PASS" if prediction == case['expected'] else "❌ FAIL"
    if prediction != case['expected']:
        all_passed = False
    
    print(f"Input: Age {input_data['Age']}, {input_data['Gender']}, "
          f"BP {input_data['Systolic_BP']}, HR {input_data['Heart_Rate']}")
    print(f"       {input_data['Symptoms']} | {input_data['Pre_Existing']}")
    print(f"\nPredicted: {prediction} (Confidence: {confidence:.1f}%)")
    print(f"Expected:  {case['expected']}")
    print(f"\nAll probabilities:")
    for risk_level in ['High', 'Medium', 'Low']:
        prob = prob_dict.get(risk_level, 0)
        bar = '█' * int(prob/2)
        print(f"  {risk_level:6s}: {prob:5.1f}% {bar}")
    print(f"\n{status}")

# ============================================================================
# STEP 8: SAVE MODEL AND ENCODERS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL AND ENCODERS")
print("="*80)

joblib.dump(model, 'triage_model.pkl')
print("✓ Saved: triage_model.pkl")

joblib.dump(le_gender, 'le_gender.pkl')
print("✓ Saved: le_gender.pkl")

joblib.dump(le_symptoms, 'le_symptoms.pkl')
print("✓ Saved: le_symptoms.pkl")

joblib.dump(le_pre, 'le_pre.pkl')
print("✓ Saved: le_pre.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*80)
print(f"✓ Model Accuracy: {accuracy*100:.2f}%")
print(f"✓ Training Samples: {len(X_train):,}")
print(f"✓ Test Samples: {len(X_test):,}")
print(f"✓ Feature Count: {X.shape[1]}")
print(f"✓ Classes: {len(model.classes_)} (High, Medium, Low)")
print(f"✓ Test Cases: {'All Passed ✅' if all_passed else 'Some Failed ❌'}")
print(f"✓ Files Saved: 4 (.pkl files)")
print("\n" + "="*80)
print("🚀 READY FOR DEPLOYMENT!")
print("="*80)
print("\nNext step: Run 'streamlit run app.py' to launch the application")
print("="*80)
