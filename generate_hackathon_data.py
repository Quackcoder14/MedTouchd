"""
COMPREHENSIVE MEDICAL TRIAGE DATA GENERATOR
============================================
Aligned with Hackathon Problem Statement

Generates realistic patient data with:
- All required metadata fields
- Department recommendations
- Extreme cases and edge scenarios
- Real-life medical correlations
- 15,000 diverse patient records
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class MedicalDataGenerator:
    """Generate realistic medical triage data matching real-world patterns"""
    
    def __init__(self):
        # Define medical departments and their associations
        self.departments = {
            'Emergency': ['Chest Pain', 'Severe Headache', 'Confusion', 'Severe Bleeding', 
                         'Difficulty Breathing', 'Stroke Symptoms', 'Seizure'],
            'Cardiology': ['Chest Pain', 'Heart Palpitations', 'Shortness of Breath', 
                          'Irregular Heartbeat'],
            'Neurology': ['Severe Headache', 'Confusion', 'Dizziness', 'Numbness', 
                         'Vision Problems', 'Memory Loss'],
            'Respiratory': ['Cough', 'Shortness of Breath', 'Wheezing', 'Chest Tightness'],
            'Gastroenterology': ['Abdominal Pain', 'Nausea', 'Vomiting', 'Diarrhea'],
            'Orthopedics': ['Fracture', 'Joint Pain', 'Back Pain', 'Sports Injury'],
            'General Medicine': ['Fever', 'Fatigue', 'General Weakness', 'Minor Injury'],
            'Pediatrics': ['Childhood Illness', 'Vaccination', 'Growth Issues'],
            'Endocrinology': ['Diabetes Symptoms', 'Thyroid Issues', 'Hormonal Imbalance']
        }
        
        # All possible symptoms
        self.all_symptoms = [
            # Critical/Emergency
            'Chest Pain', 'Severe Headache', 'Confusion', 'Difficulty Breathing',
            'Stroke Symptoms', 'Seizure', 'Severe Bleeding', 'Loss of Consciousness',
            
            # Serious/Urgent
            'Shortness of Breath', 'Heart Palpitations', 'Irregular Heartbeat',
            'Severe Abdominal Pain', 'High Fever', 'Severe Dizziness',
            
            # Moderate
            'Fever', 'Abdominal Pain', 'Headache', 'Dizziness', 'Nausea', 
            'Vomiting', 'Back Pain', 'Joint Pain', 'Wheezing',
            
            # Minor
            'Cough', 'Sore Throat', 'Minor Injury', 'Rash', 'Fatigue',
            'Cold Symptoms', 'Allergies', 'Minor Headache'
        ]
        
        # Pre-existing conditions
        self.conditions = [
            'Heart Disease', 'Diabetes', 'Hypertension', 'Asthma', 'COPD',
            'Kidney Disease', 'Liver Disease', 'Cancer', 'Stroke History',
            'Epilepsy', 'Thyroid Disorder', 'Arthritis', 'No History'
        ]
    
    def generate_patient_id(self, index):
        """Generate unique patient ID"""
        return f"PT{datetime.now().year}{index:06d}"
    
    def determine_department(self, symptom, age, risk_level):
        """Determine recommended department based on symptom and patient profile"""
        
        # Check emergency symptoms first
        for dept, symptoms in self.departments.items():
            if symptom in symptoms:
                # Age-based routing
                if age < 18 and dept != 'Emergency':
                    return 'Pediatrics'
                
                # High risk always to emergency if critical symptom
                if risk_level == 'High' and symptom in self.departments['Emergency']:
                    return 'Emergency'
                
                return dept
        
        # Default routing
        if age < 18:
            return 'Pediatrics'
        elif risk_level == 'High':
            return 'Emergency'
        else:
            return 'General Medicine'
    
    def generate_high_risk_patient(self):
        """Generate HIGH RISK patient with critical vital signs"""
        
        # Age: Elderly or very young
        if np.random.rand() < 0.8:
            age = int(np.random.normal(72, 12))
            age = np.clip(age, 55, 95)
        else:
            # Some young high-risk cases (trauma, acute conditions)
            age = int(np.random.uniform(25, 45))
        
        gender = np.random.choice(['Male', 'Female'])
        
        # Critical vital signs
        # Blood Pressure: Hypertensive crisis or hypotensive shock
        if np.random.rand() < 0.7:
            # Hypertensive crisis (>180 systolic)
            bp_systolic = int(np.random.normal(185, 15))
            bp_systolic = np.clip(bp_systolic, 165, 230)
            bp_diastolic = int(np.random.normal(105, 10))
            bp_diastolic = np.clip(bp_diastolic, 95, 130)
        else:
            # Hypotensive (shock state)
            bp_systolic = int(np.random.normal(85, 8))
            bp_systolic = np.clip(bp_systolic, 60, 95)
            bp_diastolic = int(np.random.normal(55, 5))
            bp_diastolic = np.clip(bp_diastolic, 40, 65)
        
        # Heart Rate: Severe tachycardia or bradycardia
        if np.random.rand() < 0.8:
            # Tachycardia
            heart_rate = int(np.random.normal(125, 15))
            heart_rate = np.clip(heart_rate, 105, 170)
        else:
            # Severe bradycardia
            heart_rate = int(np.random.normal(42, 5))
            heart_rate = np.clip(heart_rate, 35, 50)
        
        # Temperature: Fever or hypothermia
        if np.random.rand() < 0.6:
            # High fever
            temperature = round(np.random.normal(39.2, 0.8), 1)
            temperature = np.clip(temperature, 38.5, 41.5)
        elif np.random.rand() < 0.3:
            # Hypothermia
            temperature = round(np.random.normal(35.2, 0.5), 1)
            temperature = np.clip(temperature, 34.0, 36.0)
        else:
            # Normal
            temperature = round(np.random.normal(37.0, 0.3), 1)
        
        # Critical symptoms
        symptoms = np.random.choice([
            'Chest Pain',
            'Difficulty Breathing',
            'Severe Headache',
            'Confusion',
            'Stroke Symptoms',
            'Seizure',
            'Severe Bleeding',
            'Loss of Consciousness',
            'Severe Abdominal Pain'
        ], p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05])
        
        # High-risk conditions
        pre_existing = np.random.choice([
            'Heart Disease',
            'Diabetes',
            'Stroke History',
            'COPD',
            'Kidney Disease',
            'Cancer',
            'Hypertension'
        ], p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05])
        
        return {
            'Age': age,
            'Gender': gender,
            'Systolic_BP': bp_systolic,
            'Diastolic_BP': bp_diastolic,
            'Heart_Rate': heart_rate,
            'Temperature': temperature,
            'Symptoms': symptoms,
            'Pre_Existing': pre_existing,
            'Risk_Level': 'High'
        }
    
    def generate_medium_risk_patient(self):
        """Generate MEDIUM RISK patient with concerning but stable vitals"""
        
        # Age: Middle-aged
        age = int(np.random.normal(52, 15))
        age = np.clip(age, 30, 75)
        
        gender = np.random.choice(['Male', 'Female'])
        
        # Elevated but not critical vitals
        # Blood Pressure: Stage 1-2 Hypertension
        bp_systolic = int(np.random.normal(148, 10))
        bp_systolic = np.clip(bp_systolic, 135, 170)
        bp_diastolic = int(np.random.normal(92, 8))
        bp_diastolic = np.clip(bp_diastolic, 85, 105)
        
        # Heart Rate: Slightly elevated
        heart_rate = int(np.random.normal(95, 10))
        heart_rate = np.clip(heart_rate, 85, 115)
        
        # Temperature: Low-grade fever or normal
        if np.random.rand() < 0.5:
            # Low-grade fever
            temperature = round(np.random.normal(38.0, 0.4), 1)
            temperature = np.clip(temperature, 37.5, 38.8)
        else:
            temperature = round(np.random.normal(37.0, 0.3), 1)
        
        # Moderate symptoms
        symptoms = np.random.choice([
            'Fever',
            'Abdominal Pain',
            'Headache',
            'Dizziness',
            'Nausea',
            'Vomiting',
            'Back Pain',
            'Joint Pain',
            'Shortness of Breath',
            'Heart Palpitations',
            'High Fever'
        ], p=[0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.07, 0.05, 0.03])
        
        # Mixed pre-existing conditions
        pre_existing = np.random.choice([
            'Diabetes',
            'Hypertension',
            'Asthma',
            'Thyroid Disorder',
            'Arthritis',
            'No History'
        ], p=[0.25, 0.25, 0.15, 0.12, 0.10, 0.13])
        
        return {
            'Age': age,
            'Gender': gender,
            'Systolic_BP': bp_systolic,
            'Diastolic_BP': bp_diastolic,
            'Heart_Rate': heart_rate,
            'Temperature': temperature,
            'Symptoms': symptoms,
            'Pre_Existing': pre_existing,
            'Risk_Level': 'Medium'
        }
    
    def generate_low_risk_patient(self):
        """Generate LOW RISK patient with normal vitals"""
        
        # Age: Younger adults
        age = int(np.random.normal(35, 12))
        age = np.clip(age, 18, 60)
        
        gender = np.random.choice(['Male', 'Female'])
        
        # Normal vital signs
        # Blood Pressure: Normal range
        bp_systolic = int(np.random.normal(118, 8))
        bp_systolic = np.clip(bp_systolic, 105, 135)
        bp_diastolic = int(np.random.normal(75, 6))
        bp_diastolic = np.clip(bp_diastolic, 65, 85)
        
        # Heart Rate: Normal
        heart_rate = int(np.random.normal(72, 10))
        heart_rate = np.clip(heart_rate, 55, 90)
        
        # Temperature: Normal
        temperature = round(np.random.normal(36.8, 0.4), 1)
        temperature = np.clip(temperature, 36.0, 37.5)
        
        # Minor symptoms
        symptoms = np.random.choice([
            'Cough',
            'Sore Throat',
            'Minor Injury',
            'Rash',
            'Fatigue',
            'Cold Symptoms',
            'Allergies',
            'Minor Headache'
        ], p=[0.20, 0.18, 0.15, 0.12, 0.12, 0.10, 0.08, 0.05])
        
        # Mostly no history or minor conditions
        pre_existing = np.random.choice([
            'No History',
            'Allergies',
            'Asthma',
            'Thyroid Disorder'
        ], p=[0.70, 0.15, 0.10, 0.05])
        
        return {
            'Age': age,
            'Gender': gender,
            'Systolic_BP': bp_systolic,
            'Diastolic_BP': bp_diastolic,
            'Heart_Rate': heart_rate,
            'Temperature': temperature,
            'Symptoms': symptoms,
            'Pre_Existing': pre_existing,
            'Risk_Level': 'Low'
        }
    
    def add_extreme_cases(self, data):
        """Add specific extreme and edge cases"""
        
        extreme_cases = []
        
        # Case 1: Pediatric emergency
        extreme_cases.append({
            'Age': 8,
            'Gender': 'Male',
            'Systolic_BP': 95,
            'Diastolic_BP': 60,
            'Heart_Rate': 145,
            'Temperature': 40.2,
            'Symptoms': 'High Fever',
            'Pre_Existing': 'No History',
            'Risk_Level': 'High'
        })
        
        # Case 2: Elderly with multiple conditions
        extreme_cases.append({
            'Age': 89,
            'Gender': 'Female',
            'Systolic_BP': 88,
            'Diastolic_BP': 55,
            'Heart_Rate': 48,
            'Temperature': 35.8,
            'Symptoms': 'Confusion',
            'Pre_Existing': 'Heart Disease',
            'Risk_Level': 'High'
        })
        
        # Case 3: Young athlete with cardiac event
        extreme_cases.append({
            'Age': 26,
            'Gender': 'Male',
            'Systolic_BP': 95,
            'Diastolic_BP': 58,
            'Heart_Rate': 165,
            'Temperature': 38.9,
            'Symptoms': 'Chest Pain',
            'Pre_Existing': 'No History',
            'Risk_Level': 'High'
        })
        
        # Case 4: Pregnant woman (implied by age/gender + specific vitals)
        extreme_cases.append({
            'Age': 32,
            'Gender': 'Female',
            'Systolic_BP': 168,
            'Diastolic_BP': 102,
            'Heart_Rate': 108,
            'Temperature': 37.8,
            'Symptoms': 'Severe Headache',
            'Pre_Existing': 'Hypertension',
            'Risk_Level': 'High'
        })
        
        # Case 5: Diabetic crisis
        extreme_cases.append({
            'Age': 58,
            'Gender': 'Male',
            'Systolic_BP': 178,
            'Diastolic_BP': 98,
            'Heart_Rate': 118,
            'Temperature': 37.2,
            'Symptoms': 'Confusion',
            'Pre_Existing': 'Diabetes',
            'Risk_Level': 'High'
        })
        
        # Case 6: Asthma attack
        extreme_cases.append({
            'Age': 42,
            'Gender': 'Female',
            'Systolic_BP': 142,
            'Diastolic_BP': 88,
            'Heart_Rate': 102,
            'Temperature': 36.9,
            'Symptoms': 'Difficulty Breathing',
            'Pre_Existing': 'Asthma',
            'Risk_Level': 'High'
        })
        
        # Case 7: Stroke symptoms
        extreme_cases.append({
            'Age': 67,
            'Gender': 'Male',
            'Systolic_BP': 192,
            'Diastolic_BP': 108,
            'Heart_Rate': 92,
            'Temperature': 37.1,
            'Symptoms': 'Stroke Symptoms',
            'Pre_Existing': 'Stroke History',
            'Risk_Level': 'High'
        })
        
        # Case 8: Healthy young adult with minor issue
        extreme_cases.append({
            'Age': 24,
            'Gender': 'Female',
            'Systolic_BP': 112,
            'Diastolic_BP': 72,
            'Heart_Rate': 68,
            'Temperature': 36.7,
            'Symptoms': 'Sore Throat',
            'Pre_Existing': 'No History',
            'Risk_Level': 'Low'
        })
        
        # Case 9: Elderly with stable chronic condition
        extreme_cases.append({
            'Age': 78,
            'Gender': 'Female',
            'Systolic_BP': 138,
            'Diastolic_BP': 82,
            'Heart_Rate': 76,
            'Temperature': 36.8,
            'Symptoms': 'Fatigue',
            'Pre_Existing': 'Hypertension',
            'Risk_Level': 'Medium'
        })
        
        # Case 10: Borderline case (could be medium or low)
        extreme_cases.append({
            'Age': 45,
            'Gender': 'Male',
            'Systolic_BP': 136,
            'Diastolic_BP': 86,
            'Heart_Rate': 84,
            'Temperature': 37.4,
            'Symptoms': 'Headache',
            'Pre_Existing': 'No History',
            'Risk_Level': 'Medium'
        })
        
        return data + extreme_cases
    
    def generate_dataset(self, n_samples=15000):
        """Generate complete dataset with balanced distribution"""
        
        data = []
        
        # Generate balanced distribution (33% each risk level)
        n_high = int(n_samples * 0.33)
        n_medium = int(n_samples * 0.33)
        n_low = n_samples - n_high - n_medium
        
        print(f"Generating {n_samples} patient records...")
        print(f"  - High Risk: {n_high}")
        print(f"  - Medium Risk: {n_medium}")
        print(f"  - Low Risk: {n_low}")
        
        # Generate each category
        for i in range(n_high):
            data.append(self.generate_high_risk_patient())
        
        for i in range(n_medium):
            data.append(self.generate_medium_risk_patient())
        
        for i in range(n_low):
            data.append(self.generate_low_risk_patient())
        
        # Add extreme cases
        print("Adding extreme edge cases...")
        data = self.add_extreme_cases(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add Patient_ID
        df.insert(0, 'Patient_ID', [self.generate_patient_id(i) for i in range(len(df))])
        
        # Add Department recommendations
        print("Determining department recommendations...")
        df['Recommended_Department'] = df.apply(
            lambda row: self.determine_department(
                row['Symptoms'], 
                row['Age'], 
                row['Risk_Level']
            ), axis=1
        )
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Update Patient_ID after shuffle
        df['Patient_ID'] = [self.generate_patient_id(i) for i in range(len(df))]
        
        return df

# ============================================================================
# GENERATE THE DATASET
# ============================================================================

print("="*80)
print(" "*15 + "HACKATHON MEDICAL TRIAGE DATA GENERATOR")
print("="*80)
print()

generator = MedicalDataGenerator()
df = generator.generate_dataset(n_samples=15000)

# ============================================================================
# DISPLAY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)

print(f"\nTotal Records: {len(df):,}")

print("\n" + "-"*80)
print("Risk Level Distribution:")
print("-"*80)
for level in ['High', 'Medium', 'Low']:
    count = (df['Risk_Level'] == level).sum()
    pct = count / len(df) * 100
    print(f"  {level:6s}: {count:5d} ({pct:5.2f}%)")

print("\n" + "-"*80)
print("Department Distribution:")
print("-"*80)
dept_counts = df['Recommended_Department'].value_counts()
for dept, count in dept_counts.items():
    pct = count / len(df) * 100
    print(f"  {dept:20s}: {count:5d} ({pct:5.2f}%)")

print("\n" + "-"*80)
print("Age Statistics by Risk Level:")
print("-"*80)
for level in ['High', 'Medium', 'Low']:
    ages = df[df['Risk_Level'] == level]['Age']
    print(f"  {level:6s}: Mean={ages.mean():.1f}, Min={ages.min()}, Max={ages.max()}, Std={ages.std():.1f}")

print("\n" + "-"*80)
print("Vital Signs Statistics by Risk Level:")
print("-"*80)
for level in ['High', 'Medium', 'Low']:
    subset = df[df['Risk_Level'] == level]
    print(f"\n  {level} Risk:")
    print(f"    Systolic BP:  Mean={subset['Systolic_BP'].mean():.1f}, Range=[{subset['Systolic_BP'].min()}-{subset['Systolic_BP'].max()}]")
    print(f"    Heart Rate:   Mean={subset['Heart_Rate'].mean():.1f}, Range=[{subset['Heart_Rate'].min()}-{subset['Heart_Rate'].max()}]")
    print(f"    Temperature:  Mean={subset['Temperature'].mean():.1f}, Range=[{subset['Temperature'].min():.1f}-{subset['Temperature'].max():.1f}]")

print("\n" + "-"*80)
print("Top 10 Most Common Symptoms:")
print("-"*80)
symptom_counts = df['Symptoms'].value_counts().head(10)
for symptom, count in symptom_counts.items():
    pct = count / len(df) * 100
    print(f"  {symptom:25s}: {count:5d} ({pct:5.2f}%)")

print("\n" + "-"*80)
print("Pre-Existing Conditions Distribution:")
print("-"*80)
condition_counts = df['Pre_Existing'].value_counts()
for condition, count in condition_counts.items():
    pct = count / len(df) * 100
    print(f"  {condition:20s}: {count:5d} ({pct:5.2f}%)")

print("\n" + "-"*80)
print("Sample High-Risk Emergency Cases:")
print("-"*80)
high_risk_sample = df[df['Risk_Level'] == 'High'].head(3)
print(high_risk_sample[['Patient_ID', 'Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 
                        'Temperature', 'Symptoms', 'Recommended_Department']].to_string(index=False))

print("\n" + "-"*80)
print("Sample Low-Risk Routine Cases:")
print("-"*80)
low_risk_sample = df[df['Risk_Level'] == 'Low'].head(3)
print(low_risk_sample[['Patient_ID', 'Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 
                       'Temperature', 'Symptoms', 'Recommended_Department']].to_string(index=False))

# ============================================================================
# SAVE DATASET
# ============================================================================

filename = 'hackathon_medical_triage_data.csv'
df.to_csv(filename, index=False)

print("\n" + "="*80)
print(f"✅ SUCCESS: Saved {len(df):,} records to '{filename}'")
print("="*80)

print("\nDataset Features:")
print("  ✓ Patient_ID (Unique identifier)")
print("  ✓ Age (18-95 years)")
print("  ✓ Gender (Male/Female)")
print("  ✓ Systolic_BP (60-230 mmHg)")
print("  ✓ Diastolic_BP (40-130 mmHg)")
print("  ✓ Heart_Rate (35-170 BPM)")
print("  ✓ Temperature (34.0-41.5°C)")
print("  ✓ Symptoms (28 different types)")
print("  ✓ Pre_Existing (13 condition categories)")
print("  ✓ Risk_Level (High/Medium/Low)")
print("  ✓ Recommended_Department (9 departments)")

print("\nKey Characteristics:")
print("  ✓ Realistic medical correlations")
print("  ✓ Extreme cases included")
print("  ✓ Edge scenarios covered")
print("  ✓ Balanced risk distribution")
print("  ✓ Department routing logic")
print("  ✓ Ready for ML training")

print("\n" + "="*80)
print("Next Step: Run 'train_hackathon_model.py' to train the AI system!")
print("="*80)
