"""
HACKATHON AI TRIAGE SYSTEM - PREDICTION SCRIPT
===============================================
Standalone prediction with explainability
"""

import pandas as pd
import numpy as np
import joblib
import sys

class TriagePredictionSystem:
    """AI Triage System for Risk and Department Prediction"""
    
    def __init__(self):
        """Load all models and encoders"""
        try:
            # Load models
            self.risk_model = joblib.load('risk_model.pkl')
            self.dept_model = joblib.load('department_model.pkl')
            
            # Load encoders
            self.le_gender = joblib.load('le_gender.pkl')
            self.le_symptoms = joblib.load('le_symptoms.pkl')
            self.le_pre_existing = joblib.load('le_pre_existing.pkl')
            self.le_risk = joblib.load('le_risk.pkl')
            self.le_department = joblib.load('le_department.pkl')
            
            print("✓ All models and encoders loaded successfully")
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: Missing file - {e.filename}")
            print("Please run 'train_hackathon_model.py' first!")
            sys.exit(1)
    
    def get_feature_names(self):
        """Return feature names for explainability"""
        return ['Age', 'Gender', 'Systolic_BP', 'Diastolic_BP', 
                'Heart_Rate', 'Temperature', 'Symptoms', 'Pre_Existing']
    
    def predict(self, patient_data):
        """
        Make predictions for a patient
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary with keys:
            - Age (int)
            - Gender (str: 'Male' or 'Female')
            - Systolic_BP (int)
            - Diastolic_BP (int)
            - Heart_Rate (int)
            - Temperature (float)
            - Symptoms (str)
            - Pre_Existing (str)
        
        Returns:
        --------
        dict with prediction results and explainability
        """
        
        # Encode categorical variables
        try:
            gender_enc = self.le_gender.transform([patient_data['Gender']])[0]
            symptom_enc = self.le_symptoms.transform([patient_data['Symptoms']])[0]
            pre_enc = self.le_pre_existing.transform([patient_data['Pre_Existing']])[0]
        except ValueError as e:
            return {
                'error': f"Invalid input value: {e}",
                'valid_genders': list(self.le_gender.classes_),
                'valid_symptoms': list(self.le_symptoms.classes_),
                'valid_conditions': list(self.le_pre_existing.classes_)
            }
        
        # Create feature vector
        features = pd.DataFrame([[
            patient_data['Age'],
            gender_enc,
            patient_data['Systolic_BP'],
            patient_data['Diastolic_BP'],
            patient_data['Heart_Rate'],
            patient_data['Temperature'],
            symptom_enc,
            pre_enc
        ]], columns=['Age', 'Gender_Encoded', 'Systolic_BP', 'Diastolic_BP',
                     'Heart_Rate', 'Temperature', 'Symptoms_Encoded', 'Pre_Existing_Encoded'])
        
        # Predict Risk Level
        risk_pred = self.risk_model.predict(features)[0]
        risk_proba = self.risk_model.predict_proba(features)[0]
        risk_confidence = risk_proba[list(self.risk_model.classes_).index(risk_pred)] * 100
        
        # Predict Department
        dept_pred = self.dept_model.predict(features)[0]
        dept_proba = self.dept_model.predict_proba(features)[0]
        dept_confidence = dept_proba[list(self.dept_model.classes_).index(dept_pred)] * 100
        
        # Get feature importance
        risk_importance = self.risk_model.feature_importances_
        dept_importance = self.dept_model.feature_importances_
        
        # Create probability distributions
        risk_probabilities = {
            cls: float(prob * 100) 
            for cls, prob in zip(self.risk_model.classes_, risk_proba)
        }
        
        dept_probabilities = {
            cls: float(prob * 100)
            for cls, prob in zip(self.dept_model.classes_, dept_proba)
        }
        
        # Analyze contributing factors
        contributing_factors = self._explain_prediction(patient_data, risk_pred)
        
        return {
            'risk_level': risk_pred,
            'risk_confidence': float(risk_confidence),
            'risk_probabilities': risk_probabilities,
            'recommended_department': dept_pred,
            'department_confidence': float(dept_confidence),
            'department_probabilities': dept_probabilities,
            'contributing_factors': contributing_factors,
            'feature_importance_risk': {
                name: float(imp * 100)
                for name, imp in zip(self.get_feature_names(), risk_importance)
            },
            'feature_importance_department': {
                name: float(imp * 100)
                for name, imp in zip(self.get_feature_names(), dept_importance)
            }
        }
    
    def _explain_prediction(self, patient_data, risk_level):
        """Generate human-readable explanation of the prediction"""
        
        factors = []
        
        # Age analysis
        age = patient_data['Age']
        if age > 65 and risk_level == 'High':
            factors.append(f"Advanced age ({age} years) increases risk significantly")
        elif age > 65:
            factors.append(f"Elderly patient ({age} years) - age is a risk factor")
        elif age < 30 and risk_level == 'Low':
            factors.append(f"Young patient ({age} years) - lower baseline risk")
        
        # Blood Pressure analysis
        bp_sys = patient_data['Systolic_BP']
        bp_dia = patient_data['Diastolic_BP']
        
        if bp_sys > 180 or bp_dia > 100:
            factors.append(f"Hypertensive crisis (BP: {bp_sys}/{bp_dia}) - immediate concern")
        elif bp_sys > 160 or bp_dia > 95:
            factors.append(f"Severely elevated blood pressure ({bp_sys}/{bp_dia})")
        elif bp_sys > 140 or bp_dia > 90:
            factors.append(f"Elevated blood pressure ({bp_sys}/{bp_dia})")
        elif bp_sys < 90 or bp_dia < 60:
            factors.append(f"Low blood pressure ({bp_sys}/{bp_dia}) - potential shock")
        elif bp_sys <= 120 and bp_dia <= 80:
            factors.append(f"Normal blood pressure ({bp_sys}/{bp_dia})")
        
        # Heart Rate analysis
        hr = patient_data['Heart_Rate']
        if hr > 120:
            factors.append(f"Severe tachycardia ({hr} BPM) - significant concern")
        elif hr > 100:
            factors.append(f"Tachycardia ({hr} BPM) - elevated heart rate")
        elif hr < 50:
            factors.append(f"Bradycardia ({hr} BPM) - slow heart rate")
        elif 60 <= hr <= 100:
            factors.append(f"Normal heart rate ({hr} BPM)")
        
        # Temperature analysis
        temp = patient_data['Temperature']
        if temp > 39.0:
            factors.append(f"High fever ({temp}°C) - indicates infection or inflammation")
        elif temp > 38.0:
            factors.append(f"Fever present ({temp}°C)")
        elif temp < 36.0:
            factors.append(f"Hypothermia ({temp}°C) - concerning")
        elif 36.0 <= temp <= 37.5:
            factors.append(f"Normal temperature ({temp}°C)")
        
        # Symptom analysis
        symptom = patient_data['Symptoms']
        critical_symptoms = ['Chest Pain', 'Difficulty Breathing', 'Stroke Symptoms', 
                           'Severe Headache', 'Confusion', 'Seizure', 'Severe Bleeding']
        if symptom in critical_symptoms:
            factors.append(f"CRITICAL symptom: {symptom} - requires immediate attention")
        else:
            factors.append(f"Presenting symptom: {symptom}")
        
        # Pre-existing condition analysis
        condition = patient_data['Pre_Existing']
        high_risk_conditions = ['Heart Disease', 'Stroke History', 'COPD', 'Kidney Disease']
        if condition in high_risk_conditions:
            factors.append(f"High-risk pre-existing condition: {condition}")
        elif condition != 'No History':
            factors.append(f"Pre-existing condition: {condition}")
        
        return factors


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "AI TRIAGE PREDICTION SYSTEM")
    print("="*80)
    print()
    
    # Initialize system
    system = TriagePredictionSystem()
    
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    # Example 1: Critical Emergency
    print("\n" + "-"*80)
    print("Example 1: CRITICAL CARDIAC EMERGENCY")
    print("-"*80)
    
    patient1 = {
        'Age': 72,
        'Gender': 'Male',
        'Systolic_BP': 192,
        'Diastolic_BP': 108,
        'Heart_Rate': 128,
        'Temperature': 37.3,
        'Symptoms': 'Chest Pain',
        'Pre_Existing': 'Heart Disease'
    }
    
    result1 = system.predict(patient1)
    
    print(f"\nPatient: {patient1['Age']}yo {patient1['Gender']}")
    print(f"Vitals: BP {patient1['Systolic_BP']}/{patient1['Diastolic_BP']}, HR {patient1['Heart_Rate']}, Temp {patient1['Temperature']}°C")
    print(f"Symptom: {patient1['Symptoms']}")
    print(f"History: {patient1['Pre_Existing']}")
    
    print(f"\n🎯 PREDICTION:")
    print(f"  Risk Level: {result1['risk_level']} (Confidence: {result1['risk_confidence']:.1f}%)")
    print(f"  Department: {result1['recommended_department']} (Confidence: {result1['department_confidence']:.1f}%)")
    
    print(f"\n💡 CONTRIBUTING FACTORS:")
    for factor in result1['contributing_factors']:
        print(f"  • {factor}")
    
    # Example 2: Moderate Case
    print("\n" + "-"*80)
    print("Example 2: MODERATE RESPIRATORY")
    print("-"*80)
    
    patient2 = {
        'Age': 45,
        'Gender': 'Female',
        'Systolic_BP': 145,
        'Diastolic_BP': 88,
        'Heart_Rate': 92,
        'Temperature': 37.8,
        'Symptoms': 'Cough',
        'Pre_Existing': 'Asthma'
    }
    
    result2 = system.predict(patient2)
    
    print(f"\nPatient: {patient2['Age']}yo {patient2['Gender']}")
    print(f"Vitals: BP {patient2['Systolic_BP']}/{patient2['Diastolic_BP']}, HR {patient2['Heart_Rate']}, Temp {patient2['Temperature']}°C")
    print(f"Symptom: {patient2['Symptoms']}")
    print(f"History: {patient2['Pre_Existing']}")
    
    print(f"\n🎯 PREDICTION:")
    print(f"  Risk Level: {result2['risk_level']} (Confidence: {result2['risk_confidence']:.1f}%)")
    print(f"  Department: {result2['recommended_department']} (Confidence: {result2['department_confidence']:.1f}%)")
    
    print(f"\n💡 CONTRIBUTING FACTORS:")
    for factor in result2['contributing_factors']:
        print(f"  • {factor}")
    
    # Example 3: Low Risk
    print("\n" + "-"*80)
    print("Example 3: STABLE MINOR COMPLAINT")
    print("-"*80)
    
    patient3 = {
        'Age': 28,
        'Gender': 'Male',
        'Systolic_BP': 118,
        'Diastolic_BP': 75,
        'Heart_Rate': 72,
        'Temperature': 36.8,
        'Symptoms': 'Sore Throat',
        'Pre_Existing': 'No History'
    }
    
    result3 = system.predict(patient3)
    
    print(f"\nPatient: {patient3['Age']}yo {patient3['Gender']}")
    print(f"Vitals: BP {patient3['Systolic_BP']}/{patient3['Diastolic_BP']}, HR {patient3['Heart_Rate']}, Temp {patient3['Temperature']}°C")
    print(f"Symptom: {patient3['Symptoms']}")
    print(f"History: {patient3['Pre_Existing']}")
    
    print(f"\n🎯 PREDICTION:")
    print(f"  Risk Level: {result3['risk_level']} (Confidence: {result3['risk_confidence']:.1f}%)")
    print(f"  Department: {result3['recommended_department']} (Confidence: {result3['department_confidence']:.1f}%)")
    
    print(f"\n💡 CONTRIBUTING FACTORS:")
    for factor in result3['contributing_factors']:
        print(f"  • {factor}")
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*80)
    
    print("\nFor Risk Classification:")
    sorted_features = sorted(result1['feature_importance_risk'].items(), 
                            key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        print(f"  {feature:20s}: {importance:5.2f}%")
    
    print("\nFor Department Recommendation:")
    sorted_features = sorted(result1['feature_importance_department'].items(), 
                            key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        print(f"  {feature:20s}: {importance:5.2f}%")
    
    print("\n" + "="*80)
    print("Ready to integrate with your application!")
    print("="*80)
