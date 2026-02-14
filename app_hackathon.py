"""
HACKATHON AI TRIAGE SYSTEM - MULTI-STEP STREAMLIT APP
======================================================
Step-by-step patient intake matching Next.js UI flow:
1. Vitals
2. Symptoms
3. History
4. Review & Results
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MedTouch.ai Patient Intake",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching the modern UI
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .stApp > div > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(147, 197, 253, 0.5);
    }
    
    /* Progress bar container */
    .stepper {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .step {
        flex: 1;
        text-align: center;
        font-weight: 600;
        color: #9ca3af;
        transition: all 0.3s ease;
    }
    
    .step-active {
        color: #2563eb;
        transform: scale(1.1);
    }
    
    /* Header styling */
    h1 {
        color: #1e40af;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Risk level cards */
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.4);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4);
    }
    
    /* Factor box */
    .factor-box {
        background: linear-gradient(to right, #eff6ff, #dbeafe);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2563eb;
    }
    
    /* Department box */
    .dept-box {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.4);
    }
    
    /* Symptom card styling */
    div[data-testid="stCheckbox"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stCheckbox"]:hover {
        border-color: #2563eb;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load all models and encoders"""
    try:
        models = {
            'risk_model': joblib.load('risk_model.pkl'),
            'dept_model': joblib.load('department_model.pkl'),
            'le_gender': joblib.load('le_gender.pkl'),
            'le_symptoms': joblib.load('le_symptoms.pkl'),
            'le_pre_existing': joblib.load('le_pre_existing.pkl'),
        }
        return models, None
    except FileNotFoundError as e:
        return None, f"Missing file: {e.filename}"

models, error = load_models()

if error:
    st.error(f"❌ {error}")
    st.info("Please ensure all model files are in the same directory as this app.")
    st.stop()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'age': 45,
        'gender': 'Male',
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'heart_rate': 80,
        'temperature': 36.8,
        'symptoms': [],
        'pre_existing': 'No History'
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def next_step():
    """Move to next step"""
    st.session_state.step += 1

def prev_step():
    """Move to previous step"""
    st.session_state.step -= 1

def reset_form():
    """Reset to step 1"""
    st.session_state.step = 1
    st.session_state.form_data = {
        'age': 45,
        'gender': 'Male',
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'heart_rate': 80,
        'temperature': 36.8,
        'symptoms': [],
        'pre_existing': 'No History'
    }

def explain_prediction(patient_data, risk_level):
    """Generate explanation factors"""
    factors = []
    
    age = patient_data['age']
    if age > 65 and risk_level == 'High':
        factors.append(f"🔴 Advanced age ({age} years) increases risk significantly")
    elif age > 65:
        factors.append(f"⚠️ Elderly patient ({age} years)")
    elif age < 30:
        factors.append(f"✅ Young patient ({age} years) - lower baseline risk")
    
    bp_sys = patient_data['systolic_bp']
    bp_dia = patient_data['diastolic_bp']
    
    if bp_sys > 180 or bp_dia > 100:
        factors.append(f"🔴 Hypertensive crisis (BP: {bp_sys}/{bp_dia})")
    elif bp_sys > 140 or bp_dia > 90:
        factors.append(f"⚠️ Elevated blood pressure ({bp_sys}/{bp_dia})")
    elif bp_sys < 90:
        factors.append(f"🔴 Low blood pressure ({bp_sys}/{bp_dia})")
    else:
        factors.append(f"✅ Normal blood pressure ({bp_sys}/{bp_dia})")
    
    hr = patient_data['heart_rate']
    if hr > 120:
        factors.append(f"🔴 Severe tachycardia ({hr} BPM)")
    elif hr > 100:
        factors.append(f"⚠️ Tachycardia ({hr} BPM)")
    elif hr < 50:
        factors.append(f"⚠️ Bradycardia ({hr} BPM)")
    else:
        factors.append(f"✅ Normal heart rate ({hr} BPM)")
    
    temp = patient_data['temperature']
    if temp > 39.0:
        factors.append(f"🔴 High fever ({temp}°C)")
    elif temp > 38.0:
        factors.append(f"⚠️ Fever present ({temp}°C)")
    elif temp < 36.0:
        factors.append(f"🔴 Hypothermia ({temp}°C)")
    else:
        factors.append(f"✅ Normal temperature ({temp}°C)")
    
    # Symptoms
    if patient_data['symptoms']:
        symptom = patient_data['symptoms'][0] if isinstance(patient_data['symptoms'], list) else patient_data['symptoms']
        critical = ['Chest Pain', 'Difficulty Breathing', 'Stroke Symptoms', 'Severe Headache']
        if symptom in critical:
            factors.append(f"🔴 CRITICAL symptom: {symptom}")
        else:
            factors.append(f"📋 Symptom: {symptom}")
    
    condition = patient_data['pre_existing']
    high_risk = ['Heart Disease', 'Stroke History', 'COPD', 'Kidney Disease']
    if condition in high_risk:
        factors.append(f"⚠️ High-risk condition: {condition}")
    elif condition != 'No History':
        factors.append(f"📋 Pre-existing: {condition}")
    
    return factors

def make_prediction(patient_data):
    """Make prediction"""
    try:
        # Get primary symptom
        symptom = patient_data['symptoms'][0] if patient_data['symptoms'] else 'Fatigue'
        
        # Encode
        gender_enc = models['le_gender'].transform([patient_data['gender']])[0]
        symptom_enc = models['le_symptoms'].transform([symptom])[0]
        pre_enc = models['le_pre_existing'].transform([patient_data['pre_existing']])[0]
        
        # Create features
        features = pd.DataFrame([[
            patient_data['age'],
            gender_enc,
            patient_data['systolic_bp'],
            patient_data['diastolic_bp'],
            patient_data['heart_rate'],
            patient_data['temperature'],
            symptom_enc,
            pre_enc
        ]], columns=['Age', 'Gender_Encoded', 'Systolic_BP', 'Diastolic_BP',
                     'Heart_Rate', 'Temperature', 'Symptoms_Encoded', 'Pre_Existing_Encoded'])
        
        # Predict
        risk = models['risk_model'].predict(features)[0]
        risk_proba = models['risk_model'].predict_proba(features)[0]
        risk_conf = risk_proba[list(models['risk_model'].classes_).index(risk)] * 100
        
        dept = models['dept_model'].predict(features)[0]
        dept_proba = models['dept_model'].predict_proba(features)[0]
        dept_conf = dept_proba[list(models['dept_model'].classes_).index(dept)] * 100
        
        risk_probs = {cls: prob*100 for cls, prob in zip(models['risk_model'].classes_, risk_proba)}
        
        return {
            'risk': risk,
            'risk_confidence': risk_conf,
            'risk_probs': risk_probs,
            'department': dept,
            'dept_confidence': dept_conf,
            'factors': explain_prediction(patient_data, risk)
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# STEPPER COMPONENT
# ============================================================================
def render_stepper(current_step):
    """Render progress stepper"""
    steps = ["Vitals", "Symptoms", "History", "Review"]
    
    stepper_html = '<div class="stepper">'
    for i, label in enumerate(steps, 1):
        active_class = 'step-active' if i == current_step else ''
        stepper_html += f'<div class="step {active_class}">{label}</div>'
    stepper_html += '</div>'
    
    st.markdown(stepper_html, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("# 🏥 MedTouch.ai Patient Intake")

# Stepper
render_stepper(st.session_state.step)

# ============================================================================
# STEP 1: VITALS
# ============================================================================
if st.session_state.step == 1:
    st.markdown("## Patient Information")
    st.markdown("*Demographics & Clinical Vitals*")
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demographics")
        
        # Age
        st.session_state.form_data['age'] = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=st.session_state.form_data['age'],
            help="Enter patient's age in years"
        )
        
        # Gender
        st.session_state.form_data['gender'] = st.selectbox(
            "Gender",
            options=['Male', 'Female'],
            index=0 if st.session_state.form_data['gender'] == 'Male' else 1
        )
        
        st.markdown("### Blood Pressure")
        
        # Systolic BP
        st.session_state.form_data['systolic_bp'] = st.slider(
            "Systolic BP (mmHg)",
            min_value=80,
            max_value=220,
            value=st.session_state.form_data['systolic_bp'],
            help="Normal: 90-120 mmHg"
        )
        
        # Diastolic BP
        st.session_state.form_data['diastolic_bp'] = st.slider(
            "Diastolic BP (mmHg)",
            min_value=40,
            max_value=130,
            value=st.session_state.form_data['diastolic_bp'],
            help="Normal: 60-80 mmHg"
        )
    
    with col2:
        st.markdown("### Vital Signs")
        
        # Heart Rate
        st.session_state.form_data['heart_rate'] = st.slider(
            "Heart Rate (BPM)",
            min_value=30,
            max_value=200,
            value=st.session_state.form_data['heart_rate'],
            help="Normal: 60-100 BPM"
        )
        
        # Temperature
        st.session_state.form_data['temperature'] = st.number_input(
            "Temperature (°C)",
            min_value=34.0,
            max_value=42.0,
            value=st.session_state.form_data['temperature'],
            step=0.1,
            help="Normal: 36.0-37.5°C"
        )
    
    st.markdown("")
    st.markdown("")
    
    # Continue button
    if st.button("Continue →", key="vitals_continue"):
        next_step()
        st.rerun()

# ============================================================================
# STEP 2: SYMPTOMS
# ============================================================================
elif st.session_state.step == 2:
    st.markdown("## Select Symptoms")
    st.markdown("*Choose all that apply*")
    st.markdown("")
    
    # Get available symptoms from model
    all_symptoms = sorted(models['le_symptoms'].classes_)
    
    # Display in 3 columns
    cols = st.columns(3)
    
    for i, symptom in enumerate(all_symptoms[:18]):  # Show top 18 symptoms
        with cols[i % 3]:
            if st.checkbox(symptom, key=f"symptom_{symptom}"):
                if symptom not in st.session_state.form_data['symptoms']:
                    st.session_state.form_data['symptoms'].append(symptom)
            else:
                if symptom in st.session_state.form_data['symptoms']:
                    st.session_state.form_data['symptoms'].remove(symptom)
    
    st.markdown("")
    st.markdown("")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous", key="symptoms_prev"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue →", key="symptoms_next"):
            if not st.session_state.form_data['symptoms']:
                st.warning("⚠️ Please select at least one symptom")
            else:
                next_step()
                st.rerun()

# ============================================================================
# STEP 3: HISTORY
# ============================================================================
elif st.session_state.step == 3:
    st.markdown("## Medical History")
    st.markdown("*Pre-existing conditions*")
    st.markdown("")
    
    # Get available conditions
    all_conditions = sorted(models['le_pre_existing'].classes_)
    
    st.session_state.form_data['pre_existing'] = st.selectbox(
        "Select Pre-Existing Condition",
        options=all_conditions,
        index=all_conditions.index(st.session_state.form_data['pre_existing']) 
              if st.session_state.form_data['pre_existing'] in all_conditions else 0
    )
    
    st.markdown("")
    st.info("💡 Select 'No History' if patient has no pre-existing conditions")
    
    st.markdown("")
    st.markdown("")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous", key="history_prev"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Analyze Patient →", key="history_analyze"):
            next_step()
            st.rerun()

# ============================================================================
# STEP 4: REVIEW & RESULTS
# ============================================================================
elif st.session_state.step == 4:
    st.markdown("## 🎯 Analysis Results")
    st.markdown("")
    
    # Make prediction
    result = make_prediction(st.session_state.form_data)
    
    if 'error' in result:
        st.error(f"Error: {result['error']}")
        if st.button("← Go Back"):
            prev_step()
            st.rerun()
    else:
        # Display results in columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Risk Level
            st.markdown("### Risk Classification")
            risk = result['risk']
            conf = result['risk_confidence']
            
            if risk == 'High':
                st.markdown(f'<div class="risk-high">🔴 HIGH RISK<br/>{conf:.1f}% Confidence</div>', 
                           unsafe_allow_html=True)
            elif risk == 'Medium':
                st.markdown(f'<div class="risk-medium">🟡 MEDIUM RISK<br/>{conf:.1f}% Confidence</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">🟢 LOW RISK<br/>{conf:.1f}% Confidence</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("")
            
            # Department
            st.markdown("### Recommended Department")
            st.markdown(f'<div class="dept-box">📍 {result["department"]}<br/>{result["dept_confidence"]:.1f}% Match</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("")
            
            # Probabilities
            st.markdown("### Risk Probabilities")
            for level in ['High', 'Medium', 'Low']:
                prob = result['risk_probs'].get(level, 0)
                icon = "🔴" if level == 'High' else "🟡" if level == 'Medium' else "🟢"
                st.metric(f"{icon} {level}", f"{prob:.1f}%")
        
        with col2:
            # Clinical Recommendations
            st.markdown("### 🏥 Clinical Recommendations")
            
            if risk == 'High':
                st.error("""
                **🔴 IMMEDIATE ACTION REQUIRED**
                
                **Priority:** ESI Level 1 (Resuscitation)
                
                **Actions:**
                - Immediate trauma bay assignment
                - Alert attending physician
                - Prepare emergency equipment
                - Continuous monitoring required
                - IV access (2 large-bore)
                
                **Target:** Physician evaluation IMMEDIATELY
                """)
            elif risk == 'Medium':
                st.warning("""
                **🟡 URGENT ASSESSMENT NEEDED**
                
                **Priority:** ESI Level 2-3 (Emergent/Urgent)
                
                **Actions:**
                - Move to urgent care area
                - Vitals every 15-30 minutes
                - Priority physician evaluation
                - Prepare for diagnostic workup
                
                **Target:** Physician evaluation within 15-30 minutes
                """)
            else:
                st.success("""
                **🟢 ROUTINE PROCESSING**
                
                **Priority:** ESI Level 4-5 (Less/Non-Urgent)
                
                **Actions:**
                - Assign to general waiting area
                - Standard monitoring protocol
                - Process in queue order
                
                **Expected Wait:** 1-2 hours
                """)
        
        # Contributing Factors
        st.markdown("---")
        st.markdown("## 💡 Why This Classification?")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("### Contributing Factors")
            for factor in result['factors']:
                st.markdown(f'<div class="factor-box">{factor}</div>', unsafe_allow_html=True)
        
        with exp_col2:
            st.markdown("### Patient Summary")
            
            summary_data = {
                'Field': ['Age', 'Gender', 'Blood Pressure', 'Heart Rate', 'Temperature', 
                         'Symptoms', 'Pre-Existing'],
                'Value': [
                    f"{st.session_state.form_data['age']} years",
                    st.session_state.form_data['gender'],
                    f"{st.session_state.form_data['systolic_bp']}/{st.session_state.form_data['diastolic_bp']} mmHg",
                    f"{st.session_state.form_data['heart_rate']} BPM",
                    f"{st.session_state.form_data['temperature']}°C",
                    ', '.join(st.session_state.form_data['symptoms'][:3]) + ('...' if len(st.session_state.form_data['symptoms']) > 3 else ''),
                    st.session_state.form_data['pre_existing']
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Action buttons
        st.markdown("")
        st.markdown("---")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("← Previous", key="results_prev"):
                prev_step()
                st.rerun()
        
        with btn_col2:
            if st.button("🔄 New Patient", key="results_reset"):
                reset_form()
                st.rerun()
        
        with btn_col3:
            st.success("✅ Assessment Complete")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("")
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for demonstration purposes only.
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
        <em>MedTouch.ai v1.0 | AI-Powered Triage System</em>
    </p>
</div>
""", unsafe_allow_html=True)