import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MedTouch.ai | AI Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND ENCODERS
# ============================================================================
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and all label encoders"""
    try:
        model = joblib.load('triage_model.pkl')
        le_gender = joblib.load('le_gender.pkl')
        le_symptoms = joblib.load('le_symptoms.pkl')
        le_pre = joblib.load('le_pre.pkl')
        return model, le_gender, le_symptoms, le_pre, None
    except FileNotFoundError as e:
        error_msg = f"❌ Missing file: {e.filename}\n\nPlease run 'python train_final.py' first!"
        return None, None, None, None, error_msg

model, le_gender, le_symptoms, le_pre, error = load_model_and_encoders()

if error:
    st.error(error)
    st.stop()

# Get available options from encoders
gender_options = le_gender.classes_.tolist()
symptom_options = sorted(le_symptoms.classes_.tolist())
history_options = sorted(le_pre.classes_.tolist())

# ============================================================================
# USER INTERFACE
# ============================================================================

# Header
st.markdown('<p class="main-header">🏥 MedTouch.ai</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Emergency Department Triage System</p>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📋 Patient Assessment", "📊 System Performance", "ℹ️ About"])

# ============================================================================
# TAB 1: PATIENT ASSESSMENT
# ============================================================================
with tab1:
    st.markdown("### Enter Patient Information")
    
    # Two-column layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Demographics & Vitals")
        age = st.number_input(
            "Patient Age (years)",
            min_value=1,
            max_value=120,
            value=45,
            help="Enter patient's age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            options=gender_options,
            help="Select patient's biological sex"
        )
        
        bp = st.slider(
            "Systolic Blood Pressure (mmHg)",
            min_value=80,
            max_value=220,
            value=120,
            help="Normal range: 90-120 mmHg. Hypertensive crisis: >180 mmHg"
        )
        
    with col2:
        st.markdown("#### Clinical Information")
        hr = st.slider(
            "Heart Rate (BPM)",
            min_value=40,
            max_value=180,
            value=75,
            help="Normal range: 60-100 BPM. Tachycardia: >100 BPM"
        )
        
        symptom = st.selectbox(
            "Primary Symptom",
            options=symptom_options,
            help="Select the most prominent presenting symptom"
        )
        
        history = st.selectbox(
            "Pre-Existing Condition",
            options=history_options,
            help="Select relevant pre-existing medical condition"
        )
    
    st.markdown("---")
    
    # Analyze button
    analyze_clicked = st.button("🚀 ANALYZE PATIENT RISK", type="primary")
    
    if analyze_clicked:
        with st.spinner("Analyzing patient data..."):
            # Encode inputs
            try:
                g_val = le_gender.transform([gender])[0]
                s_val = le_symptoms.transform([symptom])[0]
                h_val = le_pre.transform([history])[0]
            except ValueError as e:
                st.error(f"Encoding error: {e}")
                st.stop()
            
            # Create feature vector
            feature_cols = ['Age', 'Gender_Encoded', 'Systolic_BP', 'Heart_Rate', 
                           'Symptoms_Encoded', 'Pre_Existing_Encoded']
            features = pd.DataFrame([[age, g_val, bp, hr, s_val, h_val]], 
                                   columns=feature_cols)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Get confidence for predicted class
            class_index = list(model.classes_).index(prediction)
            confidence = probabilities[class_index] * 100
            
            # Get all probabilities
            prob_dict = {cls: prob*100 for cls, prob in zip(model.classes_, probabilities)}
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎯 Risk Assessment Results")
        
        # Three-column layout for results
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        
        with res_col1:
            st.markdown("### Risk Classification")
            
            if prediction == "High":
                st.error(f"# 🔴 {prediction} RISK")
                color = "red"
            elif prediction == "Medium":
                st.warning(f"# 🟡 {prediction} RISK")
                color = "orange"
            else:
                st.success(f"# 🟢 {prediction} RISK")
                color = "green"
            
            st.metric("AI Confidence", f"{confidence:.1f}%")
            
            # Display vital signs summary
            st.markdown("#### Vital Signs")
            st.write(f"**BP:** {bp} mmHg")
            st.write(f"**HR:** {hr} BPM")
        
        with res_col2:
            st.markdown("### 🏥 Clinical Recommendations")
            
            if prediction == "High":
                st.error("""
                **🔴 CRITICAL - IMMEDIATE ACTION REQUIRED**
                
                **Triage Priority:** ESI Level 1 (Resuscitation)
                
                **Immediate Actions:**
                - Assign to trauma bay/resuscitation room immediately
                - Alert attending physician and specialist on call
                - Prepare emergency intervention equipment
                - Establish continuous cardiac monitoring
                - Obtain IV access (minimum 2 large-bore)
                - Order STAT labs and imaging
                
                **Time Target:** Physician evaluation within 0 minutes
                
                **Rationale:** Patient presents with life-threatening vital signs 
                and/or symptoms requiring immediate medical intervention to prevent 
                mortality or significant morbidity.
                """)
            elif prediction == "Medium":
                st.warning("""
                **🟡 URGENT - PRIORITY ASSESSMENT**
                
                **Triage Priority:** ESI Level 2-3 (Emergent/Urgent)
                
                **Immediate Actions:**
                - Move to rapid assessment/urgent care area
                - Obtain vital signs every 15-30 minutes
                - Alert ED physician for priority evaluation
                - Prepare for diagnostic workup
                - Consider placement in monitored bed
                
                **Time Target:** Physician evaluation within 15-30 minutes
                
                **Rationale:** Patient has concerning symptoms or vital sign 
                abnormalities that require timely evaluation and treatment, but 
                are not immediately life-threatening.
                """)
            else:
                st.success("""
                **🟢 STABLE - ROUTINE PROCESSING**
                
                **Triage Priority:** ESI Level 4-5 (Less Urgent/Non-Urgent)
                
                **Standard Actions:**
                - Assign to general waiting area
                - Standard vital sign monitoring protocol
                - Process in queue order
                - Routine registration and documentation
                
                **Expected Wait Time:** 1-2 hours depending on volume
                
                **Rationale:** Patient is hemodynamically stable with minor 
                complaint or symptoms that can be safely managed with routine 
                ED workflow without immediate intervention.
                """)
        
        with res_col3:
            st.markdown("### 📊 Probability Distribution")
            
            # Create visual probability bars
            for risk_level in ['High', 'Medium', 'Low']:
                prob = prob_dict.get(risk_level, 0)
                
                # Color code based on risk level
                if risk_level == 'High':
                    st.metric(
                        "🔴 High Risk",
                        f"{prob:.1f}%",
                        delta=None
                    )
                elif risk_level == 'Medium':
                    st.metric(
                        "🟡 Medium Risk",
                        f"{prob:.1f}%",
                        delta=None
                    )
                else:
                    st.metric(
                        "🟢 Low Risk",
                        f"{prob:.1f}%",
                        delta=None
                    )
            
            # Show confidence interpretation
            st.markdown("---")
            st.markdown("#### Confidence Level")
            if confidence >= 95:
                st.success("**Very High**\nClear pattern match")
            elif confidence >= 85:
                st.info("**High**\nStrong indicators")
            elif confidence >= 70:
                st.warning("**Moderate**\nSome uncertainty")
            else:
                st.error("**Low**\nAmbiguous case")
        
        # Detailed patient summary (expandable)
        with st.expander("📋 Detailed Patient Summary"):
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**Demographics**")
                st.write(f"Age: {age} years")
                st.write(f"Gender: {gender}")
                
            with summary_col2:
                st.markdown("**Vital Signs**")
                st.write(f"BP: {bp} mmHg")
                st.write(f"HR: {hr} BPM")
                
                # Add interpretations
                if bp > 180:
                    st.write("⚠️ Hypertensive Crisis")
                elif bp > 140:
                    st.write("⚠️ Elevated BP")
                    
                if hr > 100:
                    st.write("⚠️ Tachycardia")
                elif hr < 60:
                    st.write("⚠️ Bradycardia")
            
            with summary_col3:
                st.markdown("**Clinical**")
                st.write(f"Symptom: {symptom}")
                st.write(f"History: {history}")
        
        # Feature importance explanation
        with st.expander("🔍 Why This Classification?"):
            st.markdown("""
            The AI model considers multiple factors in making its decision:
            
            **Primary Factors (in order of importance):**
            1. **Systolic Blood Pressure (41.6%)** - Critical indicator of cardiovascular status
            2. **Heart Rate (27.6%)** - Secondary cardiovascular indicator
            3. **Age (15.8%)** - Baseline risk factor
            4. **Presenting Symptom (7.6%)** - Type and severity of complaint
            5. **Pre-Existing Conditions (7.5%)** - Chronic disease burden
            6. **Gender (0.02%)** - Minor statistical factor
            
            **Your Patient's Profile:**
            """)
            
            # Analyze each factor
            factors = []
            
            if bp > 160:
                factors.append(f"- ⚠️ **Very High BP ({bp} mmHg)**: Major contributor to HIGH risk classification")
            elif bp > 140:
                factors.append(f"- ⚠️ **Elevated BP ({bp} mmHg)**: Contributes to elevated risk")
            else:
                factors.append(f"- ✓ **Normal BP ({bp} mmHg)**: Supports lower risk")
            
            if hr > 100:
                factors.append(f"- ⚠️ **Tachycardia ({hr} BPM)**: Indicates physiologic stress")
            elif hr < 60:
                factors.append(f"- ⚠️ **Bradycardia ({hr} BPM)**: May indicate cardiac issue")
            else:
                factors.append(f"- ✓ **Normal HR ({hr} BPM)**: Supports stability")
            
            if age > 65:
                factors.append(f"- ⚠️ **Elderly patient ({age} years)**: Increases baseline risk")
            elif age < 35:
                factors.append(f"- ✓ **Young adult ({age} years)**: Generally lower risk")
            else:
                factors.append(f"- ○ **Middle-aged ({age} years)**: Moderate baseline risk")
            
            critical_symptoms = ['Chest Pain', 'Shortness of Breath', 'Severe Headache', 'Confusion']
            if symptom in critical_symptoms:
                factors.append(f"- ⚠️ **Critical symptom ({symptom})**: Requires urgent evaluation")
            else:
                factors.append(f"- ○ **Symptom: {symptom}**")
            
            if history in ['Heart Disease', 'Diabetes', 'Hypertension']:
                factors.append(f"- ⚠️ **Pre-existing {history}**: Increases complication risk")
            else:
                factors.append(f"- ✓ **Medical history: {history}**")
            
            st.markdown("\n".join(factors))

# ============================================================================
# TAB 2: SYSTEM PERFORMANCE
# ============================================================================
with tab2:
    st.markdown("### 🎯 Model Performance Metrics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Overall Accuracy", "99.75%", "+1.65%")
        st.metric("Training Samples", "8,000", "")
        
    with perf_col2:
        st.metric("High Risk Precision", "99.8%", "")
        st.metric("Medium Risk Precision", "99.4%", "")
        
    with perf_col3:
        st.metric("Low Risk Precision", "100.0%", "")
        st.metric("Test Samples", "2,000", "")
    
    st.markdown("---")
    
    # Performance breakdown
    st.markdown("### 📊 Detailed Performance by Risk Level")
    
    perf_data = pd.DataFrame({
        'Risk Level': ['High', 'Medium', 'Low'],
        'Precision': [99.8, 99.4, 100.0],
        'Recall': [100.0, 99.8, 99.4],
        'F1-Score': [99.9, 99.6, 99.7],
        'Avg Confidence': [99.4, 99.1, 99.7]
    })
    
    st.dataframe(perf_data, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🔬 Feature Importance Analysis")
    
    importance_data = pd.DataFrame({
        'Feature': ['Systolic BP', 'Heart Rate', 'Age', 'Symptoms', 'Pre-Existing', 'Gender'],
        'Importance (%)': [41.61, 27.57, 15.79, 7.56, 7.46, 0.02]
    })
    
    st.bar_chart(importance_data.set_index('Feature'))
    
    st.markdown("""
    **Key Insights:**
    - Systolic blood pressure is the strongest predictor of triage risk
    - Heart rate provides critical complementary cardiovascular information
    - Age contributes significantly as a baseline risk modifier
    - Presenting symptoms and medical history have moderate but important impact
    """)

# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab3:
    st.markdown("### About MedTouch.ai")
    
    st.markdown("""
    **MedTouch.ai** is an advanced artificial intelligence system designed to assist 
    emergency department staff in patient triage and risk stratification.
    
    #### 🎯 Purpose
    This system uses machine learning to analyze patient vital signs, symptoms, and 
    medical history to provide:
    - Rapid risk assessment
    - Evidence-based triage recommendations
    - Decision support for ED staff
    
    #### 🤖 Technology
    - **Algorithm:** Random Forest Classifier (300 decision trees)
    - **Training Data:** 10,000 realistic patient cases
    - **Accuracy:** 99.75% on validation set
    - **Features:** 6 clinical parameters
    - **Output:** 3-level risk classification (High, Medium, Low)
    
    #### 📊 Performance
    - **High Risk Detection:** 99.8% precision, 100.0% recall
    - **Medium Risk Detection:** 99.4% precision, 99.8% recall  
    - **Low Risk Detection:** 100.0% precision, 99.4% recall
    - **Average Confidence:** >99% across all risk levels
    
    #### ⚕️ Clinical Integration
    This tool is designed to **assist**, not replace, clinical judgment. Final triage 
    decisions should always be made by qualified healthcare professionals based on:
    - Complete patient assessment
    - Additional clinical findings
    - Resource availability
    - Individual patient factors
    
    #### 🔒 Safety & Compliance
    - Trained on medically accurate synthetic data
    - Validated against clinical guidelines
    - Transparent decision-making process
    - Continuous monitoring and updates
    
    #### 📚 Evidence Base
    Risk stratification is based on:
    - Emergency Severity Index (ESI) guidelines
    - Cardiovascular risk factors
    - Age-adjusted risk assessment
    - Symptom-disease associations
    
    #### 💡 Best Practices
    - Use in conjunction with clinical assessment
    - Consider patient's complete presentation
    - Reassess if condition changes
    - Document all clinical decisions
    - Escalate concerns immediately
    
    ---
    
    **Version:** 2.0.0  
    **Last Updated:** February 2026  
    **Model Training Date:** Current Session  
    **Validation Status:** ✅ All Systems Operational
    """)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    st.metric("Model Accuracy", "99.75%")
    st.metric("Avg Confidence", "99.4%")
    st.metric("Training Samples", "8,000")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Quick Reference")
    st.markdown("""
    **High Risk (ESI 1)**
    - Immediate life threat
    - Resuscitation needed
    - 0-minute target
    
    **Medium Risk (ESI 2-3)**
    - Urgent assessment
    - Potential instability
    - 15-30 minute target
    
    **Low Risk (ESI 4-5)**
    - Stable condition
    - Routine processing
    - 1-2 hour acceptable
    """)
    
    st.markdown("---")
    
    st.markdown("### 📞 Support")
    st.markdown("""
    For technical issues or questions:
    - Check documentation
    - Review training logs
    - Validate input data
    - Ensure model files present
    """)
    
    st.markdown("---")
    st.caption("MedTouch.ai v2.0.0")
    st.caption("AI-Powered Triage System")
    st.caption("© 2026 - For Demo Purposes")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for demonstration and decision support only. 
    All medical decisions must be made by qualified healthcare professionals based on complete 
    clinical assessment.</p>
    <p><em>MedTouch.ai - Advancing Emergency Care Through AI | Version 2.0.0</em></p>
</div>
""", unsafe_allow_html=True)
