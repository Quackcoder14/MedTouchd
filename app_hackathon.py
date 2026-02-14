"""
HACKATHON AI TRIAGE SYSTEM - MULTI-STEP STREAMLIT APP
======================================================
Enhanced version with AI-powered document analysis and extraction
Step-by-step patient intake:
1. Vitals
2. Symptoms
3. History (with intelligent document upload & analysis)
4. Review & Results
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import base64
import io
import re
from PIL import Image
import pytesseract

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MedTouch.ai Patient Intake",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS with powder blue gradient theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background - Powder blue gradient */
    .stApp {
        background: linear-gradient(180deg, 
            #B8D8E8 0%,    /* Powder blue */
            #D4E8F0 30%,   /* Light powder blue */
            #E8F3F8 60%,   /* Very light powder blue */
            #F0F7FA 100%   /* Almost white with blue tint */
        );
        background-attachment: fixed;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1B3A52 !important;  /* Deep navy blue */
        font-weight: 700 !important;
        animation: fadeInDown 0.6s ease-out;
    }
    
    /* Regular text */
    p, div, span, label {
        color: #3C4043 !important;  /* Charcoal gray */
    }
    
    /* Progress stepper - Enhanced */
    .stepper {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(27, 58, 82, 0.08);
        backdrop-filter: blur(10px);
        animation: slideInDown 0.5s ease-out;
    }
    
    .step {
        flex: 1;
        text-align: center;
        font-weight: 600;
        font-size: 1.05rem;
        color: #9CA3AF;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    .step::before {
        content: '';
        position: absolute;
        top: -2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 40px;
        height: 40px;
        background: #E5E7EB;
        border-radius: 50%;
        transition: all 0.4s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .step-active {
        color: #1B3A52;  /* Deep navy blue */
        transform: scale(1.05);
        font-weight: 700;
        background: rgba(184, 216, 232, 0.2);  /* Powder blue tint */
    }
    
    .step-active::before {
        background: linear-gradient(135deg, #1B3A52, #2D5F7F);
        box-shadow: 0 4px 15px rgba(27, 58, 82, 0.3);
        transform: translateX(-50%) scale(1.1);
    }
    
    .step-active::after {
        content: '';
        position: absolute;
        bottom: -1rem;
        left: 50%;
        transform: translateX(-50%);
        width: 70%;
        height: 4px;
        background: linear-gradient(90deg, #1B3A52, #4A90E2);
        border-radius: 2px;
        animation: expandWidth 0.4s ease-out;
    }
    
    /* Card containers */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(27, 58, 82, 0.08);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(27, 58, 82, 0.12);
    }
    
    /* Risk level cards - Enhanced */
    .risk-high {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(220, 38, 38, 0.3);
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #F59E0B, #FBBF24);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.3);
        animation: fadeInScale 0.6s ease-out;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3);
        animation: fadeInScale 0.6s ease-out;
    }
    
    /* Factor box */
    .factor-box {
        background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1B3A52;
        font-size: 0.95rem;
        color: #3C4043;
        transition: all 0.3s ease;
        animation: slideInLeft 0.5s ease-out;
    }
    
    .factor-box:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(27, 58, 82, 0.1);
    }
    
    /* Department box */
    .dept-box {
        background: linear-gradient(135deg, #1B3A52, #2D5F7F);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(27, 58, 82, 0.25);
        animation: fadeInScale 0.6s ease-out;
    }
    
    /* Section headers */
    .section-header {
        color: #1B3A52 !important;  /* Deep navy blue */
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    .section-subheader {
        color: #6B7280 !important;
        font-size: 1rem;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid rgba(184, 216, 232, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(27, 58, 82, 0.1);
    }
    
    /* Slider styling - Powder blue theme */
    .stSlider > div > div > div > div {
        background-color: #1B3A52 !important;  /* Deep navy blue */
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #B8D8E8, #1B3A52) !important;
    }
    
    /* Buttons - Enhanced with light colors for visibility */
    .stButton > button {
        background: linear-gradient(135deg, #D4E8F0, #B8D8E8) !important;
        color: #1B3A52 !important;  /* Dark text on light background */
        border: 2px solid #1B3A52 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(27, 58, 82, 0.15) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(27, 58, 82, 0.25) !important;
        background: linear-gradient(135deg, #B8D8E8, #A0C8DC) !important;
        border-color: #2D5F7F !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        border: 2px dashed #1B3A52 !important;
        padding: 1rem !important;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed #B8D8E8;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1B3A52;
        background: rgba(184, 216, 232, 0.1);
    }
    
    /* Info tooltip */
    .info-tooltip {
        display: inline-block;
        position: relative;
        margin-left: 8px;
        cursor: help;
    }
    
    .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #4A90E2, #1B3A52);
        color: white;
        border-radius: 50%;
        font-size: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .info-icon:hover {
        transform: scale(1.2);
        box-shadow: 0 4px 12px rgba(27, 58, 82, 0.3);
    }
    
    .tooltip-text {
        visibility: hidden;
        width: 280px;
        background-color: #1B3A52;
        color: white;
        text-align: left;
        border-radius: 10px;
        padding: 12px 16px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.3s, visibility 0.3s;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #1B3A52 transparent transparent transparent;
    }
    
    .info-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* Document upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed #B8D8E8;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.7s ease-out;
    }
    
    .upload-section:hover {
        border-color: #1B3A52;
        background: rgba(184, 216, 232, 0.1);
        transform: translateY(-2px);
    }
    
    .upload-icon {
        text-align: center;
        font-size: 3rem;
        color: #1B3A52;
        margin-bottom: 1rem;
        animation: bounce 2s ease-in-out infinite;
    }
    
    /* Extracted data display */
    .extracted-data {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: slideInRight 0.5s ease-out;
    }
    
    .extracted-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .extracted-item:last-child {
        border-bottom: none;
    }
    
    /* Processing indicator */
    .processing {
        text-align: center;
        padding: 2rem;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 0.5rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stCheckbox:hover {
        background: rgba(184, 216, 232, 0.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 8px 30px rgba(220, 38, 38, 0.3);
        }
        50% {
            box-shadow: 0 8px 40px rgba(220, 38, 38, 0.5);
        }
    }
    
    @keyframes expandWidth {
        from { width: 0%; }
        to { width: 70%; }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px !important;
        border: 1px solid #B8D8E8 !important;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px !important;
        animation: fadeInScale 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_image(image_file):
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        content = txt_file.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        else:
            text = content
        return text
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF - simplified version"""
    try:
        # For demonstration, return a message
        # In production, you would use PyPDF2 or pdfplumber
        return "PDF text extraction requires PyPDF2 or pdfplumber library. For demo purposes, please use TXT or image files, or manually enter the condition."
    except Exception as e:
        return f"Error extracting from PDF: {str(e)}"

def analyze_medical_document(text, available_conditions):
    """
    Analyze medical document text and extract relevant medical history
    using pattern matching and keyword detection
    """
    text_lower = text.lower()
    
    # Medical condition keywords mapping
    condition_keywords = {
        'Diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'blood sugar', 'glucose', 'insulin', 'type 1', 'type 2', 'dm', 't1d', 't2d'],
        'Hypertension': ['hypertension', 'high blood pressure', 'htn', 'bp', 'elevated blood pressure', 'hypertensive'],
        'Asthma': ['asthma', 'asthmatic', 'bronchospasm', 'wheezing', 'inhaler', 'bronchodilator'],
        'Heart Disease': ['heart disease', 'cardiac', 'coronary', 'chd', 'cvd', 'myocardial', 'angina', 'heart attack', 'mi', 'cardiomyopathy'],
        'Kidney Disease': ['kidney disease', 'renal', 'ckd', 'nephropathy', 'dialysis', 'kidney failure', 'renal failure'],
        'Cancer': ['cancer', 'carcinoma', 'tumor', 'tumour', 'malignancy', 'oncology', 'chemotherapy', 'radiation therapy', 'neoplasm'],
        'Stroke': ['stroke', 'cva', 'cerebrovascular', 'brain attack', 'tia', 'transient ischemic'],
        'COPD': ['copd', 'chronic obstructive', 'emphysema', 'chronic bronchitis'],
        'Arthritis': ['arthritis', 'rheumatoid', 'osteoarthritis', 'joint pain', 'ra'],
        'Depression': ['depression', 'depressive', 'major depressive disorder', 'mdd', 'antidepressant'],
        'Anxiety': ['anxiety', 'anxious', 'panic', 'gad', 'generalized anxiety'],
        'Obesity': ['obesity', 'obese', 'overweight', 'bmi', 'body mass index'],
        'Thyroid Disease': ['thyroid', 'hypothyroid', 'hyperthyroid', 'goiter', 'thyroiditis'],
        'Liver Disease': ['liver disease', 'hepatic', 'cirrhosis', 'hepatitis', 'liver failure'],
    }
    
    detected_conditions = []
    confidence_scores = {}
    
    # Search for conditions in the text
    for condition in available_conditions:
        if condition == 'No History':
            continue
            
        # Check if condition or its keywords are in the text
        condition_found = False
        match_count = 0
        
        # Direct match
        if condition.lower() in text_lower:
            condition_found = True
            match_count += 2
        
        # Keyword match
        if condition in condition_keywords:
            for keyword in condition_keywords[condition]:
                if keyword in text_lower:
                    condition_found = True
                    match_count += 1
        
        if condition_found:
            detected_conditions.append(condition)
            confidence_scores[condition] = min(match_count * 20, 100)  # Cap at 100%
    
    # Extract potential vitals from document
    vitals_data = extract_vitals_from_text(text)
    
    return {
        'conditions': detected_conditions,
        'confidence': confidence_scores,
        'vitals': vitals_data,
        'raw_text': text[:500]  # First 500 chars for preview
    }

def extract_vitals_from_text(text):
    """Extract vital signs from medical document text"""
    vitals = {}
    
    # Blood pressure pattern (e.g., 120/80, 140/90)
    bp_pattern = r'(\d{2,3})\s*/\s*(\d{2,3})'
    bp_matches = re.findall(bp_pattern, text)
    if bp_matches:
        systolic, diastolic = bp_matches[0]
        vitals['systolic_bp'] = int(systolic)
        vitals['diastolic_bp'] = int(diastolic)
    
    # Heart rate pattern
    hr_patterns = [
        r'heart rate[:\s]+(\d{2,3})',
        r'hr[:\s]+(\d{2,3})',
        r'pulse[:\s]+(\d{2,3})',
        r'(\d{2,3})\s*bpm'
    ]
    for pattern in hr_patterns:
        match = re.search(pattern, text.lower())
        if match:
            vitals['heart_rate'] = int(match.group(1))
            break
    
    # Temperature pattern
    temp_patterns = [
        r'temperature[:\s]+(\d{2,3}\.?\d*)',
        r'temp[:\s]+(\d{2,3}\.?\d*)',
        r'(\d{2}\.?\d*)\s*°?[cf]'
    ]
    for pattern in temp_patterns:
        match = re.search(pattern, text.lower())
        if match:
            temp = float(match.group(1))
            # Convert if necessary (assume Fahrenheit if > 45)
            if temp > 45:
                temp = (temp - 32) * 5/9
            vitals['temperature'] = round(temp, 1)
            break
    
    # Age pattern
    age_patterns = [
        r'age[:\s]+(\d{1,3})',
        r'(\d{1,3})\s*years?\s*old',
        r'(\d{1,3})\s*y/?o'
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text.lower())
        if match:
            vitals['age'] = int(match.group(1))
            break
    
    # Gender pattern
    if re.search(r'\b(male|man|m)\b', text.lower()) and not re.search(r'\b(female|woman|f)\b', text.lower()):
        vitals['gender'] = 'Male'
    elif re.search(r'\b(female|woman|f)\b', text.lower()):
        vitals['gender'] = 'Female'
    
    return vitals

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
        'pre_existing': 'No History',
        'uploaded_document': None,
        'document_name': None,
        'extracted_data': None
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
        'pre_existing': 'No History',
        'uploaded_document': None,
        'document_name': None,
        'extracted_data': None
    }

def info_icon(tooltip_text):
    """Create an info icon with tooltip"""
    return f'''
    <span class="info-tooltip">
        <span class="info-icon">i</span>
        <span class="tooltip-text">{tooltip_text}</span>
    </span>
    '''

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
        factors.append(f"🔴 Tachycardia ({hr} BPM)")
    elif hr > 100:
        factors.append(f"⚠️ Elevated heart rate ({hr} BPM)")
    elif hr < 50:
        factors.append(f"⚠️ Bradycardia ({hr} BPM)")
    else:
        factors.append(f"✅ Normal heart rate ({hr} BPM)")
    
    temp = patient_data['temperature']
    if temp > 38.5:
        factors.append(f"🔴 High fever ({temp}°C)")
    elif temp > 37.5:
        factors.append(f"⚠️ Mild fever ({temp}°C)")
    elif temp < 36.0:
        factors.append(f"⚠️ Hypothermia ({temp}°C)")
    else:
        factors.append(f"✅ Normal temperature ({temp}°C)")
    
    if patient_data['pre_existing'] != 'No History':
        factors.append(f"⚠️ Pre-existing condition: {patient_data['pre_existing']}")
    
    symptom_count = len(patient_data['symptoms'])
    if symptom_count >= 4:
        factors.append(f"🔴 Multiple symptoms present ({symptom_count} symptoms)")
    elif symptom_count >= 2:
        factors.append(f"⚠️ Several symptoms reported ({symptom_count} symptoms)")
    
    # Check for specific high-risk symptoms
    high_risk_symptoms = ['Chest Pain', 'Difficulty Breathing', 'Seizures', 'Unconsciousness']
    found_high_risk = [s for s in high_risk_symptoms if s in patient_data['symptoms']]
    if found_high_risk:
        factors.append(f"🔴 Critical symptoms: {', '.join(found_high_risk)}")
    
    return factors

def make_prediction(patient_data):
    """Make risk and department predictions"""
    try:
        # Prepare features
        gender_encoded = models['le_gender'].transform([patient_data['gender']])[0]
        
        # Handle symptoms
        symptoms_list = patient_data['symptoms']
        if not symptoms_list:
            return {'error': 'No symptoms selected'}
        
        primary_symptom = symptoms_list[0]
        symptom_encoded = models['le_symptoms'].transform([primary_symptom])[0]
        
        # Handle pre-existing condition
        pre_existing = patient_data['pre_existing']
        pre_existing_encoded = models['le_pre_existing'].transform([pre_existing])[0]
        
        # Create feature vector
        X = np.array([[
            patient_data['age'],
            gender_encoded,
            patient_data['systolic_bp'],
            patient_data['diastolic_bp'],
            patient_data['heart_rate'],
            patient_data['temperature'],
            symptom_encoded,
            pre_existing_encoded
        ]])
        
        # Make predictions
        risk_pred = models['risk_model'].predict(X)[0]
        risk_proba = models['risk_model'].predict_proba(X)[0]
        
        dept_pred = models['dept_model'].predict(X)[0]
        dept_proba = models['dept_model'].predict_proba(X)[0]
        
        # Get class labels
        risk_classes = models['risk_model'].classes_
        dept_classes = models['dept_model'].classes_
        
        # Build result
        result = {
            'risk': risk_pred,
            'risk_confidence': max(risk_proba) * 100,
            'risk_probs': {cls: prob * 100 for cls, prob in zip(risk_classes, risk_proba)},
            'department': dept_pred,
            'dept_confidence': max(dept_proba) * 100,
            'dept_probs': {cls: prob * 100 for cls, prob in zip(dept_classes, dept_proba)},
            'factors': explain_prediction(patient_data, risk_pred)
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 style="text-align: center; margin-bottom: 0.5rem;">🏥 MedTouch.ai Patient Intake</h1>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem;">AI-Powered Medical Triage System with Document Analysis</p>', 
            unsafe_allow_html=True)

# ============================================================================
# PROGRESS STEPPER
# ============================================================================
steps = ["Vitals", "Symptoms", "History", "Results"]
stepper_html = '<div class="stepper">'
for i, step_name in enumerate(steps, 1):
    active_class = "step-active" if i == st.session_state.step else ""
    stepper_html += f'<div class="step {active_class}">{step_name}</div>'
stepper_html += '</div>'
st.markdown(stepper_html, unsafe_allow_html=True)

# ============================================================================
# STEP 1: VITALS
# ============================================================================
if st.session_state.step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Patient Vitals</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Enter the patient\'s vital signs and basic information</p>', unsafe_allow_html=True)
    
    # Demographics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('**Age (years)** ' + info_icon(
            "Patient's age in years. Pediatric: <18, Adult: 18-65, Geriatric: >65"
        ), unsafe_allow_html=True)
        st.session_state.form_data['age'] = st.slider(
            "Age",
            min_value=0,
            max_value=120,
            value=st.session_state.form_data['age'],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown('**Gender**', unsafe_allow_html=True)
        st.session_state.form_data['gender'] = st.selectbox(
            "Gender",
            options=['Male', 'Female'],
            index=0 if st.session_state.form_data['gender'] == 'Male' else 1,
            label_visibility="collapsed"
        )
    
    st.markdown("")
    
    # Blood Pressure
    st.markdown('**Blood Pressure (mmHg)** ' + info_icon(
        "Normal: 120/80 mmHg | Elevated: 120-129/<80 | High Stage 1: 130-139/80-89 | High Stage 2: ≥140/≥90 | Hypertensive Crisis: >180/>120"
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.form_data['systolic_bp'] = st.slider(
            "Systolic (Upper)",
            min_value=60,
            max_value=220,
            value=st.session_state.form_data['systolic_bp'],
            help="Upper number - pressure when heart beats"
        )
    
    with col2:
        st.session_state.form_data['diastolic_bp'] = st.slider(
            "Diastolic (Lower)",
            min_value=40,
            max_value=140,
            value=st.session_state.form_data['diastolic_bp'],
            help="Lower number - pressure between beats"
        )
    
    st.markdown("")
    
    # Heart Rate
    st.markdown('**Heart Rate (BPM)** ' + info_icon(
        "Normal resting: 60-100 BPM | Bradycardia: <60 BPM | Tachycardia: >100 BPM | Athletes may have lower resting heart rate (40-60 BPM)"
    ), unsafe_allow_html=True)
    
    st.session_state.form_data['heart_rate'] = st.slider(
        "Heart Rate",
        min_value=30,
        max_value=200,
        value=st.session_state.form_data['heart_rate'],
        label_visibility="collapsed"
    )
    
    st.markdown("")
    
    # Temperature
    st.markdown('**Body Temperature (°C)** ' + info_icon(
        "Normal: 36.1-37.2°C (97-99°F) | Low-grade fever: 37.3-38.0°C | Fever: 38.1-39.0°C | High fever: >39.0°C | Hypothermia: <35.0°C"
    ), unsafe_allow_html=True)
    
    st.session_state.form_data['temperature'] = st.slider(
        "Temperature",
        min_value=34.0,
        max_value=42.0,
        value=st.session_state.form_data['temperature'],
        step=0.1,
        format="%.1f",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Continue button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Continue →", key="vitals_continue", use_container_width=True):
            next_step()
            st.rerun()

# ============================================================================
# STEP 2: SYMPTOMS
# ============================================================================
elif st.session_state.step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🩺 Select Symptoms</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Choose all symptoms the patient is experiencing</p>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous", key="symptoms_prev", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue →", key="symptoms_next", use_container_width=True):
            if not st.session_state.form_data['symptoms']:
                st.warning("⚠️ Please select at least one symptom")
            else:
                next_step()
                st.rerun()

# ============================================================================
# STEP 3: HISTORY (WITH INTELLIGENT DOCUMENT UPLOAD)
# ============================================================================
elif st.session_state.step == 3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📋 Medical History</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Upload health document for automatic extraction or select manually</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Document Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-icon">🤖</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header" style="text-align: center; margin-top: 0;">AI-Powered Document Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280; margin-bottom: 1.5rem;">Upload patient\'s EHR/EMR document - AI will automatically extract medical history</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'jpg', 'jpeg', 'png'],
        help="Supported formats: PDF, TXT, JPG, PNG",
        label_visibility="collapsed",
        key="doc_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state.form_data['uploaded_document'] = uploaded_file
        st.session_state.form_data['document_name'] = uploaded_file.name
        
        # Display uploaded file info
        st.success(f"✅ Document uploaded successfully: **{uploaded_file.name}**")
        
        # Process the document
        with st.spinner("🔍 Analyzing document with AI..."):
            # Extract text based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                extracted_text = extract_text_from_image(uploaded_file)
            elif file_extension == 'txt':
                extracted_text = extract_text_from_txt(uploaded_file)
            elif file_extension == 'pdf':
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = "Unsupported file format"
            
            # Analyze the extracted text
            available_conditions = sorted(models['le_pre_existing'].classes_)
            analysis_result = analyze_medical_document(extracted_text, available_conditions)
            
            st.session_state.form_data['extracted_data'] = analysis_result
        
        # Display extracted information
        if analysis_result and analysis_result['conditions']:
            st.markdown('<div class="extracted-data">', unsafe_allow_html=True)
            st.markdown("### 🎯 Extracted Medical Information")
            
            st.markdown("**Detected Conditions:**")
            for condition in analysis_result['conditions']:
                confidence = analysis_result['confidence'].get(condition, 0)
                st.markdown(f'<div class="extracted-item">✓ <strong>{condition}</strong> (Confidence: {confidence}%)</div>', 
                           unsafe_allow_html=True)
            
            # Auto-select the most confident condition
            if analysis_result['conditions']:
                best_condition = max(analysis_result['conditions'], 
                                   key=lambda x: analysis_result['confidence'].get(x, 0))
                st.session_state.form_data['pre_existing'] = best_condition
                st.info(f"✨ Auto-selected: **{best_condition}** (Highest confidence)")
            
            # Display extracted vitals if any
            if analysis_result['vitals']:
                st.markdown("**Extracted Vitals:**")
                for vital_name, vital_value in analysis_result['vitals'].items():
                    st.markdown(f'<div class="extracted-item">📊 {vital_name.replace("_", " ").title()}: <strong>{vital_value}</strong></div>', 
                               unsafe_allow_html=True)
                    
                    # Auto-populate vitals
                    if vital_name in st.session_state.form_data:
                        st.session_state.form_data[vital_name] = vital_value
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Option to apply extracted data
            if st.button("🔄 Apply Extracted Data to Form", key="apply_data", use_container_width=True):
                st.success("✅ Extracted data applied! You can review in Step 1 (Vitals)")
        
        elif analysis_result:
            st.warning("⚠️ No medical conditions detected in the document. Please select manually below.")
        
    elif st.session_state.form_data['document_name']:
        st.info(f"📄 Previously uploaded: **{st.session_state.form_data['document_name']}**")
        if st.session_state.form_data['extracted_data']:
            st.info(f"✓ Detected: {', '.join(st.session_state.form_data['extracted_data']['conditions'][:3])}")
    
    st.markdown('<p style="text-align: center; color: #6B7280; font-size: 0.9rem; margin-top: 1rem;">💡 Drag and drop a file or click to browse</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Manual Selection Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Manual Selection")
    st.markdown("Review or manually select pre-existing condition:")
    
    # Get available conditions
    all_conditions = sorted(models['le_pre_existing'].classes_)
    
    st.session_state.form_data['pre_existing'] = st.selectbox(
        "Select Pre-Existing Condition",
        options=all_conditions,
        index=all_conditions.index(st.session_state.form_data['pre_existing']) 
              if st.session_state.form_data['pre_existing'] in all_conditions else 0,
        key="manual_condition"
    )
    
    st.info("💡 Select 'No History' if patient has no pre-existing conditions")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous", key="history_prev", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Analyze Patient →", key="history_analyze", use_container_width=True):
            next_step()
            st.rerun()

# ============================================================================
# STEP 4: REVIEW & RESULTS
# ============================================================================
elif st.session_state.step == 4:
    st.markdown('<p class="section-header" style="text-align: center;">🎯 Analysis Results</p>', unsafe_allow_html=True)
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
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Risk Level
            st.markdown('<p class="section-header">Risk Classification</p>', unsafe_allow_html=True)
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
            st.markdown('<p class="section-header">Recommended Department</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="dept-box">📍 {result["department"]}<br/>{result["dept_confidence"]:.1f}% Match</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("")
            
            # Probabilities
            st.markdown('<p class="section-header">Risk Probabilities</p>', unsafe_allow_html=True)
            for level in ['High', 'Medium', 'Low']:
                prob = result['risk_probs'].get(level, 0)
                icon = "🔴" if level == 'High' else "🟡" if level == 'Medium' else "🟢"
                st.metric(f"{icon} {level}", f"{prob:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Clinical Recommendations
            st.markdown('<p class="section-header">🏥 Clinical Recommendations</p>', unsafe_allow_html=True)
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Contributing Factors
        st.markdown("")
        
        col_factor, col_summary = st.columns(2)
        
        with col_factor:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">💡 Contributing Factors</p>', unsafe_allow_html=True)
            for factor in result['factors']:
                st.markdown(f'<div class="factor-box">{factor}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_summary:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📊 Patient Summary</p>', unsafe_allow_html=True)
            
            doc_source = "AI-Extracted" if st.session_state.form_data.get('extracted_data') else "Manual"
            
            summary_data = {
                'Field': ['Age', 'Gender', 'Blood Pressure', 'Heart Rate', 'Temperature', 
                         'Symptoms', 'Pre-Existing', 'Document', 'Data Source'],
                'Value': [
                    f"{st.session_state.form_data['age']} years",
                    st.session_state.form_data['gender'],
                    f"{st.session_state.form_data['systolic_bp']}/{st.session_state.form_data['diastolic_bp']} mmHg",
                    f"{st.session_state.form_data['heart_rate']} BPM",
                    f"{st.session_state.form_data['temperature']}°C",
                    ', '.join(st.session_state.form_data['symptoms'][:3]) + ('...' if len(st.session_state.form_data['symptoms']) > 3 else ''),
                    st.session_state.form_data['pre_existing'],
                    st.session_state.form_data['document_name'] if st.session_state.form_data['document_name'] else 'None',
                    doc_source
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("← Previous", key="results_prev", use_container_width=True):
                prev_step()
                st.rerun()
        
        with btn_col2:
            if st.button("🔄 New Patient", key="results_reset", use_container_width=True):
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
<div style='text-align: center; color: #6B7280; padding: 1.5rem; animation: fadeIn 1s ease-out;'>
    <p style='margin: 0; font-size: 0.95rem;'>
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for demonstration purposes only.
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        <em>MedTouch.ai v2.5 | AI-Powered Triage System with Document Intelligence</em>
    </p>
</div>
""", unsafe_allow_html=True)